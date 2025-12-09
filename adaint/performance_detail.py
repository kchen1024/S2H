import os
import argparse
import time
import json

import torch
import numpy as np
from mmcv.parallel import collate, scatter

from mmedit.apis import init_model
from torchprofile import profile_macs
from torch.profiler import profile, record_function, ProfilerActivity

def create_dummy_data(model, height=2160, width=3840, batch_size=1):
    """Create dummy input data for benchmarking"""
    cfg = model.cfg
    device = next(model.parameters()).device

    # Create dummy tensor
    dummy_tensor = torch.randn(batch_size, 3, height, width, dtype=torch.float32, device=device)

    # Create data dict in the format expected by the model
    data = {
        'lq': dummy_tensor,
        'meta': [{
            'lq_path': f'dummy_input_{height}x{width}',
            'key': f'dummy_{i}'
        } for i in range(batch_size)]
    }

    return data

def profile_model_detailed(model, input_data, output_dir="./profiling_results"):
    """?????????"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== Detailed Model Profiling ===")
    
    # 1. PyTorch Profiler ??
    print("Running PyTorch Profiler...")
    model.eval()
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                _ = model(test_mode=True, **input_data)
    
    # ?????profiling?? - ??????
    try:
        # ??????????
        prof_table = prof.key_averages().table(
            sort_by="cuda_time_total", 
            row_limit=50
        )
    except TypeError:
        # ????,??????
        prof_table = prof.key_averages().table(
            sort_by="cuda_time_total", 
            row_limit=50
        )
    
    print("\nTop 20 CUDA operations by time:")
    try:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    except Exception as e:
        print(f"Error displaying table: {e}")
        # ?????20???
        key_averages = prof.key_averages()
        sorted_events = sorted(key_averages, key=lambda x: x.cuda_time_total, reverse=True)
        print(f"{'Operation':<50} {'CUDA Time (ms)':<15} {'CPU Time (ms)':<15}")
        print("-" * 80)
        for i, event in enumerate(sorted_events[:20]):
            cuda_time = event.cuda_time_total / 1000  # ?????
            cpu_time = event.cpu_time_total / 1000
            print(f"{event.key[:49]:<50} {cuda_time:<15.3f} {cpu_time:<15.3f}")
    
    # ?????profiling??
    with open(os.path.join(output_dir, "profiling_table.txt"), "w") as f:
        f.write(prof_table)
    
    # ??Chrome trace??
    trace_file = os.path.join(output_dir, "trace.json")
    try:
        prof.export_chrome_trace(trace_file)
        print(f"Chrome trace saved to: {trace_file}")
    except Exception as e:
        print(f"Failed to export Chrome trace: {e}")
    
    # ??????
    stacks_file = os.path.join(output_dir, "stacks.txt")
    try:
        prof.export_stacks(stacks_file, "self_cuda_time_total")
        print(f"Stack traces saved to: {stacks_file}")
    except Exception as e:
        print(f"Failed to export stacks: {e}")
    
    return prof

def profile_layer_by_layer(model, input_data, num_runs=10):
    """??????"""
    print("\n=== Layer-by-Layer Profiling ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping layer-by-layer profiling")
        return {}, 0
    
    model.eval()
    
    # ??????????????
    def get_layer_times():
        # ????????
        leaf_modules = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0 and any(p.requires_grad for p in module.parameters()):
                leaf_modules.append((name, module))
        
        if not leaf_modules:
            print("No leaf modules found for profiling")
            return {}, 0
        
        # ??profiler???????
        layer_times = {}
        
        with profile(
            activities=[ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False
        ) as prof:
            with torch.no_grad():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                _ = model(test_mode=True, **input_data)
                end_event.record()
                
                torch.cuda.synchronize()
                total_time = start_event.elapsed_time(end_event)
        
        # ??profiler??
        events = prof.key_averages()
        for event in events:
            if event.cuda_time_total > 0:
                # ???????
                layer_name = event.key.split('.')[-1] if '.' in event.key else event.key
                cuda_time = event.cuda_time_total / 1000  # ?????
                
                if layer_name not in layer_times:
                    layer_times[layer_name] = []
                layer_times[layer_name].append(cuda_time)
        
        return layer_times, total_time
    
    # ???????
    all_layer_times = {}
    total_times = []
    
    print(f"Running {num_runs} iterations for layer timing...")
    for run in range(num_runs):
        try:
            layer_times_run, total_time = get_layer_times()
            total_times.append(total_time)
            
            for layer_name, times in layer_times_run.items():
                if layer_name not in all_layer_times:
                    all_layer_times[layer_name] = []
                all_layer_times[layer_name].extend(times)
            
            if (run + 1) % 5 == 0:
                print(f"Completed {run + 1}/{num_runs} runs")
        except Exception as e:
            print(f"Error in run {run + 1}: {e}")
            continue
    
    if not all_layer_times:
        print("No layer timing data collected")
        return {}, np.mean(total_times) if total_times else 0
    
    # ??????
    avg_layer_times = {}
    for layer_name, times in all_layer_times.items():
        if times:  # ???????
            avg_layer_times[layer_name] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'count': len(times)
            }
    
    # ????
    if avg_layer_times:
        print("\nLayer timing results (top 15 by average time):")
        sorted_layers = sorted(avg_layer_times.items(), 
                              key=lambda x: x[1]['mean'], reverse=True)
        
        print(f"{'Layer/Operation':<50} {'Mean (ms)':<12} {'Std (ms)':<12} {'Count':<8}")
        print("-" * 85)
        
        for layer_name, times in sorted_layers[:15]:
            print(f"{layer_name[:49]:<50} {times['mean']:<12.3f} {times['std']:<12.3f} {times['count']:<8}")
    
    avg_total_time = np.mean(total_times) if total_times else 0
    print(f"\nTotal inference time: {avg_total_time:.3f} ~~ {np.std(total_times):.3f} ms")
    
    return avg_layer_times, avg_total_time

def profile_memory_usage(model, input_data):
    """??????"""
    print("\n=== Memory Usage Analysis ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory analysis")
        return {}
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # ??????
    initial_memory = torch.cuda.memory_allocated()
    
    model.eval()
    with torch.no_grad():
        # ???????
        pre_inference_memory = torch.cuda.memory_allocated()
        
        # ????
        result = model(test_mode=True, **input_data)
        
        # ???????
        post_inference_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
    
    print(f"Initial memory: {initial_memory / 1024**2:.1f} MB")
    print(f"Pre-inference memory: {pre_inference_memory / 1024**2:.1f} MB")
    print(f"Post-inference memory: {post_inference_memory / 1024**2:.1f} MB")
    print(f"Peak memory: {peak_memory / 1024**2:.1f} MB")
    print(f"Memory increase during inference: {(post_inference_memory - pre_inference_memory) / 1024**2:.1f} MB")
    print(f"Peak memory increase: {(peak_memory - initial_memory) / 1024**2:.1f} MB")
    
    return {
        'initial_memory_mb': initial_memory / 1024**2,
        'pre_inference_memory_mb': pre_inference_memory / 1024**2,
        'post_inference_memory_mb': post_inference_memory / 1024**2,
        'peak_memory_mb': peak_memory / 1024**2,
        'inference_memory_increase_mb': (post_inference_memory - pre_inference_memory) / 1024**2,
        'peak_memory_increase_mb': (peak_memory - initial_memory) / 1024**2
    }

def profile_cuda_kernels(model, input_data, num_runs=5):
    """??CUDA????"""
    print("\n=== CUDA Kernel Analysis ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping kernel analysis")
        return {}
    
    model.eval()
    kernel_stats = {}
    
    for run in range(num_runs):
        with profile(
            activities=[ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False
        ) as prof:
            with torch.no_grad():
                _ = model(test_mode=True, **input_data)
        
        # ????
        for event in prof.key_averages():
            if event.cuda_time_total > 0:
                kernel_name = event.key
                cuda_time = event.cuda_time_total / 1000  # ?????
                
                if kernel_name not in kernel_stats:
                    kernel_stats[kernel_name] = []
                kernel_stats[kernel_name].append(cuda_time)
    
    # ??????
    kernel_summary = {}
    for kernel_name, times in kernel_stats.items():
        kernel_summary[kernel_name] = {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'total_time_ms': np.sum(times),
            'call_count': len(times)
        }
    
    # ??top??
    if kernel_summary:
        print("Top 10 CUDA kernels by total time:")
        sorted_kernels = sorted(kernel_summary.items(), 
                               key=lambda x: x[1]['total_time_ms'], reverse=True)
        
        print(f"{'Kernel Name':<60} {'Total (ms)':<12} {'Mean (ms)':<12} {'Calls':<8}")
        print("-" * 95)
        
        for kernel_name, stats in sorted_kernels[:10]:
            print(f"{kernel_name[:59]:<60} {stats['total_time_ms']:<12.3f} "
                  f"{stats['mean_time_ms']:<12.3f} {stats['call_count']:<8}")
    
    return kernel_summary

def profile_model(model, input_data):
    """??torchprofile????"""
    
    class TestModeWrapper(torch.nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.model = original_model
        
        def forward(self, lq):
            # ?????????????
            data = {'lq': lq}
            result = self.model(test_mode=True, **data)
            
            # ??????
            if isinstance(result, dict):
                # ??????????
                if 'sr' in result:
                    return result['sr']
                elif 'output' in result:
                    return result['output']
                else:
                    return list(result.values())[0]
            return result
    
    model.eval()
    wrapped_model = TestModeWrapper(model)
    
    # ??input_data???,????
    if isinstance(input_data, dict):
        input_tensor = input_data['lq']  # ??????????
    else:
        input_tensor = input_data
    
    try:
        macs = profile_macs(wrapped_model, input_tensor)
        total_params = sum(p.numel() for p in model.parameters())

        print("=== Torchprofile Analysis ===")
        print(f"MACs: {macs:,}")
        print(f"GMACs: {macs / 1e9:.2f}")
        print(f"Parameters: {total_params:,}")

        return macs, total_params
    except Exception as e:
        print(f"Torchprofile analysis failed: {e}")
        return None, None

def benchmark_inference(model, data, warmup_runs=10, test_runs=100, enable_detailed_profiling=True):
    """Benchmark GPU inference time with detailed profiling"""
    device = next(model.parameters()).device

    print(f"\n=== Benchmarking Inference Time ===")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # Get input shape info
    lq_tensor = data['lq']
    batch_size, channels, height, width = lq_tensor.shape
    pixels_per_batch = height * width * batch_size

    print(f"Input shape: {lq_tensor.shape}")
    print(f"Input size: {lq_tensor.numel() * 4 / 1024 ** 2:.1f} MB")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Test runs: {test_runs}")

    # Test forward pass first
    try:
        model.eval()
        with torch.no_grad():
            result = model(test_mode=True, **data)
        output = result['output']
        print(f"Output shape: {output.shape}")
        print(f"Output size: {output.numel() * 4 / 1024 ** 2:.1f} MB")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        return None

    # Warmup runs
    print("\nWarming up...")
    with torch.no_grad():
        for i in range(warmup_runs):
            _ = model(test_mode=True, **data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            if (i + 1) % 5 == 0:
                print(f"Warmup: {i + 1}/{warmup_runs}")

    # ??????
    profiling_results = {}
    if enable_detailed_profiling:
        try:
            # 1. ??????
            macs, params = profile_model(model, data)
            if macs is not None:
                profiling_results['macs'] = macs
                profiling_results['parameters'] = params
            
            # 2. ???PyTorch Profiler??
            prof = profile_model_detailed(model, data)
            
            # 3. ????
            layer_times, total_time = profile_layer_by_layer(model, data, num_runs=5)
            profiling_results['layer_times'] = layer_times
            profiling_results['layer_analysis_total_time'] = total_time
            
            # 4. ??????
            memory_stats = profile_memory_usage(model, data)
            profiling_results['memory_stats'] = memory_stats
            
            # 5. CUDA????
            kernel_stats = profile_cuda_kernels(model, data, num_runs=3)
            profiling_results['kernel_stats'] = kernel_stats
            
        except Exception as e:
            print(f"Error during detailed profiling: {e}")
            profiling_results['profiling_error'] = str(e)

    # Clear cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Benchmark runs - Individual timing
    print("\nRunning individual timing benchmark...")
    individual_times = []

    with torch.no_grad():
        for i in range(test_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()

            start_time = time.time()
            _ = model(test_mode=True, **data)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            individual_times.append(inference_time)

            if (i + 1) % 20 == 0:
                print(f"Completed {i + 1}/{test_runs} runs")

    # Benchmark runs - Batch timing
    print("\nRunning batch timing benchmark...")
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()

        batch_start = time.time()
        for _ in range(test_runs):
            _ = model(test_mode=True, **data)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        batch_duration = time.time() - batch_start

    # Calculate statistics
    individual_times = np.array(individual_times)
    mean_time = np.mean(individual_times)
    std_time = np.std(individual_times)
    min_time = np.min(individual_times)
    max_time = np.max(individual_times)
    median_time = np.median(individual_times)
    p95_time = np.percentile(individual_times, 95)
    p99_time = np.percentile(individual_times, 99)

    # Batch timing statistics
    avg_batch_time = (batch_duration / test_runs) * 1000  # ms per image
    batch_fps = test_runs / batch_duration
    individual_fps = 1000 / mean_time

    # Calculate throughput
    mpixels_per_sec_individual = (pixels_per_batch * individual_fps) / 1e6
    mpixels_per_sec_batch = (pixels_per_batch * batch_fps) / 1e6

    print(f"\n=== Benchmark Results ===")
    print(f"Resolution: {height}x{width}")
    print(f"Batch size: {batch_size}")
    print(f"Total pixels per batch: {pixels_per_batch:,}")

    print(f"\nIndividual Run Statistics:")
    print(f"  Mean: {mean_time:.2f} ~~ {std_time:.2f} ms")
    print(f"  Median: {median_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    print(f"  95th percentile: {p95_time:.2f} ms")
    print(f"  99th percentile: {p99_time:.2f} ms")
    print(f"  FPS: {individual_fps:.2f}")
    print(f"  Megapixels/sec: {mpixels_per_sec_individual:.2f}")

    print(f"\nBatch Run Statistics:")
    print(f"  Total duration: {batch_duration:.4f} seconds")
    print(f"  Average per image: {avg_batch_time:.2f} ms")
    print(f"  FPS: {batch_fps:.2f}")
    print(f"  Megapixels/sec: {mpixels_per_sec_batch:.2f}")

    if device.type == 'cuda':
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB")

    results = {
        'resolution': f"{height}x{width}",
        'batch_size': batch_size,
        'pixels_per_batch': pixels_per_batch,
        'individual_stats': {
            'mean_time': mean_time,
            'std_time': std_time,
            'min_time': min_time,
            'max_time': max_time,
            'median_time': median_time,
            'p95_time': p95_time,
            'p99_time': p99_time,
            'fps': individual_fps,
            'mpixels_per_sec': mpixels_per_sec_individual
        },
        'batch_stats': {
            'total_duration': batch_duration,
            'avg_time_per_image': avg_batch_time,
            'fps': batch_fps,
            'mpixels_per_sec': mpixels_per_sec_batch
        }
    }
    
    # ????????
    if enable_detailed_profiling:
        results['profiling'] = profiling_results
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='Enhancement model benchmarking')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument('--warmup_runs', type=int, default=10, help='Number of warmup runs')
    parser.add_argument('--test_runs', type=int, default=100, help='Number of test runs')
    parser.add_argument('--height', type=int, default=2160, help='Input height for dummy data')
    parser.add_argument('--width', type=int, default=3840, help='Input width for dummy data')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--disable_detailed_profiling', action='store_true', 
                       help='Disable detailed profiling to speed up benchmarking')
    parser.add_argument('--output_dir', type=str, default='./profiling_results',
                       help='Directory to save profiling results')
    return parser.parse_args()

def main():
    args = parse_args()

    print("Enhancement Model Inference Benchmark")
    print("=" * 50)

    # Initialize model
    print("Loading model...")
    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")

    # Create dummy data
    print(f"\nCreating dummy input: {args.batch_size}x3x{args.height}x{args.width}")
    dummy_data = create_dummy_data(model, args.height, args.width, args.batch_size)

    # Run benchmark
    enable_profiling = not args.disable_detailed_profiling
    benchmark_results = benchmark_inference(
        model, dummy_data, args.warmup_runs, args.test_runs, enable_profiling
    )

    if benchmark_results:
        print(f"\n=== Final Summary ===")
        print(
            f"Individual timing - Average: {benchmark_results['individual_stats']['mean_time']:.2f} ms, FPS: {benchmark_results['individual_stats']['fps']:.2f}")
        print(
            f"Batch timing - Average: {benchmark_results['batch_stats']['avg_time_per_image']:.2f} ms, FPS: {benchmark_results['batch_stats']['fps']:.2f}")
        print(f"Throughput: {benchmark_results['individual_stats']['mpixels_per_sec']:.2f} Megapixels/sec")

        # Save benchmark results
        os.makedirs(args.output_dir, exist_ok=True)
        results_file = os.path.join(args.output_dir, 
                                   f"benchmark_results_{args.height}x{args.width}_batch{args.batch_size}_gpu{args.device}.json")
        
        benchmark_info = {
            'config': args.config,
            'checkpoint': args.checkpoint,
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name() if device.type == 'cuda' else None,
            'pytorch_version': torch.__version__,
            'warmup_runs': args.warmup_runs,
            'test_runs': args.test_runs,
            'detailed_profiling_enabled': enable_profiling,
            'results': benchmark_results
        }

        with open(results_file, 'w') as f:
            json.dump(benchmark_info, f, indent=2)

        print(f"\nBenchmark results saved to: {results_file}")
        
        if enable_profiling:
            print(f"Detailed profiling results saved to: {args.output_dir}")
            print("You can view the Chrome trace by opening trace.json in Chrome's chrome://tracing")

    print("\nBenchmark completed!")

if __name__ == '__main__':
    main()