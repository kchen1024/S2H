import os
import argparse
import time
import json

import torch
import numpy as np
from mmcv.parallel import collate, scatter

from mmedit.apis import init_model
from torchprofile import profile_macs

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
    except Exception as e:
        print(f"Profiling failed: {e}")
        return None, None

    total_params = sum(p.numel() for p in model.parameters())

    print("=== Torchprofile Analysis ===")
    print(f"MACs: {macs:,}")
    print(f"GMACs: {macs / 1e9:.2f}")
    print(f"Parameters: {total_params:,}")

    return macs, total_params

def benchmark_inference(model, data, warmup_runs=10, test_runs=100):
    """Benchmark GPU inference time"""
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
    profile_model(model, data)

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

    # Benchmark runs - Batch timing (similar to test_speed method)
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
    print(f"  Mean: {mean_time:.2f} ~ {std_time:.2f} ms")
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

    return {
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
    benchmark_results = benchmark_inference(
        model, dummy_data, args.warmup_runs, args.test_runs
    )

    if benchmark_results:
        print(f"\n=== Final Summary ===")
        print(
            f"Individual timing - Average: {benchmark_results['individual_stats']['mean_time']:.2f} ms, FPS: {benchmark_results['individual_stats']['fps']:.2f}")
        print(
            f"Batch timing - Average: {benchmark_results['batch_stats']['avg_time_per_image']:.2f} ms, FPS: {benchmark_results['batch_stats']['fps']:.2f}")
        print(f"Throughput: {benchmark_results['individual_stats']['mpixels_per_sec']:.2f} Megapixels/sec")

        # Save benchmark results
        results_file = f"benchmark_results_{args.height}x{args.width}_batch{args.batch_size}_gpu{args.device}.json"
        benchmark_info = {
            'config': args.config,
            'checkpoint': args.checkpoint,
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name() if device.type == 'cuda' else None,
            'warmup_runs': args.warmup_runs,
            'test_runs': args.test_runs,
            'results': benchmark_results
        }

        with open(results_file, 'w') as f:
            json.dump(benchmark_info, f, indent=2)

        print(f"\nBenchmark results saved to: {results_file}")

    print("\nBenchmark completed!")


if __name__ == '__main__':
    main()