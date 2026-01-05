# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import torch
from mmcv.runner import Hook
from torch.utils.data import DataLoader


class EvalIterHook(Hook):
    """Non-Distributed evaluation hook for iteration-based runner.

    This hook will regularly perform evaluation in a given interval when
    performing in non-distributed environment.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval. Default: 1.
        save_best (bool): Whether to save the best checkpoint. Default: True.
        key_indicator (str): The metric key for best checkpoint. Default: 'PSNR'.
        eval_kwargs (dict): Other eval kwargs. It contains:
            save_image (bool): Whether to save image.
            save_path (str): The path to save image.
    """

    def __init__(self, dataloader, interval=1, save_best=True, 
                 key_indicator='PSNR', **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, '
                            f'but got { type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.eval_kwargs = eval_kwargs
        self.save_image = self.eval_kwargs.pop('save_image', False)
        self.save_path = self.eval_kwargs.pop('save_path', None)
        
        # Best checkpoint saving
        self.save_best = save_best
        self.key_indicator = key_indicator
        self.best_score = -float('inf')
        self.best_iter = -1

    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
        """
        if not self.every_n_iters(runner, self.interval):
            return
        runner.log_buffer.clear()
        from mmedit.apis import single_gpu_test
        results = single_gpu_test(
            runner.model,
            self.dataloader,
            save_image=self.save_image,
            save_path=self.save_path,
            iteration=runner.iter)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        """Evaluation function.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
            results (dict): Model forward results.
        """
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        
        # Save best checkpoint
        if self.save_best and self.key_indicator in eval_res:
            current_score = eval_res[self.key_indicator]
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_iter = runner.iter
                self._save_best_checkpoint(runner)
                runner.logger.info(
                    f'New best {self.key_indicator}: {current_score:.4f} at iter {runner.iter}')

    def _save_best_checkpoint(self, runner):
        """Save the best checkpoint.
        
        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
        """
        best_ckpt_path = osp.join(runner.work_dir, f'best_{self.key_indicator}.pth')
        
        # Build checkpoint meta
        meta = dict(
            iter=runner.iter + 1,
            best_score=self.best_score,
            key_indicator=self.key_indicator
        )
        if hasattr(runner, 'meta') and runner.meta is not None:
            meta.update(runner.meta)
        
        # Save checkpoint
        optimizer = runner.optimizer
        checkpoint = {
            'meta': meta,
            'state_dict': runner.model.state_dict(),
        }
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        
        torch.save(checkpoint, best_ckpt_path)
        runner.logger.info(f'Best checkpoint saved to {best_ckpt_path}')


class DistEvalIterHook(EvalIterHook):
    """Distributed evaluation hook.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval. Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
        save_best (bool): Whether to save the best checkpoint. Default: True.
        key_indicator (str): The metric key for best checkpoint. Default: 'PSNR'.
        eval_kwargs (dict): Other eval kwargs. It may contain:
            save_image (bool): Whether save image.
            save_path (str): The path to save image.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 save_best=True,
                 key_indicator='PSNR',
                 **eval_kwargs):
        super().__init__(dataloader, interval, save_best, key_indicator, **eval_kwargs)
        self.gpu_collect = gpu_collect

    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
        """
        if not self.every_n_iters(runner, self.interval):
            return
        runner.log_buffer.clear()
        from mmedit.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect,
            save_image=self.save_image,
            save_path=self.save_path,
            iteration=runner.iter)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)
