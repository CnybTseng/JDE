import os
import torch
import numpy as np
import os.path as osp
from functools import wraps
from collections import defaultdict
from mot.utils import move_tensors_to_gpu
from mot.utils import build_excel, append_excel
from mot.datasets import HotchpotchDataset, DBSHotchpotchDataset

def Trigger(unit):
    def decorator(fun):
        @wraps(fun)
        def wrapper(self, *args, **kwargs):
            if unit == 'BATCH':
                if self._batch == 1 or \
                    self._batch % self.trigger_inter[fun.__name__] == 0:
                    fun(self, *args, **kwargs)
            elif unit == 'EPOCH':
                if self._epoch % self.trigger_inter[fun.__name__] == 0:
                    fun(self, *args, **kwargs)
        return wrapper
    return decorator

class Runner(object):
    '''Simple runner for training and validation.
    
    Param
    -----
    model       : The model to be run.
    dataloader  : The dataloader.
    optimizer   : The model optimizer.
    lr_scheduler: The learning scheduler.
    total_epochs: Total number of epochs for training.
    epoch       : The start epoch for training.
    batch       : Total number of batches been seen.
    warmup      : Learning rate warmup iterations.
    work_flow   : A list of working flow defines how many epochs
                  the training or validation mode runs.
    logger      : The system logger.
    task_dir    : Directory for the task generated stuff.
    log_inter   : Logger working period. Unit: batch.
    model_save_inter: Model saving period. Unit: epoch.
    report_inter: E-mail reporting period. Unit: epoch.
    '''
    def __init__(self, model, dataloader, optimizer, lr_scheduler,
        total_epochs, epoch=0, batch=0, warmup=1000,
        work_flow=[('train', 1), ('val', 0)], logger=None,
        task_dir=os.getcwd(), log_inter=40, model_save_inter=2,
        report_inter=10, report_args=''):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self._total_epochs = total_epochs
        self._epoch = epoch
        self.warmup = warmup
        self.work_flow = work_flow
        self.logger = logger
        self.task_dir = task_dir
        self._batch = batch
        self.init_lr = []
        for pg in optimizer.param_groups:
            self.init_lr.append(pg['lr'])
        self.trigger_inter = {
            self.log_metrics.__name__: log_inter,
            self.save_model.__name__: model_save_inter,
            self.report_email.__name__: report_inter}
        self.rm_metrics = defaultdict(float)
        self.xlspath = os.path.join(task_dir, 'log.xls')
        self.report_args = report_args

    def train(self, **kwargs):
        """Training mode"""
        self._epoch += 1
        self.model.train()
        self.optimizer.zero_grad()
        for i, data in enumerate(self.dataloader):
            self._batch += 1
            self.lr_warmup()
            data = move_tensors_to_gpu(data)
            loss, metrics = self.model(*data)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.smooth_metrics(metrics, i + 1)
        self.lr_scheduler.step()
        self.save_checkpoint()
        self.report_email()
    
    def val(self, **kwargs):
        """Validation mode"""
        raise NotImplementedError('validation mode has not'
            ' been implemented')
    
    def run(self, **kwargs):
        """Execute training task"""
        self.logger.info('Start running')
        while self._epoch < self._total_epochs:
            # Switch between 'train' and 'val' mode.
            for mode, epochs in self.work_flow:
                if isinstance(mode, str):
                    if not hasattr(self, mode):
                        raise ValueError('Runner has no method named'
                            ' {} to run'.format(mode))
                    runner = getattr(self, mode)
                else:
                    raise TypeError('str type is expected, but got'
                        ' {}'.format(type(mode)))
                for _ in range(epochs):
                    runner(**kwargs)
    
    def lr_warmup(self, power=4):
        """Learning rate warmup"""
        if self._batch > self.warmup:
            return
        for pg, lr in zip(self.optimizer.param_groups, self.init_lr):
            pg['lr'] = lr * (self._batch / self.warmup) ** power
    
    @Trigger('BATCH')
    def log_metrics(self, batch_id):
        """Log metrics in file"""
        # For simplify, only supported str and float
        # type metrics return by model.
        npg = len(self.optimizer.param_groups)
        mnames = list(self.rm_metrics.keys())
        if self._batch == 1:
            head = ['Epoch', 'Batch'] + \
                ['LR{}'.format(i) for i in range(npg)] + mnames
            head_fmt = ('%8s%10s' + '%10s' * (npg + \
                len(mnames))) % tuple(head)
            self.logger.info(head_fmt)
            build_excel(self.xlspath, head, 'training')
        
        # DataLoader with IterableDataset has no method 'len'
        if isinstance(self.dataloader.dataset, HotchpotchDataset):
            nbatch_per_epoch = len(self.dataloader)
        else:
            nbatch_per_epoch = np.Inf
        
        mval = [('%g/%g') % (self._epoch, self._total_epochs),
            ('%g/%g') % (batch_id, nbatch_per_epoch)] + \
            [pg['lr'] for pg in self.optimizer.param_groups] + \
            list(self.rm_metrics.values())
        fmt = '%8s%10s' + '%10.3g' * npg
        for v in self.rm_metrics.values():
            if isinstance(v, float):
                fmt += '%10.3g'
            elif isinstance(v, str):
                fmt += '%10s'
            else:
                raise TypeError('expect float or str, but got'
                    ' {}'.format(type(v)))
        mval_fmt = (fmt) % tuple(mval)
        self.logger.info(mval_fmt)
        append_excel(self.xlspath, [mval])
    
    def smooth_metrics(self, metrics, batch_id):
        """Calculate running mean of all metrics"""
        # For simplify, only supported str and float
        # type metrics return by model.
        for k, v in metrics.items():
            if isinstance(v, str):
                self.rm_metrics[k] = v
            elif isinstance(v, float):
                self.rm_metrics[k] = (self.rm_metrics[k] * (
                    batch_id - 1) + v) / batch_id
            else:
                raise TypeError('expect float or str, but got'
                    ' {}'.format(type(v)))
        self.log_metrics(batch_id)
    
    @Trigger('EPOCH')
    def save_model(self):
        """Save model only"""
        mname = type(self.model).__name__
        mname = '%s-%03d.pth' % (mname, self._epoch)
        torch.save(self.model.state_dict(),
            os.path.join(self.task_dir, mname))
    
    def save_checkpoint(self):
        """Save everything for resume training"""
        cname = os.path.join(self.task_dir, 'latest.pth')
        torch.save({'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': self._epoch, 'batch': self._batch}, cname)
        self.save_model()
    
    @Trigger('EPOCH')
    def report_email(self):
        cwd = os.getcwd()
        cmd = 'python ' + osp.join(cwd, 'tools', 'emailsender.py') + \
            self.report_args + ' -e ' + osp.join(cwd,
            self.task_dir, 'log.xls')
        os.system(cmd)
    
    @property
    def epoch(self):
        """Get current epoch number"""
        return self._epoch
    
    @property
    def total_epochs(self):
        """Get total number of epochs"""
        return _total_epochs
    
    @property
    def batch(self):
        """Get current batch number"""
        return self._batch