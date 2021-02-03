import os
import torch
from mot.utils import get_logger
from mot.datasets import build_dataloader
from mot.optimizers import build_optimizer, build_lr_scheduler
from mot.runner import Runner

def train_tracker(model, dataset, config):
    model.cuda()

    # Build logger.
    logger_path = os.path.join(config.SYSTEM.TASK_DIR, 'log.txt')
    logger = get_logger(path=logger_path)
    
    # Build dataloader.
    dataloader = build_dataloader(dataset, config)
    logger.info('Build dataloader done.')
    
    # Build optimizer.
    config.SOLVER.OPTIM.ARGS[0].append(model.parameters())
    optimizer = build_optimizer(config.SOLVER.OPTIM)
    logger.info('Build optimizer done.')
    
    # Build learning scheduler.
    config.SOLVER.LR_SCHEDULER.ARGS[0].insert(0, optimizer)
    lr_scheduler = build_lr_scheduler(config.SOLVER.LR_SCHEDULER)
    config.freeze()
    logger.info('Build learning scheduler done.')
    
    # If resume training from checkpoint:
    epoch, batch = 0, 0
    if config.SYSTEM.RESUME:
        cpath = os.path.join(config.SYSTEM.TASK_DIR, 'latest.pth')
        checkpoint = torch.load(cpath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        epoch = checkpoint['epoch']
        batch = checkpoint['batch']
        logger.info('Resume training:'
            ' load checkpoint from {} done.'.format(cpath))
    
    # Build task runner.
    runner = Runner(model, dataloader, optimizer, lr_scheduler,
        config.SOLVER.EPOCHS, epoch=epoch, batch=batch,
        warmup=config.SOLVER.WARMUP,
        work_flow=config.SOLVER.WORK_FLOW, logger=logger,
        task_dir=config.SYSTEM.TASK_DIR,
        log_inter=config.SYSTEM.LOG_INTERVAL,
        model_save_inter=config.SYSTEM.MODEL_SAVE_INTERVAL,
        report_inter=config.SYSTEM.REPORT_INTERVAL,
        report_args=config.SYSTEM.REPORT_ARGS)
    logger.info('Build task runner done.')
    
    # Execute training task now.
    runner.run()