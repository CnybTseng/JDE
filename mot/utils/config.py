from yacs.config import CfgNode as CN

_C = CN()

_C.COPYRIGHT = CN()
_C.COPYRIGHT.AUTHOR = "Zhiwei Tseng"

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1
_C.SYSTEM.NUM_WORKERS = 8
_C.SYSTEM.PIN_MEMORY = True
_C.SYSTEM.TASK_DIR = './tasks/'
_C.SYSTEM.RESUME = False
_C.SYSTEM.LOG_INTERVAL = 40
_C.SYSTEM.MODEL_SAVE_INTERVAL = 2
_C.SYSTEM.REPORT_INTERVAL = 10
_C.SYSTEM.REPORT_ARGS = ''

_C.DATASET = CN()
_C.DATASET.NAME = "HotchpotchDataset"
_C.DATASET.ARGS = [["/data/tseng/dataset/jde"],
    {'cfg': './data/train.txt', 'backbone': 'shufflenetv2', 'augment': True}]

_C.DATALOADER = CN()
_C.DATALOADER.COLLATE = CN()
_C.DATALOADER.COLLATE.NAME = "TrackerCollate"
_C.DATALOADER.COLLATE.ARGS = [[], {'multiscale': False}]
_C.DATALOADER.SHUFFLE = True

_C.MODEL = CN()
_C.MODEL.NAME = "JDE"
_C.MODEL.ARGS = CN()
_C.MODEL.ARGS.INPUT = CN()
_C.MODEL.ARGS.INPUT.WIDTH = 576
_C.MODEL.ARGS.INPUT.HEIGHT = 320
_C.MODEL.ARGS.BACKBONE = CN()
_C.MODEL.ARGS.BACKBONE.NAME = "ShuffleNetV2"
_C.MODEL.ARGS.BACKBONE.ARGS = [[], {
    'arch': {
        'conv1':  {'out_channels': 24},
        'stage2': {'out_channels': 116, 'repeate': 4, 'out': True},
        'stage3': {'out_channels': 232, 'repeate': 8, 'out': True},
        'stage4': {'out_channels': 464, 'repeate':4, 'out': True},
        'conv5':  {'out_channels': 1024}},
    'pretrained': None
}]
_C.MODEL.ARGS.NECK = CN()
_C.MODEL.ARGS.NECK.NAME = "FPN"
_C.MODEL.ARGS.NECK.ARGS = [[], {'in_channels': [116, 232, 464]}]
_C.MODEL.ARGS.HEAD = CN()
_C.MODEL.ARGS.HEAD.NAME = "JDEHead"
_C.MODEL.ARGS.HEAD.ARGS = [[], {
    'build_block': 'ShuffleNetV2BuildBlock',
    'block_repeat': {'detect': 3, 'identify': 3},
    'block_args': [128, 128, 1, {}],
    'num_ide': 0,
    'anchor': [[[85, 255], [120, 360], [170, 420], [340, 320]],
        [[21, 64], [30, 90], [43, 128], [60, 180]],
        [[6, 16], [8, 23], [11, 32], [16, 45]]],
    'num_class': 1,
    'embd_dim': 128,
    'box_loss': 'DIOULoss',
    'ide_thresh': 0.5,
    'obj_thresh': 0.5,
    'bkg_thresh': 0.4,
    's_box': [0, 0, 0],
    's_cls': [0, 0, 0],
    's_ide': [0, 0, 0],
    'im_size': [_C.MODEL.ARGS.INPUT.HEIGHT, _C.MODEL.ARGS.INPUT.WIDTH]
}]

_C.SOLVER = CN()
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.ACCUMULATED_BATCHES = 1
_C.SOLVER.WARMUP = 1000
_C.SOLVER.EPOCHS = 100
_C.SOLVER.WORK_FLOW = [('train', 1), ('val', 0)]
_C.SOLVER.OPTIM = CN()
_C.SOLVER.OPTIM.NAME = "SGD"
_C.SOLVER.OPTIM.ARGS = [[], {'lr': 0.025, 'momentum': 0.9, 'weight_decay': 0.0001}]
_C.SOLVER.LR_SCHEDULER = CN()
_C.SOLVER.LR_SCHEDULER.NAME = "MultiStepLR"
_C.SOLVER.LR_SCHEDULER.ARGS = [[[50, 75]], {'gamma': 0.1}]