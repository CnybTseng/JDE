from yacs.config import CfgNode as CN

_C = CN()

_C.COPYRIGHT = CN()
_C.COPYRIGHT.AUTHOR = "Zhiwei Tseng"

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1
_C.SYSTEM.NUM_WORKERS = 8
_C.SYSTEM.PIN_MEMORY = True

_C.DATASET = CN()
_C.DATASET.ROOT_DIR = "/data/tseng/dataset/jde"
_C.DATASET.TRAIN_SET = "./data/train.txt"
_C.DATASET.TEST_SET = "./data/test.txt"

_C.MODEL = CN()
_C.MODEL.NAME = "JDE"
_C.MODEL.ARGS = CN()
_C.MODEL.ARGS.INPUT = CN()
_C.MODEL.ARGS.INPUT.WIDTH = 576
_C.MODEL.ARGS.INPUT.HEIGHT = 320
_C.MODEL.ARGS.BACKBONE = CN()
_C.MODEL.ARGS.BACKBONE.NAME = "ShuffleNetV2"
_C.MODEL.ARGS.BACKBONE.ARGS = [{
    'stage_repeat': {'stage2': 4, 'stage3': 8, 'stage4': 4},
    'stage_out_channels': {'conv1': 24, 'stage2': 48, 'stage3': 96, 'stage4': 192, 'conv5': 1024},
    'pretrained': '/home/image/tseng/project/JDE/models/ShuffleNetV2.0.5x.pth.tar'
}]
_C.MODEL.ARGS.NECK = CN()
_C.MODEL.ARGS.NECK.NAME = "FPN"
_C.MODEL.ARGS.NECK.ARGS = [{}]
_C.MODEL.ARGS.HEAD = CN()
_C.MODEL.ARGS.HEAD.NAME = "JDEHead"
_C.MODEL.ARGS.HEAD.ARGS = [{
    'build_block': 'ShuffleNetV2BuildBlock',
    'block_repeat': {'detect': 3, 'identify': 3},
    'block_args': [128, 128, 1, {}],
    'num_ide': 14500,
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
    'im_size': [_C.MODEL.ARGS.INPUT.WIDTH, _C.MODEL.ARGS.INPUT.HEIGHT]
}]
_C.MODEL.ARGS.LOSS = CN()
_C.MODEL.ARGS.LOSS.BOX = "DIOULoss"
_C.MODEL.ARGS.LOSS.CLASS = "CrossEntropyLoss"
_C.MODEL.ARGS.LOSS.IDENDITY = "CrossEntropyLoss"

_C.SOLVER = CN()
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.ACCUMULATED_BATCHES = 1
_C.SOLVER.WARMUP = 1000
_C.SOLVER.EPOCHS = 50
_C.SOLVER.LR = 0.01
_C.SOLVER.LR_GAMMA = 0.1
_C.SOLVER.MILESTONES = [0.5, 0.75]
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.OPTIM = "SGD"