import struct
import argparse
import torchreid
from torchreid.utils import load_pretrained_weights

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', '-dr', type=str,
    default='/home/image/tseng/dataset/reid',
    help='the root directory of dataset')
parser.add_argument('--model-path', '-mp', type=str,
    help='model path')
parser.add_argument('--wts', type=str,
    default='./tasks/osnet_x1_0.wts',
    help='path to the generated .wts file (for tensorrt)')
args = parser.parse_args()

datamanager = torchreid.data.ImageDataManager(
    root=args.data_root,
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=32,
    transforms=['random_flip', 'random_crop'])

model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True)

load_pretrained_weights(model, args.model_path)

print(model)
with open(args.wts, 'w') as fd:
    fd.write('{}\n'.format(len(model.state_dict().keys())))
    for name, param in model.state_dict().items():
        print('export {}, shape {}'.format(name, param.shape))
        w = param.reshape(-1).cpu().numpy()
        fd.write('{} {}'.format(name, len(w)))
        for wi in w:
            fd.write(' ')
            fd.write(struct.pack('>f', float(wi)).hex())
        fd.write('\n')