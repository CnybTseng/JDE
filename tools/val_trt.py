import sys
import torch
import collections
import numpy as np
sys.path.append('.')
import jde
import shufflenetv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = shufflenetv2.ShuffleNetV2(model_size='1.0x').to(device)

model_dict = model.state_dict()
trained_model_dict = torch.load('workspace/joint-20201216-01/checkpoint/jde-ckpt-049.pth', map_location='cpu')
trained_model_dict = {k : v for (k, v) in trained_model_dict.items() if k in model_dict}
trained_model_dict = collections.OrderedDict(trained_model_dict)
model_dict.update(trained_model_dict)
model.load_state_dict(model_dict)

model.eval()

decoder = jde.JDEcoder((320, 576))

input = torch.from_numpy(np.fromfile(open('in.bin', 'rb'), dtype=np.float32)).view(1, 3, 320, 576).to(device)

with torch.no_grad():
    outputs = model(input)

#outputs = decoder(outputs)
for output in outputs:
    print(output.shape)

trt_outputs = []
size = [[1, 10, 18, 152], [1, 20, 36, 152], [1, 40, 72, 152]]
for i in range(3):
    data = np.fromfile(open('out{}.bin'.format(i)), dtype=np.float32)
    data = torch.from_numpy(data).view(size[i]).to(device)
    data = data.permute(0, 3, 1, 2)
#    det = data[:, : 24, ...]
#    det = det.view(1, 4, 6, size[i][1], size[i][2])
#    det = det.permute(0, 1, 3, 4, 2).contiguous()
#    det = det.view(1, -1, 6)
#    
#    ide = data[:, 24 :, ...]
#    ide = ide.unsqueeze(1).repeat(1, 4, 1, 1, 1)
#    ide = ide.permute(0, 1, 3, 4, 2).contiguous()
#    ide = ide.view(1, -1, 128)
#    
#    data = torch.cat([det, ide], dim=-1)
    trt_outputs.append(data)

#trt_outputs = torch.cat(trt_outputs, dim=1).detach().cpu()
for trt_output in trt_outputs:
    print(trt_output.shape)

# diff = torch.abs(outputs - trt_outputs)
# print(diff.mean())
# print(diff.min())
# print(diff.max())

for output, trt_output in zip(outputs, trt_outputs):
    diff = torch.abs(output - trt_output)
    print(diff.mean())
    print(diff.min())
    print(diff.max())    