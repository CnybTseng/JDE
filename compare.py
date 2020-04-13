import os
import torch
import argparse

import yolov3
import darknet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compare the results of reference and own implementation')
    parser.add_argument('--path', type=str, help='path to the reference data')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = darknet.DarkNet().to(device)
    model.load_state_dict(torch.load(os.path.join(args.path, 'jde.pth'), map_location='cpu'))
    model.eval()
    
    x = torch.load(os.path.join(args.path, 'input.pt'), map_location=device)
    with torch.no_grad():
        outputs = model(x)
    
    refers = []
    refers.append(torch.load(os.path.join(args.path, 'out1.pt'), map_location=device))
    refers.append(torch.load(os.path.join(args.path, 'out2.pt'), map_location=device))
    refers.append(torch.load(os.path.join(args.path, 'out3.pt'), map_location=device))
    
    for i, (output, refer) in enumerate(zip(outputs, refers)):
        print(f'the {i+1}th output size is {output.size()}, correct?{torch.equal(output, refer)}')
    
    in_size = x.shape[2:]
    num_classes = 1
    if '320x576' in args.path:
        anchors = ((6,16), (8,23), (11,32), (16,45), (21,64), (30,90), (43,128), (60,180), (85,255), (120,360), (170,420), (340,320))
    elif '480x864' in args.path:
        anchors = ((6,19), (9,27), (13,38), (18,54), (25,76), (36,107), (51,152), (71,215), (102,305), (143,429), (203,508), (407,508))
    elif '608x1088' in args.path:
        anchors = ((8,24), (11,34), (16,48), (23,68), (32,96), (45,135), (64,192), (90,271), (128,384), (180,540), (256,640), (512,640))

    decoder = yolov3.YOLOv3EvalDecoder(in_size, num_classes, anchors)
    decoded_outputs = decoder(outputs)
    
    refer = torch.load(os.path.join(args.path, 'pred.pt'), map_location='cpu')
    print(f'the decoded output size is {decoded_outputs.size()}, correct?{torch.equal(decoded_outputs, refer)}')