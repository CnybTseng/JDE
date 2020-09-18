import cv2
import numpy as np

class RandomAffine(object):
    def __init__(self, degrees=(-10,10), translate=(.1,.1),
        scale=(.9,1.1), shear=(-2,2), fillcolor=(127,127,127)):
        '''随机的仿射变换.
        
        参数
        ----
        degrees  : 旋转角度采样范围.
        translate: 平移量采样范围.
        scale    : 缩放系数采样范围.
        shear    : 剪切量采样范围.
        fillcolor: 无定义区域填充色.
        '''
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.fillcolor = fillcolor
    
    def __call__(self, image, targets=None):
        '''随机的仿射变换.
        
        参数
        ----
        image  : CV_8UC3类型图像
        targets: 标签
        '''
        
        # 旋转矩阵
        center = (image.shape[1] / 2, image.shape[0] / 2)
        angle = np.random.uniform(low=self.degrees[0], high=self.degrees[1])
        scale = np.random.uniform(low=self.scale[0], high=self.scale[1])        
        
        R = np.eye(3)
        R[:2] = cv2.getRotationMatrix2D(center, angle, scale)
        
        # 平移矩阵
        tx = self.translate[0] * image.shape[1]
        ty = self.translate[1] * image.shape[0]
        
        T = np.eye(3)
        T[0, 2] = np.random.uniform(low=-tx, high=tx)
        T[1, 2] = np.random.uniform(low=-ty, high=ty)
        
        # 剪切矩阵
        shx = np.random.uniform(low=self.shear[0], high=self.shear[1])
        shy = np.random.uniform(low=self.shear[0], high=self.shear[1])
        
        S = np.eye(3)
        S[0, 1] = np.tan(shx * np.pi / 180)
        S[1, 0] = np.tan(shy * np.pi / 180)
        
        # 合成仿射变换矩阵
        M = S @ T @ R
        
        # 对图像进行仿射变换
        dst = cv2.warpPerspective(image, M, dsize=(image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR, borderValue=self.fillcolor)
        
        if targets is None:
            return dst
        
        if len(targets) == 0:
            return dst, targets
        
        # 对标签进行仿射变换
        
        
        return dst

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str,
        help='path to the image')
    args = parser.parse_args()
    
    T = RandomAffine()
    image = cv2.imread(args.image)
    dst = T(image)
    cv2.imwrite("T.png", dst)