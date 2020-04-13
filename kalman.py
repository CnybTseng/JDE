import cv2
import numpy as np

class KalmanFilter(cv2.KalmanFilter):
    '''多目标跟踪的卡尔曼滤波器. 状态向量s=(x,y,a,h,vx,vy,va,vh),
        其中的a是目标建议框横纵比. 状态转移函数如下:
        x  = x + vx
        y  = y + vy
        a  = a + va
        h  = h + vh
        vx = vx
        vy = vy
        va = va
        vh = vh
    '''
    def __init__(self):
        super(KalmanFilter, self).__init__(8, 4)
        self.transitionMatrix = np.eye(8, 8)
        for i in range(4):
            self.transitionMatrix[i, 4+i] = 1.0
        self.measurementMatrix = np.eye(4, 8)
        self.std_weight_position = 1. / 20
        self.std_weight_velocity = 1. / 160
    
    def predict(self, *args, **kwargs):
        '''预测
        '''
        
        # 计算过程噪声协方差
        self.processNoiseCov = np.diag(np.square([
            self.std_weight_position * self.statePre[3],
            self.std_weight_position * self.statePre[3],
            1e-2,
            self.std_weight_position * self.statePre[3],
            self.std_weight_velocity * self.statePre[3],
            self.std_weight_velocity * self.statePre[3],
            1e-5,
            self.std_weight_velocity * self.statePre[3]
        ]))
        
        # 预测状态和误差估计协方差
        return super().predict(*args, **kwargs)
    
    def correct(self, measurement):
        '''更新
        '''
        
        # 计算测量噪声协方差
        self.measurementNoiseCov = np.diag(np.square(
            self.std_weight_position * self.statePre[3],
            self.std_weight_position * self.statePre[3],
            1e-1,
            self.std_weight_position * self.statePre[3]
        ))
        
        return super().correct(measurement)

if __name__ == '__main__':
    kf = KalmanFilter()
    print(kf.transitionMatrix)
    print(kf.measurementMatrix)