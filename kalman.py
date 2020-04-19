import cv2
import scipy
import numpy as np
from scipy.spatial.distance import mahalanobis

# 取95%置信度时卡方分布上侧alpha分数
# 同样来自Matlab的chi2inv()函数
chi2inv95 = {
    1:3.841459,
    2:5.991465,
    3:7.814728,
    4:9.487729,
    5:11.070498,
    6:12.591587,
    7:14.067140,
    8:15.507313,
    9:16.918978
}

class KalmanFilter(object):
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
        # Inheritance from cv2.KalmanFilter is not supported
        # Please reference https://github.com/opencv/opencv/issues/15804
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.eye(8, 8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, 4+i] = 1.0
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        self.std_weight_position = 1. / 20
        self.std_weight_velocity = 1. / 160
    
    def initialize(self, measurement):
        '''初始化后验状态和后验误差估计协方差
        
        Args:
            measurement (numpy.ndarray): 测量, measurement=[x,y,a,h]
        '''
        self.kf.statePost = np.r_[measurement, np.zeros_like(measurement)].astype(np.float32)
        self.kf.errorCovPost = np.diag(np.square([
            2  * self.std_weight_position * measurement[3],
            2  * self.std_weight_position * measurement[3],
            1e-2,
            2  * self.std_weight_position * measurement[3],
            10 * self.std_weight_velocity * measurement[3],
            10 * self.std_weight_velocity * measurement[3],
            1e-5,
            10 * self.std_weight_velocity * measurement[3]
        ])).astype(np.float32)
        self.kf.statePre = self.kf.statePost
        self.kf.errorCovPre = self.kf.errorCovPost
    
    def predict(self, *args, **kwargs):
        '''预测
        '''
        
        # 计算过程噪声协方差
        self.kf.processNoiseCov = np.diag(np.square([
            self.std_weight_position * self.kf.statePre[3],
            self.std_weight_position * self.kf.statePre[3],
            1e-2,
            self.std_weight_position * self.kf.statePre[3],
            self.std_weight_velocity * self.kf.statePre[3],
            self.std_weight_velocity * self.kf.statePre[3],
            1e-5,
            self.std_weight_velocity * self.kf.statePre[3]
        ])).astype(np.float32)
        
        # 预测状态和误差估计协方差
        return self.kf.predict(*args, **kwargs)
    
    def correct(self, measurement):
        '''更新, 第二帧的self.tracked_trajectories的is_activated都为False,
            意味着都没有加入trajectory_pool, 也就没有predict, 导致statePre,
            errorCovPre都没有更新, 全是0
        '''
        
        # 计算测量噪声协方差
        self.kf.measurementNoiseCov = np.diag(np.square([
            self.std_weight_position * self.kf.statePre[3],
            self.std_weight_position * self.kf.statePre[3],
            1e-1,
            self.std_weight_position * self.kf.statePre[3]
        ])).astype(np.float32)
        
        return self.kf.correct(measurement)
    
    def project(self):
        '''将状态分布投影到测量空间
        '''
        measurementNoiseCov = np.diag(np.square([
            self.std_weight_position * self.kf.statePost[3],
            self.std_weight_position * self.kf.statePost[3],
            1e-1,
            self.std_weight_position * self.kf.statePost[3]
        ]))
        
        mean = np.dot(self.kf.measurementMatrix, self.kf.statePost)
        covariance = np.linalg.multi_dot((self.kf.measurementMatrix,
            self.kf.errorCovPost, self.kf.measurementMatrix.T)) + measurementNoiseCov
        return mean, covariance
    
    def gating_distance(self, measurement, only_position=False, metric='maha'):
        '''计算测量和状态分布之间的门限距离(Mahalanobis distance)
            Please reference https://en.wikipedia.org/wiki/Mahalanobis_distance
        
        Args:
            measurement (numpy.ndarray): 测量, measurement=[x,y,a,h]
            only_position (bool, optional): 仅使用位置信息, 如果为真,
                仅考虑建议框的中心坐标
            metric (str, optional): 距离度量方法
        Returns:
            dists (numpy.ndarray): 测量和状态分布之间的门限距离
        '''
        mean, covariance = self.project()
        covariance = np.linalg.inv(covariance)
        dists = [mahalanobis(x, mean, covariance) for x in measurement]
        dists = np.square(dists)
        # d = measurement - mean
        # cholesky_factor = np.linalg.cholesky(covariance)
        # z = scipy.linalg.solve_triangular(
        #         cholesky_factor, d.T, lower=True, check_finite=False,
        #         overwrite_b=True)
        # dists = np.sum(z * z, axis=0)
        return dists

if __name__ == '__main__':
    kfs = [KalmanFilter() for _ in range(50)]
    print(kfs)
    print(kfs[0].kf.transitionMatrix)
    print(kfs[0].kf.measurementMatrix)
    
    measurement = np.load('measurements.npy')
    dists = kfs[0].gating_distance(measurement)