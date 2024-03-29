import os
import re
import cv2
import lap
import torch
import argparse
import collections
import numpy as np
from enum import IntEnum
from torchvision.ops import nms
from cython_bbox import bbox_overlaps
from scipy.spatial.distance import cdist

import jde
import kalman
import yolov3
import darknet
import dataset
import shufflenetv2
import sys
sys.path.append(os.getcwd())
from mot.utils import config
from mot.models import build_tracker

def parse_args():
    '''解析命令行参数
    '''
    parser = argparse.ArgumentParser(
        description='single class multiple object tracking')
    parser.add_argument('--config', type=str, default='',
        help='training configuration file path')
    parser.add_argument('--img-path', type=str, help='path to image path')
    parser.add_argument('--model', type=str, help='path to tracking model')
    parser.add_argument('--insize', type=str, default='320x576',
        help='network input size, default=320x576, other options are'
        ' 480x864 and 608x1088')
    parser.add_argument('--score-thresh', type=float, default=0.5,
        help='nms score threshold, default=0.5, it must be in [0,1]')
    parser.add_argument('--iou-thresh', type=float, default=0.4,
        help='nms iou threshold, default=0.4, it must be in [0,1]')
    parser.add_argument('--only-detect', action='store_true',
        help='only detecting object, no tracking')
    parser.add_argument('--backbone', type=str, default='shufflenetv2',
        help='backbone arch, default is shufflenetv2, candidate is darknet')
    parser.add_argument('--thin', type=str, default='0.5x',
        help='shufflenetv2 thin, default is 0.5x, candidates are 0.5x, 1.0x, 1.5x')
    parser.add_argument('--embedding', '-emd', type=int, default=128,
        help='embedding dimension, default is 128')
    parser.add_argument('--workspace', type=str, default='workspace',
        help='workspace path')
    return parser.parse_args()

def mkdir(path):
    '''目录不存在, 则创建之
    
    Args：
        path (str): 需要创建的目录
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def xywh2ltrb(boxes):
    '''转换建议框的数据格式
    
    Args:
        boxes (torch.Tensor): [x,y,w,h]格式的建议框
    Returns:
        ltrb (torch.Tensor): [l,t,r,b]格式的建议框
    '''
    ltrb = torch.zeros_like(boxes)
    ltrb[:,0] = boxes[:,0] - boxes[:,2]/2
    ltrb[:,1] = boxes[:,1] - boxes[:,3]/2
    ltrb[:,2] = boxes[:,0] + boxes[:,2]/2
    ltrb[:,3] = boxes[:,1] + boxes[:,3]/2
    return ltrb

def ltrb2xyah(ltrb):
    '''转换建议框的数据格式
    
    Args:
        ltrb (numpy.ndarray): [l,t,r,b]格式的建议框
    Returns:
        xyah (numpy.ndarray): [x,y,a,h]格式的建议框
    '''
    dim = len(ltrb.shape)
    ltrb = ltrb.reshape(-1, 4)
    xyah = np.zeros_like(ltrb)
    xyah[:,0] = (ltrb[:,0] + ltrb[:,2]) / 2
    xyah[:,1] = (ltrb[:,1] + ltrb[:,3]) / 2
    xyah[:,3] = (ltrb[:,3] - ltrb[:,1])
    xyah[:,2] = (ltrb[:,2] - ltrb[:,0]) / xyah[:,3]
    if dim == 1:
        xyah = xyah.reshape(-1)
    return xyah

def xyah2ltrb(xyah):
    dim = len(xyah.shape)
    xywh = xyah.copy().reshape(-1, 4)
    xywh[:,2] *= xywh[:,3]
    ltrb = np.zeros_like(xywh)
    ltrb[:,0] = xywh[:,0] - xywh[:,2] / 2
    ltrb[:,1] = xywh[:,1] - xywh[:,3] / 2
    ltrb[:,2] = xywh[:,0] + xywh[:,2] / 2
    ltrb[:,3] = xywh[:,1] + xywh[:,3] / 2
    if dim == 1:
        ltrb = ltrb.reshape(-1)
    return ltrb

def nonmax_suppression(dets, score_thresh=0.5, iou_thresh=0.4):
    '''检测器输出的非最大值抑制
    
    Args:
        dets (torch.Tensor): 检测器输出, dets.size()=[batch_size,
            #proposals, #dim], 其中#proposals是所有尺度输出的建议
            框数量, #dim是每个建议框的属性维度
        score_thresh (float): 置信度阈值, score_thresh∈[0,1]
        iou_thresh (float): 重叠建议框的叫并面积比阈值, iou_thresh∈[0,1]
    Returns:
        nms_dets (torch.Tensor): 经NMS的检测器输出
    '''
    nms_dets = [None for _ in range(dets.size(0))]
    for i, det in enumerate(dets):
        keep = det[:,4] > score_thresh
        det = det[keep]
        if not det.size(0):
            continue
        det[:, :4] = xywh2ltrb(det[:, :4])
        keep = nms(det[:, :4], det[:, 4], iou_thresh)
        det = det[keep]
        nms_dets[i] = det
    return nms_dets

def ltrb_net2img(boxes, net_size, img_size):
    '''将神经网络坐标系下的建议框投影到图像坐标系下
    
    Args:
        boxes (torch.Tensor): 神经网络坐标系下[l, t, r, b]格式的建议框
        net_size (tuple of int): 神经网络输入大小, net_size=(height, width)
        img_size (tuple of int): 图像大小, img_size=(height, width)
    Returns:
        boxes (torch.Tensor): 图像坐标系下[l, t, r, b]格式的建议框
    '''
    net_size = np.array(net_size)
    img_size = np.array(img_size)
    s = (net_size / img_size).min()
    simg_size = s * img_size
    dy, dx = (net_size - simg_size) / 2
    boxes[:, [0,2]] -= dx
    boxes[:, [1,3]] -= dy
    boxes /= s
    boxes = torch.clamp(boxes, min=0)
    return boxes

def overlap(dets, im):
    '''叠加检测结果到图像上
    
    Args:
        dets (torch.Tensor): 含检测结果的二维数组, dets[:]=
            [l,t,r,b,objecness,class,embedding]
        im (numpy.ndarray): BGR格式图像
    Returns:
        im (numpy,ndarray): BGR格式图像
    '''
    for det in dets:
        l, t, r, b = det[:4].round().int().numpy().tolist()
        color = np.random.randint(0, 256, size=(3,)).tolist()
        cv2.rectangle(im, (l, t), (r, b), color, 2)
    return im

last_footprint = collections.defaultdict()

def overlap_trajectory(trajectories, im):
    '''叠加跟踪轨迹到图像上
    
    Args:
        trajectories (list of Trajectory): 跟踪的轨迹列表
        im (numpy.ndarray): BGR格式图像
    Returns:
        im (numpy.ndarray): BGR格式图像
    '''
    for trajectory in trajectories:
        np.random.seed(trajectory.id)
        color = np.random.randint(0, 256, size=(3,)).tolist()
        l, t, r, b = trajectory.ltrb.round().astype(np.int32).tolist()
        im = cv2.rectangle(im, (l, t), (r, b), color, 2)
        text = '{}'.format(trajectory.id)
        text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)
        x = min(max(l, 0), im.shape[1] - text_size[0] - 1)
        y = min(max(b - baseline, text_size[1]), im.shape[0] - baseline - 1)
        im = cv2.putText(im, text, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,255), thickness=1)
        footprint = ((l + r) // 2, b)
        if trajectory.id in last_footprint.keys():
            cfp = footprint
            for lfp in reversed(last_footprint[trajectory.id]):
                im = cv2.line(im, cfp, lfp, color, thickness=2)
                cfp = lfp
        else:
            last_footprint[trajectory.id] = collections.deque([], maxlen=300)
        last_footprint[trajectory.id].append(footprint)
    return im

class TrajectoryState(IntEnum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class Trajectory(kalman.KalmanFilter):
    '''轨迹
    '''
    count = 0
    def __init__(self, ltrb, score, embedding):
        super(Trajectory, self).__init__()
        self.ltrb = ltrb
        self.xyah = ltrb2xyah(ltrb)
        self.score = score
        self.smooth_embedding = None
        self.id = 0
        self.is_activated = False
        self.eta = 0.9
        self.timestamp = 0
        self.length = 0
        self.starttime = 0
        self.update_embedding(embedding)
    
    def update_embedding(self, embedding):
        self.current_embedding = embedding / np.linalg.norm(embedding)
        if self.smooth_embedding is None:
            self.smooth_embedding = self.current_embedding
        else:
            self.smooth_embedding = self.eta * self.smooth_embedding + \
                (1 - self.eta) * self.current_embedding
        self.smooth_embedding /= np.linalg.norm(self.smooth_embedding)
    
    @staticmethod
    def next_id():
        Trajectory.count += 1
        return Trajectory.count
    
    def predict(self, *args, **kwargs):
        if self.state != TrajectoryState.Tracked:
            self.kf.statePost[7] = 0
        return super().predict(*args, **kwargs)
    
    def update(self, trajectory, timestamp, update_embedding=True):
        self.timestamp = timestamp
        self.length += 1
        # self.ltrb = trajectory.ltrb
        # self.xyah = trajectory.xyah
        mean = super().correct(trajectory.xyah)
        self.xyah = mean[:4].copy()
        self.ltrb = xyah2ltrb(self.xyah)
        self.state = TrajectoryState.Tracked
        self.is_activated = True
        self.score = trajectory.score
        if update_embedding:
            self.update_embedding(trajectory.current_embedding)
    
    def activate(self, timestamp):
        self.id = self.next_id()
        super().initialize(self.xyah)
        self.length = 0
        self.state = TrajectoryState.Tracked
        self.timestamp = timestamp
        self.starttime = timestamp
    
    def reactivate(self, trajectory, timestamp, newid=False):
        mean = super().correct(trajectory.xyah)
        self.xyah = mean[:4].copy()
        self.ltrb = xyah2ltrb(self.xyah)
        self.update_embedding(trajectory.current_embedding)
        self.length = 0
        self.state = TrajectoryState.Tracked
        self.is_activated = True
        self.timestamp = timestamp
        if newid:
            self.id = self.next_id()
    
    def mark_lost(self):
        self.state = TrajectoryState.Lost
    
    def mark_removed(self):
        self.state = TrajectoryState.Removed
    
    def timestamp(self):
        return self.timestamp

def joint_trajectories(A, B):
    '''合并两个轨迹组
    
    Args:
        A (list of Trajectory): 轨迹组A
        B (list of Trajectory): 轨迹组B
    Returns:
        C (list of Trajectory): 合并后的轨迹组
    '''
    exists = {}
    C = []
    for t in A:
        exists[t.id] = 1
        C.append(t)
    for t in B:
        if not exists.get(t.id, 0):
            exists[t.id] = 1
            C.append(t)
    return C

def exclude_trajectories(A, B):
    '''从轨迹组A中排除那些也在轨迹组B中的轨迹
    
    Args:
        A (list of Trajectory): 轨迹组A
        B (list of Trajectory): 轨迹组B
    Returns:
        C (list of Trajectory): 排除共同轨迹之后的轨迹组A
    '''
    exists = {}
    C = []
    for t in B:
        exists[t.id] = 1
    for t in A:
        if not exists.get(t.id, 0):
            C.append(t)
    return C

def embedding_distance(A, B):
    '''计算表观嵌入之间的代价矩阵
    
    Args:
        A (list of Trajectory): 轨迹组A
        B (list of Trajectory): 轨迹组B
    Returns:
        costs (numpy.ndarray): 代价矩阵
    '''
    costs = np.zeros((len(A), len(B)))
    if costs.size == 0:
        return costs
    XA = np.asarray([trajectory.smooth_embedding for trajectory in A])
    XB = np.asarray([trajectory.current_embedding for trajectory in B])
    costs = np.maximum(0, cdist(XA, XB))
    return costs

def iou_distance(A, B):
    '''计算轨迹之间的IOU距离
    
    Args:
        A (list of Trajectory): 轨迹组A
        B (list of Trajectory): 轨迹组B
    Returns:
        costs (numpy.ndarray): 代价矩阵
    '''
    BA = [a.ltrb for a in A]
    BB = [b.ltrb for b in B]
    ious = np.zeros((len(A), len(B)), dtype=np.float)
    if ious.size == 0:
        return ious   
    ious = bbox_overlaps(np.ascontiguousarray(BA, dtype=np.float),
        np.ascontiguousarray(BB, dtype=np.float))
    costs = 1 - ious
    return costs

def remove_duplicate_trajectories(A, B, thresh=0.15):
    '''从轨迹组A和轨迹组B中那些重叠的轨迹
    
    Args:
        A (list of Trajectory): 轨迹组A
        B (list of Trajectory): 轨迹组B
    Returns:
        (A,B) (tuple of list of Trajectory): 排除共同轨迹之后的轨迹组A和轨迹组B
    '''
    dists = iou_distance(A, B)
    rows, cols = np.where(dists<thresh)
    DA, DB = [], []
    for r, c in zip(rows, cols):
        ta = A[r].timestamp - A[r].starttime
        tb = B[c].timestamp - B[c].starttime
        if ta > tb:
            DB.append(c)
        else:
            DA.append(r)
    A = [a for i,a in enumerate(A) if not i in DA]
    B = [b for i,b in enumerate(B) if not i in DB]
    return A, B

def merge_mahalanobis_distance(trajectory_pool, candidates, dists, lamb=0.98):
    '''融合表观嵌入距离和马氏距离
    
    Args:
        trajectory_pool (list of Trajectory): 轨迹池
        candidates (list of Trajectory): 候选轨迹
        dists (numpy.ndarray): 轨迹池和候选轨迹之间的代价矩阵
        lamb (float, optional): 融合两种距离的权重系数
    Returns:
        dists (numpy.ndarray): 轨迹池和候选轨迹之间的代价矩阵
    '''
    if dists.size == 0:
        return dists
    gate_thresh = kalman.chi2inv95[4]
    measurements = np.asarray([candidate.xyah for candidate in candidates])
    for i,trajectory in enumerate(trajectory_pool):
        gdist = trajectory.gating_distance(measurements)
        dists[i, gdist > gate_thresh] = np.inf
        dists[i] = lamb * dists[i] + (1 - lamb) * gdist
    return dists

def linear_assignment(cost, cost_limit):
    '''线性分配
    
    Args:
        cost (numpy.ndarray): 代价矩阵
        cost_limit (float): 线性分配代价上限
    Returns:
        matches (numpy.ndarray): 一个N*2的匹配索引矩阵, N为匹配对数
            matches[i,j]表示第j个候选轨迹分配给第i个轨迹池轨迹
        mismatch_row (numpy.ndarray): 没有分配的行(轨迹池中的轨迹)
        mismatch_col (numpy.ndarray): 没有分配的列(候选集中的轨迹)
    '''
    matches = np.empty((0, 2), dtype=int)
    mismatch_row = np.arange(cost.shape[0], dtype=int)
    mismatch_col = np.arange(cost.shape[1], dtype=int)
    if cost.size == 0:
        return matches, mismatch_row, mismatch_col
    
    matches = []
    opt, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=cost_limit)
    for i,xi in enumerate(x):
        if xi >= 0:
            matches.append([i, xi])
    
    matches = np.asarray(matches)
    mismatch_row = np.where(x < 0)[0]
    mismatch_col = np.where(y < 0)[0]
    return matches, mismatch_row, mismatch_col

class JDETracker(object):
    '''联合检测和嵌入的目标跟踪器
    '''
    def __init__(self):
        self.timestamp = 0
        self.tracked_trajectories = []
        self.lost_trajectories    = []
        self.removed_trajectories = []
        self.max_lost_time = 30
    
    def update(self, dets):
        '''根据检测结果更新跟踪器
        
        Args:
            dets (numpy.ndarray): 含检测结果的二维数组, dets[:]=
            [t,l,b,r,objecness,class,embedding]
        '''
        
        self.timestamp += 1
        
        # 用检测结果初始化候选轨迹, 候选轨迹要么融入轨迹池中已有轨迹,
        # 要么成为新的轨迹加入轨迹池
        candidates = []
        for det in dets:
            candidates += [Trajectory(det[:4], det[4], det[6:])]
        
        # 构造轨迹池
        tracked_trajectories = []
        unconfirmed_trajectories = []
        for trajectory in self.tracked_trajectories:
            if trajectory.is_activated:
                tracked_trajectories.append(trajectory)
            else:
                unconfirmed_trajectories.append(trajectory)
        
        trajectory_pool = joint_trajectories(tracked_trajectories, self.lost_trajectories)
        
        # 预测轨迹池中轨迹在当前帧的状态
        for trajectory in trajectory_pool:
            trajectory.predict()
        
        # 根据表观嵌入和马氏距离关联候选轨迹和池轨迹
        dists = embedding_distance(trajectory_pool, candidates)
        dists = merge_mahalanobis_distance(trajectory_pool, candidates, dists)
        matches, mismatch_row, mismatch_col = linear_assignment(dists, cost_limit=0.7)
        
        activated_trajectories = []
        retrieved_trajectories = []
        for pid, cid in matches:
            pool_trajectory = trajectory_pool[pid]
            cand_trajectory = candidates[cid]
            if pool_trajectory.state == TrajectoryState.Tracked:
                # 如果池中的轨迹已激活, 将候选轨迹融入轨迹池
                pool_trajectory.update(cand_trajectory, self.timestamp)
                activated_trajectories.append(pool_trajectory)
            else:
                # 如果池中的轨迹处于休眠状态, 重新激活它
                pool_trajectory.reactivate(cand_trajectory, self.timestamp)
                retrieved_trajectories.append(pool_trajectory)
        
        # 根据IoU关联候选轨迹和轨迹池
        candidates = [candidates[i] for i in mismatch_col]
        # 轨迹池中一直跟踪到上一帧, 但当前帧没有匹配轨迹的那些轨迹
        mismatch_tracked_pool_trajectories = [trajectory_pool[i] \
            for i in mismatch_row if trajectory_pool[i].state == TrajectoryState.Tracked]
        dists = iou_distance(mismatch_tracked_pool_trajectories, candidates)
        matches, mismatch_row, mismatch_col = linear_assignment(dists, cost_limit=0.5)
        
        for pid, cid in matches:
            pool_trajectory = mismatch_tracked_pool_trajectories[pid]
            cand_trajectory = candidates[cid]
            if pool_trajectory.state == TrajectoryState.Tracked:
                pool_trajectory.update(cand_trajectory, self.timestamp)
                activated_trajectories.append(pool_trajectory)
            else:
                pool_trajectory.reactivate(cand_trajectory, self.timestamp)
                retrieved_trajectories.append(pool_trajectory)
        
        # 轨迹池中跟丢的轨迹
        lost_trajectories = []
        for i in mismatch_row:
            pool_trajectory = mismatch_tracked_pool_trajectories[i]
            if not pool_trajectory.state == TrajectoryState.Lost:
                pool_trajectory.mark_lost()
                lost_trajectories.append(pool_trajectory)
        
        # 跟踪过的轨迹中不确定状态的轨迹
        candidates = [candidates[i] for i in mismatch_col]
        dists = iou_distance(unconfirmed_trajectories, candidates)
        matches, mismatch_row, mismatch_col = linear_assignment(dists, cost_limit=0.7)
        for pid, cid in matches:
            unconfirmed_trajectories[pid].update(candidates[cid], self.timestamp)
            activated_trajectories.append(unconfirmed_trajectories[pid])
        
        # 跟踪过的不确定状态的轨迹中完全没有匹配上的轨迹
        removed_trajectories = []
        for i in mismatch_row:
            unconfirmed_trajectories[i].mark_removed()
            removed_trajectories.append(unconfirmed_trajectories[i])
        
        # 名花无主的候选轨迹转正, 加入轨迹库
        for i in mismatch_col:
            candidates[i].activate(self.timestamp)
            activated_trajectories.append(candidates[i])
        
        # 标记那些跟丢时间过长的轨迹
        for trajectory in self.lost_trajectories:
            if self.timestamp - trajectory.timestamp > self.max_lost_time:
                trajectory.mark_removed()
                removed_trajectories.append(trajectory)
        
        # 更新跟踪过的轨迹列表
        self.tracked_trajectories = [t for t in self.tracked_trajectories \
            if t.state == TrajectoryState.Tracked]
        self.tracked_trajectories = joint_trajectories(self.tracked_trajectories, activated_trajectories)
        self.tracked_trajectories = joint_trajectories(self.tracked_trajectories, retrieved_trajectories)
        
        # 更新跟丢的轨迹列表
        self.lost_trajectories = exclude_trajectories(self.lost_trajectories, self.tracked_trajectories)
        self.lost_trajectories.extend(lost_trajectories)
        self.lost_trajectories = exclude_trajectories(self.lost_trajectories, self.removed_trajectories)
        
        # 更新移除的轨迹列表
        self.removed_trajectories.extend(removed_trajectories)
        self.tracked_trajectories, self.lost_trajectories = remove_duplicate_trajectories(\
            self.tracked_trajectories, self.lost_trajectories)
        
        return [trajectory for trajectory in self.tracked_trajectories if trajectory.is_activated]

def save_trajectories(path, trajectories, frame_id):
    line = '{},{},{},{},{},{},1,-1,-1,-1\n'
    with open(path, 'a') as file:
        for trajectory in trajectories:
            if trajectory.id < 0:
                continue
            l = trajectory.ltrb[0]
            t = trajectory.ltrb[1]
            w = trajectory.ltrb[2] - trajectory.ltrb[0]
            h = trajectory.ltrb[3] - trajectory.ltrb[1]
            file.write(line.format(frame_id, trajectory.id, l, t, w, h))
        file.close()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    if os.path.isfile(args.config):
        config.merge_from_file(args.config)
    config.freeze()

    model = build_tracker(config.MODEL)
    if os.path.isfile(args.model):
        model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.cuda().eval()

    h, w = [int(s) for s in args.insize.split('x')]
    tracker = JDETracker()
    if os.path.isfile(args.img_path):
        dataloader = dataset.VideoLoader(args.img_path, (h,w,3))
    else:
        dataloader = dataset.ImagesLoader(args.img_path, (h,w,3), formats=['*.jpg', '*.png'])
    
    strs = re.split(r'[\\, /]', args.img_path)
    imgpath = os.path.join(args.workspace, 'result', strs[-3], 'img')
    mkdir(imgpath)
    traj_path = os.path.join(args.workspace, 'result', '{}.txt'.format(strs[-3]))
    
    os.system('rm -f {}'.format(os.path.join(imgpath, '*')))
    for i, (path, im, lb_im) in enumerate(dataloader):
        input = torch.from_numpy(lb_im).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input)
        print('{} {} {} {}'.format(path, im.shape, lb_im.shape, outputs.size()), end=' ')
        outputs =  nonmax_suppression(outputs, args.score_thresh, args.iou_thresh)[0]
        
        if outputs is None:
            print('no object detected!')
            segments = re.split(r'[\\, /]', path)
            cv2.imwrite(os.path.join(imgpath, segments[-1]), im)
            continue
        print('{}'.format(outputs.size()), end=' ')
        outputs[:, :4] = ltrb_net2img(outputs[:, :4], (h,w), im.shape[:2])
        if not args.only_detect:
            trajectories = tracker.update(outputs.numpy())
            print('{}'.format(len(trajectories)))        
            result = overlap_trajectory(trajectories, im)
            save_trajectories(traj_path, trajectories, i + 1)
        else:
            print('')
            result = overlap(outputs, im)
        segments = re.split(r'[\\, /]', path)
        cv2.imwrite(os.path.join(imgpath, segments[-1]), result)

    os.system('ffmpeg -f image2 -i {} {}.mp4 -y'.format(os.path.join(imgpath, '%06d.jpg'),
        os.path.join(args.workspace, 'result', strs[-3])))

if __name__ == '__main__':
    args = parse_args()
    main(args)