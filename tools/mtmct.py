import os
import cv2
import sys
import copy
import math
import time
import torch
import argparse
import warnings
import threading
import numpy as np
import os.path as osp
from queue import Queue
from threading import Thread
from multiprocessing import Value
import xml.etree.ElementTree as ET
from collections import defaultdict
from torchreid.utils import FeatureExtractor
sys.path.append(os.getcwd())

import dataset
import trackernew
from mot.utils import (config, mkdirs)
from mot.models import build_tracker

def parse_calibration_data(fpath):
    """Load camera calibration parameters from .xml file"""
    tree = ET.parse(fpath)
    root = tree.getroot()
    params = {}
    for child in root:
        params[child.tag] = child.attrib
    for k1, v1 in params.items():
        for k2, v2 in v1.items():
            params[k1][k2] = float(v2)
    return params

class ImageToWorldTsai:
    '''Convert image coordinate to world coordinate with Tsai camera model.
       http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/DIAS1/
    '''
    def __init__(self, calib):
        # Extrinsic parameters.
        self.tx = calib['Extrinsic']['tx']
        self.ty = calib['Extrinsic']['ty']
        self.tz = calib['Extrinsic']['tz']
        self.rx = calib['Extrinsic']['rx']
        self.ry = calib['Extrinsic']['ry']
        self.rz = calib['Extrinsic']['rz']
        # Intrinsic parameters.
        self.focal  = calib['Intrinsic']['focal']
        self.kappa1  = calib['Intrinsic']['kappa1']
        self.cx = calib['Intrinsic']['cx']
        self.cy = calib['Intrinsic']['cy']
        self.sx  = calib['Intrinsic']['sx']
        # Geometry parameters.
        self.width = calib['Geometry']['width']
        self.height = calib['Geometry']['height']
        self.ncx = calib['Geometry']['ncx']
        self.nfx = calib['Geometry']['nfx']
        self.dx = calib['Geometry']['dx']
        self.dy = calib['Geometry']['dy']
        self.dpx = calib['Geometry']['dpx']
        self.dpy = calib['Geometry']['dpy']
        # Rotation matrix
        sinrx = math.sin(self.rx)
        cosrx = math.cos(self.rx)
        sinry = math.sin(self.ry)
        cosry = math.cos(self.ry)
        sinrz = math.sin(self.rz)
        cosrz = math.cos(self.rz)
        self.r1 = cosry * cosrz
        self.r2 = cosrz * sinrx * sinry - cosrx * sinrz
        self.r3 = sinrx * sinrz + cosrx * cosrz * sinry
        self.r4 = cosry * sinrz
        self.r5 = sinrx * sinry * sinrz + cosrx * cosrz
        self.r6 = cosrx * sinry * sinrz - cosrz * sinrx
        self.r7 = -sinry
        self.r8 = cosry * sinrx
        self.r9 = cosrx * cosry
    
    def __call__(self, pi):
        # Assuming that zw = 0
        xf, yf = pi
        # Image co-ordinates to distorted co-ordinates in image plane.
        xd = (xf - self.cx) * self.dpx / self.sx
        yd = (yf - self.cy) * self.dpy
        # Distorted co-ordinates to undistorted co-ordinates in image plane.
        r2 = xd * xd + yd * yd
        xu = xd * (1 + self.kappa1 * r2)
        yu = yd * (1 + self.kappa1 * r2)
        # Undistorted co-ordinates to 3D world co-ordinates.
        a = xu / self.focal
        b = yu / self.focal
        A = np.array([[self.r1, self.r2, -a], [self.r4, self.r5, -b],
            [self.r7, self.r8, -1]], dtype=np.float)
        B = np.array([-self.tx, -self.ty, -self.tz], dtype=np.float)
        X = np.linalg.solve(A, B)   # solve A*X=B, X = [xw, yw, zi]
        return tuple(X[:2].tolist())

class Datastore(Thread):
    def __init__(self, tid, data_path, images, insize=(320, 576, 3)):
        super(Datastore, self).__init__(name=tid)
        self.tid = tid
        self.data_path = data_path
        self.images = images
        self.insize = insize
        self.dataset = dataset.ImagesLoader(data_path,
            insize, formats=['*.jpg', '*.png'])
    
    def run(self):
        print('start thread {} ...'.format(self.tid))
        for path, im, lb_im in self.dataset:
            lb_im = torch.from_numpy(lb_im).unsqueeze(0).cuda()
            data = {'path': path, 'raw_img': im, 'lb_im': lb_im}
            self.images.put(data)
            time.sleep(0.04)
        print('exit thread {}.'.format(self.tid))

class Tracklet:
    def __init__(self, chan, id, box, im=None, location=None):
        self.chan = chan    # channel index
        self.id = id        # local tracklet index
        self.box = box      # ltrb
        self.im = im        # clipped image
        self.feat = None    # ReID feature vector
        self.time = time.time()        # timestamp
        self.location = location       # ground plane location

class MTSCT(Thread):
    def __init__(self, tid, images, tracklets, exit, model,
        locator=None, insize=(320, 576)):
        super(MTSCT, self).__init__(name=tid)
        self.tid = tid
        self.images = images
        self.tracklets = tracklets
        self.exit = exit
        self.model = model.cuda()
        self.model.eval()
        self.locator = locator
        self.insize = insize
        self.score_thresh = 0.5
        self.iou_thresh = 0.4
        self.tracker = trackernew.JDETracker()
        self.save_dir = osp.join('tasks', 'mtsct_{}'.format(self.tid))
        mkdirs(self.save_dir)
    
    def run(self):
        print('start thread {} ...'.format(self.tid))
        counter = 0
        while self.exit.value == 0:
            # Fetch original image.
            try:
                data = self.images.get(timeout=1)
            except:
                continue
            path = data['path']
            raw_img = data['raw_img']
            input = data['lb_im']
            # Track local tracklets.
            with torch.no_grad():
                outputs = self.model(input)
            outputs = trackernew.nonmax_suppression(outputs,
                self.score_thresh, self.iou_thresh)[0]
            tracklets = []  # for alignment, even empty tracklets is necessary.
            resimg = raw_img
            if outputs is not None:
                outputs[:, :4] = trackernew.ltrb_net2img(
                    outputs[:, :4], self.insize, raw_img.shape[:2])
                tracks = self.tracker.update(outputs.numpy())
                # Package local tracklets.
                for i, track in enumerate(tracks):
                    im = self.clip(track.ltrb, raw_img)
                    xf = (track.ltrb[0] + track.ltrb[2]) / 2
                    yf = track.ltrb[3]
                    location = self.locator((xf, yf))
                    tracklet = Tracklet(self.tid, track.id, track.ltrb, im, location)
                    tracklets.append(tracklet)
                resimg = trackernew.overlap_trajectory(tracks, raw_img)
            # Update tracklet queue.
            self.tracklets.put(tracklets)
            cv2.imwrite(osp.join(self.save_dir, '%06d.jpg' % counter), resimg)
            self.images.task_done()
            counter += 1
        print('exit thread {}.'.format(self.tid))
    
    def clip(self, ltrb, im):
        l, t, r, b = ltrb.round().astype(np.int32).tolist()
        l = np.clip(l, 0, im.shape[1] - 1)
        t = np.clip(t, 0, im.shape[0] - 1)
        r = np.clip(r, 0, im.shape[1] - 1)
        b = np.clip(b, 0, im.shape[0] - 1)
        return copy.deepcopy(im[t : b + 1, l : r + 1, :])

class Trajectory:
    def __init__(self, id, location, time):
        self.id = id
        self.data = [{'location': location, 'time': time}]

    def append(self, location, time):
        self.data.append({'location': location, 'time': time})

class MTMCT(Thread):
    def __init__(self, tid, tracklets, trajectories, exit, model=None):
        super(MTMCT, self).__init__(name=tid)
        self.tid = tid
        self.tracklets = tracklets
        self.trajectories = trajectories
        self.exit = exit
        self.model = model
        self.gplane = np.zeros((2048, 2048, 3), dtype=np.uint8)
        self.counter = 0
        self.save_dir = osp.join('tasks', 'gplane')
        mkdirs(self.save_dir)
        self.global2local = defaultdict(defaultdict)

    def run(self):
        print('start thread {} ...'.format(self.tid))
        while self.exit.value == 0:
            # Fetch tracklets from all cameras.
            tracklets = self.fetch_tracklet()
            if tracklets is None:
                print('fetch tracklet failed')
                continue
            tracklets = [t for t in tracklets if len(t) > 0]
            # self.draw_all_tracklets(tracklets)
            # Extract ReID feature from clipped images.
            self.extract_feature(tracklets)
            # Init global trajectories if necessary.
            self.init_global_trajectory(tracklets)
            # Look up local to global matching table.
            self.try_direct_match(tracklets)
            # Construct cost matrix.
            
            # Linear assignment trajectories
            
            # Update local to global matching table.
            
            # Init new global trajectories.

            for i in range(len(self.tracklets)):
                self.tracklets[i].task_done()
            self.counter += 1
            self.draw_trajectories()
            self.print_trajectory()
        print('exit thread {}.'.format(self.tid))
    
    def fetch_tracklet(self):
        tracklets = []
        for i in range(len(self.tracklets)):
            ntry = 10
            ti = None
            while self.exit.value == 0 and ntry > 0:
                try:
                    ti = self.tracklets[i].get(timeout=1)
                    break
                except:
                    time.sleep(0.005)
                    ntry -= 1
                    continue
            tracklets.append(ti)
        if None in tracklets:
            return None
        return tracklets
    
    def extract_feature(self, tracklets):
        for tracklet in tracklets:  # one channel
            for t in tracklet:
                t.feat = self.model(t.im)
    
    def init_global_trajectory(self, tracklets):
        if len(self.trajectories) > 0:
            return
        for tracklet in tracklets:  # one channel
            for j, t in enumerate(tracklet):
                id = j + 1  # global index
                self.trajectories.append(
                    Trajectory(id, t.location, t.time))
                self.global2local[t.chan][id] = [t.id]
            break
    
    def try_direct_match(self, tracklets):
        if len(self.trajectories) <= 0:
            return
        for tracklet in tracklets:  # tracklets of a channel
            for t in tracklet:
                mid = None
                if t.chan in self.global2local.keys():
                    for id, assigns in self.global2local[t.chan].items():
                        if t.id in assigns:
                            mid = id
                            break
                if mid is not None:
                    for k, gt in enumerate(self.trajectories):
                        if gt.id == mid:
                            self.trajectories[k].append(t.location, t.time)
                            break
                else:
                    pass
    
    def draw_all_tracklets(self, tracklets):
        radius = 5
        self.gplane.fill(0)
        for i, ti in enumerate(tracklets):
            for j, tij in enumerate(ti):
                xw, yw = tij.location
                xw = np.clip(xw, -20000, 20000)
                yw = np.clip(yw, -20000, 20000)
                xi = (xw / 20000) * (self.gplane.shape[1] / 2) + \
                    self.gplane.shape[1] / 2
                yi = (yw / 20000) * (self.gplane.shape[0] / 2) + \
                    self.gplane.shape[0] / 2
                center = (int(xi), int(yi))
                np.random.seed(tij.id + i * 10000)
                color = np.random.randint(0, 256, size=(3,)).tolist()
                cv2.circle(self.gplane, center, radius, color, cv2.FILLED)
        cv2.imwrite(osp.join(self.save_dir, '%06d.png' % self.counter), self.gplane)
    
    def draw_trajectories(self):
        radius = 5
        for trajectory in self.trajectories:
            for foot in trajectory.data:
                xw, yw = foot['location']
                xw = np.clip(xw, -20000, 20000)
                yw = np.clip(yw, -20000, 20000)
                xi = (xw / 20000) * (self.gplane.shape[1] / 2) + \
                    self.gplane.shape[1] / 2
                yi = (yw / 20000) * (self.gplane.shape[0] / 2) + \
                    self.gplane.shape[0] / 2
                center = (int(xi), int(yi))
                np.random.seed(trajectory.id)
                color = np.random.randint(0, 256, size=(3,)).tolist()
                cv2.circle(self.gplane, center, radius, color, cv2.FILLED)
        cv2.imwrite(osp.join(self.save_dir, '%06d.png' % self.counter), self.gplane)
    
    def print_trajectory(self):
        for t in self.trajectories:
            # for ti in t.data:
            #     print('{} {} {}'.format(t.id, ti['location'], ti['time']))
            print('{} {}'.format(t.id, len(t.data)))

def parse_args():
    parser = argparse.ArgumentParser(
        description='Multiple targets multiple cameras tracking')
    parser.add_argument('--inputs', '-inp', type=str, nargs='+',
        help='multi-camera input data')
    parser.add_argument('--config', type=str, default='',
        help='training configuration file path')
    parser.add_argument('--tracker', type=str,
        help='path to tracking model')
    parser.add_argument('--reid', type=str,
        help='path to ReID model')
    parser.add_argument('--calibration', '-cal', type=str, nargs='+',
        help='multi-camera calibration data')
    return parser.parse_args()

def main():
    args = parse_args()    
    if osp.isfile(args.config):
        config.merge_from_file(args.config)
    config.freeze()
    
    calibs = [parse_calibration_data(f) for f in args.calibration]
    print('calibration:\n{}'.format(calibs))
    
    # Create input and output queue for each tracker.
    ncamera = len(args.inputs)    
    images = []     # image queues
    tracklets = []  # tracklet queues
    for i in range(ncamera):
        images.append(Queue(maxsize=0))
        tracklets.append(Queue(maxsize=0))
    trajectories = [] # global trajectory list
    
    # Feature extractor.
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path=args.reid,
        device='cuda')
    
    # Create working threads.
    tid = 0
    threads = []
    exit = Value('i', 0)    # shared thread exit switch
    for i in range(ncamera):
        # Datastore thread.
        tid += 1
        threads.append(Datastore(tid, args.inputs[i], images[i]))
        # MTSCT thread.
        tid += 1
        model = build_tracker(config.MODEL)
        model.load_state_dict(torch.load(args.tracker, map_location='cpu'))
        locator = ImageToWorldTsai(calibs[i])
        threads.append(MTSCT(tid, images[i], tracklets[i], exit, model, locator))
    # MTMCT thread.
    tid += 1
    threads.append(MTMCT(tid, tracklets, trajectories, exit, extractor))

    # Start all threads.
    for thread in threads:
        thread.start()
    
    # Waiting for Datastore finish.
    ndead = 0
    while ndead != ncamera:
        ndead = sum([int(not t.is_alive()) for t in threads[0::2]])
        time.sleep(1)
    print('Datastore done.')
    
    # Waiting for MTSCT finish.
    nempty = 0
    while nempty != ncamera:
        nempty = sum([int(q.empty()) for q in images])
        time.sleep(1)
    print('MTSCT done.')
    
    # Waiting for MTMCT finish.
    nempty = 0
    while nempty != ncamera:
        nempty = sum([int(q.empty()) for q in tracklets])
        time.sleep(1)
    print('MTMCT done.')
    
    exit.value = 1
    for thread in threads:
        thread.join()
    print('All works done.')

if __name__ == '__main__':
    main()