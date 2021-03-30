import os
import cv2
import lap
import sys
import copy
import math
import time
import torch
import random
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
from scipy.spatial.distance import cdist
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
    """Local images provider"""
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
    """Local tracklet"""
    def __init__(self, chan, id, box, im=None, location=None, fcount=0):
        self.chan = chan    # channel number
        self.id = id        # local tracklet identity
        self.box = box      # ltrb
        self.im = im        # clipped image
        self.feat = None    # ReID feature vector
        self.time = time.time()        # timestamp
        self.location = location       # ground plane location
        self.fcount = fcount    # frame count

class MTSCT(Thread):
    def __init__(self, tid, images, tracklets, exit, model,
        locator=None, insize=(320, 576)):
        super(MTSCT, self).__init__(name=tid)
        self.tid = tid
        self.images = images
        self.tracklets = tracklets
        self.exit = exit
        self.model = model.cuda()   # Tracker model
        self.model.eval()
        self.locator = locator      # ground plane location
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
                    xf = (track.ltrb[0] + track.ltrb[2]) / 2    # x of footprint
                    yf = track.ltrb[3]                          # y of footprint
                    location = self.locator((xf, yf))
                    tracklet = Tracklet(self.tid, track.id, track.ltrb, im, location, counter)
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
    """Global trajectory"""
    def __init__(self, id, tracklet):
        self.id = id
        if isinstance(tracklet, Tracklet):
            self.data = [tracklet]
            self.drew = [False]
        elif isinstance(tracklet, list):
            self.data = tracklet
            self.drew = [False] * len(tracklet)
        else:
            raise TypeError('expect Tracklet or list type,'
                ' but got {}'.format(type(tracklet)))

    def append(self, tracklet):
        if isinstance(tracklet, Tracklet):
            self.data.append(tracklet)
            self.drew.append(False)
        elif isinstance(tracklet, list):
            self.data += tracklet
            self.drew += [False] * len(tracklet)
        else:
            raise TypeError('expect Tracklet or list type,'
                ' but got {}'.format(type(tracklet)))
    
    def randel(self):
        if len(self.data) > 1:
            unlucky = random.randint(0, len(self.data) - 2)
            del self.data[unlucky]

class MTMCT(Thread):
    def __init__(self, tid, tracklets, trajectories, exit, model=None):
        super(MTMCT, self).__init__(name=tid)
        self.tid = tid
        self.tracklets = tracklets
        self.trajectories = trajectories
        self.exit = exit
        self.model = model  # ReID model
        self.gplane = np.zeros((2048, 2048, 3), dtype=np.uint8)
        self.counter = 0
        self.gplane_dir = osp.join('tasks', 'gplane')
        mkdirs(self.gplane_dir)
        self.local2global = defaultdict(int)    # `channel-local_id`
        self.undetermined = defaultdict(lambda: defaultdict(list))    # `channel`->`local_id`
        self.least = 5 # at least `least` tracklets for matching
        self.atmost = 15    # at most `atmost` tracklets for matching
        self.ncamera = 0
        self.max_dist = 25
        self.traj_dir = osp.join('tasks', 'trajectories')
        mkdirs(self.traj_dir)

    def run(self):
        print('start thread {} ...'.format(self.tid))
        while self.exit.value == 0:
            print('\n' * 3 + '>' * 32 + '{}'.format(self.counter))
            # Fetch tracklets from all cameras synchronously.
            tracklets = self.fetch_tracklet()
            if tracklets is None:
                print('fetch tracklet failed')
                continue
            # Extract ReID feature from clipped images.
            self.extract_feature(tracklets)
            # Init global trajectories if necessary.
            self.init_global_trajectory(tracklets)
            # Assign local tracklets to global trajectories.
            self.local_to_global_match(tracklets)
            for i in range(len(self.tracklets)):
                self.tracklets[i].task_done()
            # self.draw_trajectory()  # debug
            # self.save_trajectory()  # debug
            self.overlap_global_identity()  # debug
            self.print_trajectory() # debug
            self.counter += 1
        print('exit thread {}.'.format(self.tid))
    
    def fetch_tracklet(self):
        """Fetch tracklets from all channels"""
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
        """Extract embedding from cripped image"""
        for tracklet in tracklets:
            for t in tracklet:
                t.feat = self.model(t.im).detach().cpu().numpy().reshape(1, -1)
    
    def init_global_trajectory(self, tracklets):
        """Initialize global trajectories"""
        if len(self.trajectories) > 0:
            return False
        for channel, tracklet in enumerate(tracklets):
            for i, t in enumerate(tracklet):
                id = i + 1  # global identity
                self.trajectories.append(Trajectory(id, t))
                self.update_match_table(t, id)
            # Choose the first channel which contains local tracklet.
            if len(self.trajectories) > 0:
                tracklets[channel] = []   # no need for processing again
                break
        return True
    
    def update_match_table(self, tracklet, id):
        """Update local2global table"""
        key = '{}-{}'.format(tracklet.chan, tracklet.id)
        mid = self.local2global.get(key, None)
        if mid is None:
            self.local2global[key] = id
        else:
            print('find the key `{}` exists!!!'.format(key))
    
    def local_to_global_match(self, tracklets):
        """Assign local tracklet to global trajectory"""
        for channel, tracklet in enumerate(tracklets):
            # Look up local to global matching table.
            matched = self.direct_match(channel, tracklet)
            print('matched {}'.format(matched))
            gallery = self.make_gallery(matched)
            undetermined = self.make_undetermined(channel)
            # Appearance distance matrix.
            appr = self.calc_appr_dist(undetermined, gallery)
            # Linear assignment trajectories
            m, gum, lum = self.linear_assignment(appr, cost_limit=0.9)
            # Update global trajectories and local2global table.
            self.update_trajectory(gallery, undetermined, m)
            # Build global trajectory.
            self.build_trajectory(undetermined, lum)
            # Clean undetermined items which have been processed.
            self.clean_undermined(channel, undetermined)
            print('undetermined{}: {}'.format(channel,
                list(self.undetermined[channel].keys())))
            print('-' * 32)
        self.local2local_dist(tracklets)
    
    def make_gallery(self, exclude=[]):
        """Make gallery set"""
        gallery = []
        for t in self.trajectories:
            if t.id in exclude:
                continue
            # Require enough samples for each trajectory.
            if len(t.data) < self.least:
                continue
            # Limit gallery set size.
            k = min(len(t.data), self.atmost)
            data = random.sample(t.data, k)
            gallery.append({'id': t.id, 'data': data})
        return gallery
    
    def make_undetermined(self, channel):
        undetermined = {}
        for k, v in self.undetermined[channel].items():
            if len(v) >= self.least:
                undetermined[k] = v
        return undetermined
    
    def direct_match(self, channel, tracklets):
        """Matching based on local2global table"""
        matched = []
        for t in tracklets:
            mid = None
            key = '{}-{}'.format(t.chan, t.id)
            mid = self.local2global.get(key, None)
            if mid is not None:
                for k, gt in enumerate(self.trajectories):
                    if gt.id == mid:
                        print('direct assign {} to {}'.format(key, mid))
                        self.trajectories[k].append(t)
                        break
                matched.append(mid)
            else:
                # Accumulate undetermined local tracklets.
                self.undetermined[channel][t.id].append(t)
                print('add undetermined {}'.format(key))
        return matched
    
    def calc_appr_dist(self, undetermined, gallery):
        """Calculate appearance distance"""
        dists = []
        for gt in gallery:
            dist = []
            for id, lt in undetermined.items():
                # Distance between gallery and probe set.
                d = []
                for gti in gt['data']:
                    for lti in lt:
                        d.append(cdist(gti.feat, lti.feat))
                d = np.mean(d).item()
                dist.append(d)
            dists.append(dist)
        dists = np.array(dists)
        if dists.size == 0:
            return dists.reshape(0, 0)
        # print('probe to gallery dists:\n{}'.format(dists))   
        dists[dists > self.max_dist] = self.max_dist
        dists /= self.max_dist
        # print('norm dists:\n{}'.format(dists))
        return dists
    
    def linear_assignment(self, cost, cost_limit):
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

    def update_trajectory(self, gallery, undetermined, m):
        if m.size == 0:
            return
        for i, (lid, lt) in enumerate(undetermined.items()):
            j = np.where(m[:, 1] == i)[0]
            if j.size == 0:
                continue
            j = m[j, 0].item() # Find the matched gallery item
            for gt in self.trajectories:
                if gt.id == gallery[j]['id']:
                    gt.append(lt)
                    self.update_match_table(lt[0], gt.id)
                    print('add {}-{} to {}, length {}'.format(
                        lt[0].chan, lt[0].id, gt.id, len(lt)))
                    break
            else:
                print('impossible error happended!!!')

    def build_trajectory(self, undetermined, lum):
        """Build new trajectory from tracklet"""
        for i, (lid, lt) in enumerate(undetermined.items()):
            if i not in lum:    # local unmatched
                continue
            gid = self.trajectories[-1].id + 1   # global identity
            self.trajectories.append(Trajectory(gid, lt))
            self.update_match_table(lt[0], gid)
            print('build trajectory {} from {}-{}'.format(
                gid, lt[0].chan, lt[0].id))

    def clean_undermined(self, channel, undetermined):
        """Clean undetermined tracklets"""
        for key in undetermined.keys():
            self.undetermined[channel].pop(key, None)
    
    def local2local_dist(self, tracklets):
        dists = {}
        for t1 in tracklets[0]:
            for t2 in tracklets[1]:
                key = '{}-{}:{}-{}'.format(t1.chan, t1.id, t2.chan, t2.id)
                dists[key] = cdist(t1.feat, t2.feat).item()
        print('local2local_dist:\n{}'.format(dists))
    
    def draw_trajectory(self):
        radius = 2
        for trajectory in self.trajectories:
            for foot in trajectory.data:
                xw, yw = foot.location
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
        cv2.imwrite(osp.join(self.gplane_dir, '%06d.png' % self.counter), self.gplane)
    
    def save_trajectory(self):
        for i, trajectory in enumerate(self.trajectories):
            dir = osp.join(self.traj_dir, '%06d' % trajectory.id)
            mkdirs(dir)
            for j, tracklet in enumerate(trajectory.data):
                fname = osp.join(dir, '%06d.%d.jpg' % (j, tracklet.chan))
                cv2.imwrite(fname, tracklet.im)
    
    def overlap_global_identity(self):
        for trajectory in self.trajectories:
            text = '{}'.format(trajectory.id)
            text_size, baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, 1)
            for i, tracklet in enumerate(trajectory.data):
                if trajectory.drew[i]:
                    continue
                path = osp.join('tasks',
                    'mtsct_{}'.format(tracklet.chan),
                    '%06d.jpg' % tracklet.fcount)
                im = cv2.imread(path)
                box = tracklet.box
                l, t, r, b = box.round().astype(np.int32).tolist()
                x = min(max(r - text_size[0], 0), im.shape[1] - text_size[0] - 1)
                y = min(max(t + text_size[1], text_size[1]), im.shape[0] - baseline - 1)
                im = cv2.putText(im, text, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    2, (0,255,255), thickness=1)
                cv2.imwrite(path, im)
                trajectory.drew[i] = True
    
    def print_trajectory(self):
        for t in self.trajectories:
            print('gid:{}, len:{}'.format(t.id, len(t.data)))

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
        ndead = sum([int(not t.is_alive()) for t in threads[:-1][0::2]])
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