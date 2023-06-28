# General utilities
import os
from tqdm import tqdm
from time import time
from fastprogress import progress_bar
from collections import Counter
import gc
from  torchvision import utils as vutils
from sklearn.cluster import DBSCAN
from torchvision.transforms import Resize
import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import h5py
from IPython.display import clear_output
from collections import defaultdict
from copy import deepcopy
import random
import math
import matplotlib.pyplot as plt
# CV/ML
import cv2
import torch
import torch.nn.functional as F
import kornia as K
import kornia.feature as KF
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# 3D reconstruction
# import pycolmap
import sys
sys.path.append('..')
from superglue.models.matching import Matching
from superglue.models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)
from OANetmaster.demo.learnedmatcher import LearnedMatcher
print('Kornia version', K.__version__)
# print('Pycolmap version', pycolmap.__version__)

LOCAL_FEATURE = 'DISK'
device = torch.device('cuda')


# Can be LoFTR, KeyNetAffNetHardNet, or DISK
def arr_to_str(a):
    return ';'.join([str(x) for x in a.reshape(-1)])


def load_torch_image(fname, device=torch.device('cpu')):
    img = K.image_to_tensor(cv2.imread(fname), False).float() / 255.
    img = K.color.bgr_to_rgb(img.to(device))
    return img


# We will use ViT global descriptor to get matching shortlists.
def get_global_desc(fnames, model,
                    device=torch.device('cpu')):
    model = model.eval()
    model = model.to(device)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    global_descs_convnext = []
    for i, img_fname_full in tqdm(enumerate(fnames), total=len(fnames)):
        key = os.path.splitext(os.path.basename(img_fname_full))[0]
        img = Image.open(img_fname_full).convert('RGB')
        timg = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            desc = model.forward_features(timg.to(device)).mean(dim=(-1, 2))  #
            # print (desc.shape)
            desc = desc.view(1, -1)
            desc_norm = F.normalize(desc, dim=1, p=2)
        # print (desc_norm)
        global_descs_convnext.append(desc_norm.detach().cpu())
    global_descs_all = torch.cat(global_descs_convnext, dim=0)
    return global_descs_all


def get_img_pairs_exhaustive(img_fnames):
    index_pairs = []
    for i in range(len(img_fnames)):
        for j in range(i + 1, len(img_fnames)):
            index_pairs.append((i, j))
    return index_pairs


def get_image_pairs_shortlist(fnames,
                              sim_th=0.6,  # should be strict
                              min_pairs=5,
                              exhaustive_if_less=20,
                              device=torch.device('cpu')):
    num_imgs = len(fnames)

    if num_imgs <= exhaustive_if_less:
        return get_img_pairs_exhaustive(fnames)

    model = timm.create_model('tf_efficientnet_b7',
                              checkpoint_path='../params/tf_efficientnet_b7_ra-6c08e654.pth')
    model.eval()
    descs = get_global_desc(fnames, model, device=device)
    dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()
    # removing half
    mask = dm <= sim_th
    total = 0
    matching_list = []
    ar = np.arange(num_imgs)
    already_there_set = []
    for st_idx in range(num_imgs - 1):
        mask_idx = mask[st_idx]
        #to_match = ar[mask_idx]
        to_match = np.argsort(dm[st_idx])[:min_pairs]
        # if len(to_match) < min_pairs:
        #     to_match = np.argsort(dm[st_idx])[:min_pairs]
        for idx in to_match:
            if st_idx == idx:
                continue
            if dm[st_idx, idx] < 1:
                matching_list.append(tuple(sorted((st_idx, idx.item()))))
                total += 1
    matching_list = sorted(list(set(matching_list)))
    return matching_list


# Code to manipulate a colmap data```base.
# Forked from https://github.com/colmap/colmap/blob/dev/scripts/python/database.py

# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

# This script is based on an original implementation by True Price.

import sys
import sqlite3
import numpy as np

IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2 ** 31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.zeros(4), prior_t=np.zeros(3), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert (len(keypoints.shape) == 2)
        assert (keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def add_matches(self, image_id1, image_id2, matches):
        assert (len(matches.shape) == 2)
        assert (matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def add_two_view_geometry(self, image_id1, image_id2, matches,
                              F=np.eye(3), E=np.eye(3), H=np.eye(3), config=2):
        assert (len(matches.shape) == 2)
        assert (matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
                                          array_to_blob(F), array_to_blob(E), array_to_blob(H)))


# Code to interface DISK with Colmap.
# Forked from https://github.com/cvlab-epfl/disk/blob/37f1f7e971cea3055bb5ccfc4cf28bfd643fa339/colmap/h5_to_db.py

#  Copyright [2020] [Michał Tyszkiewicz, Pascal Fua, Eduard Trulls]
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os, argparse, h5py, warnings
import numpy as np
from tqdm import tqdm
from PIL import Image, ExifTags


def get_focal(image_path, err_on_default=False):
    image = Image.open(image_path)
    max_size = max(image.size)

    exif = image.getexif()
    focal = None
    if exif is not None:
        focal_35mm = None
        # https://github.com/colmap/colmap/blob/d3a29e203ab69e91eda938d6e56e1c7339d62a99/src/util/bitmap.cc#L299
        for tag, value in exif.items():
            focal_35mm = None
            if ExifTags.TAGS.get(tag, None) == 'FocalLengthIn35mmFilm':
                focal_35mm = float(value)
                break

        if focal_35mm is not None:
            focal = focal_35mm / 35. * max_size

    if focal is None:
        if err_on_default:
            raise RuntimeError("Failed to find focal length")

        # failed to find it in exif, use prior
        FOCAL_PRIOR = 1.2
        focal = FOCAL_PRIOR * max_size

    return focal


def create_camera(db, image_path, camera_model):
    image = Image.open(image_path)
    width, height = image.size

    focal = get_focal(image_path)

    if camera_model == 'simple-pinhole':
        model = 0  # simple pinhole
        param_arr = np.array([focal, width / 2, height / 2])
    if camera_model == 'pinhole':
        model = 1  # pinhole
        param_arr = np.array([focal, focal, width / 2, height / 2])
    elif camera_model == 'simple-radial':
        model = 2  # simple radial
        param_arr = np.array([focal, width / 2, height / 2, 0.1])
    elif camera_model == 'opencv':
        model = 4  # opencv
        param_arr = np.array([focal, focal, width / 2, height / 2, 0., 0., 0., 0.])

    return db.add_camera(model, width, height, param_arr)


def add_keypoints(db, h5_path, image_path, img_ext, camera_model, single_camera=True):
    keypoint_f = h5py.File(os.path.join(h5_path, 'keypoints.h5'), 'r')

    camera_id = None
    fname_to_id = {}
    for filename in tqdm(list(keypoint_f.keys())):
        keypoints = keypoint_f[filename][()]

        fname_with_ext = filename.replace('\\', '/')  # + img_ext
        path = os.path.join(image_path, fname_with_ext)
        if not os.path.isfile(path):
            raise IOError(f'Invalid image path {path}')

        if camera_id is None or not single_camera:
            camera_id = create_camera(db, path, camera_model)
        image_id = db.add_image(fname_with_ext, camera_id)
        fname_to_id[filename] = image_id

        db.add_keypoints(image_id, keypoints)

    return fname_to_id


def add_matches(db, h5_path, fname_to_id):
    match_file = h5py.File(os.path.join(h5_path, 'matches.h5'), 'r')

    added = set()
    n_keys = len(match_file.keys())
    n_total = (n_keys * (n_keys - 1)) // 2

    with tqdm(total=n_total) as pbar:
        for key_1 in match_file.keys():
            group = match_file[key_1]
            for key_2 in group.keys():
                id_1 = fname_to_id[key_1]
                id_2 = fname_to_id[key_2]

                pair_id = image_ids_to_pair_id(id_1, id_2)
                if pair_id in added:
                    warnings.warn(f'Pair {pair_id} ({id_1}, {id_2}) already added!')
                    continue

                matches = group[key_2][()]
                db.add_matches(id_1, id_2, matches)

                added.add(pair_id)

                pbar.update(1)


# Making kornia local features loading w/o internet
class KeyNetAffNetHardNet(KF.LocalFeature):
    """Convenience module, which implements KeyNet detector + AffNet + HardNet descriptor.

    .. image:: _static/img/keynet_affnet.jpg
    """

    def __init__(
            self,
            num_features: int = 5000,
            upright: bool = False,
            device=torch.device('cpu'),
            scale_laf: float = 1.0,
    ):
        ori_module = KF.PassLAF() if upright else KF.LAFOrienter(angle_detector=KF.OriNet(False)).eval()
        if not upright:
            weights = torch.load('../params/OriNet.pth')['state_dict']
            ori_module.angle_detector.load_state_dict(weights)
        detector = KF.KeyNetDetector(
            False, num_features=num_features, ori_module=ori_module, aff_module=KF.LAFAffNetShapeEstimator(False).eval()
        ).to(device)
        kn_weights = torch.load('../params/keynet_pytorch.pth')['state_dict']
        detector.model.load_state_dict(kn_weights)
        affnet_weights = torch.load('../params/AffNet.pth')['state_dict']
        detector.aff.load_state_dict(affnet_weights)

        hardnet = KF.HardNet(False).eval()
        hn_weights = torch.load('../params/HardNetLib.pth')['state_dict']
        hardnet.load_state_dict(hn_weights)
        descriptor = KF.LAFDescriptor(hardnet, patch_size=32, grayscale_descriptor=True).to(device)
        super().__init__(detector, descriptor, scale_laf)


def detect_features(img_fnames,
                    num_feats=256,
                    upright=False,
                    device=torch.device('cpu'),
                    feature_dir='.featureout',
                    resize_small_edge_to=600,name = 'None'):
    # the boundary of cropped images
    l = 0
    r = 0
    u = 0
    d = 0
    try:
        if LOCAL_FEATURE == 'DISK':
            # Load DISK from Kaggle models so it can run when the notebook is offline.

            #disk = KF.DISK.from_pretrained('depth').to(device)
            # disk = KF.DISK.from_pretrained('depth').to(device)
            # disk.eval()
            device = torch.device('cuda:0')
            disk = KF.DISK().to(device)
            pretrained_dict = torch.load('../params/depth-save.pth', map_location=device)
            disk.load_state_dict(pretrained_dict['extractor'])
            disk.eval()
        if LOCAL_FEATURE == 'KeyNetAffNetHardNet':
            feature = KeyNetAffNetHardNet(num_feats, upright, device).to(device).eval()
        if LOCAL_FEATURE == 'Crop_DISK':
            num_feats = 512
            # disk = KF.DISK.from_pretrained('depth').to(device)
            # disk.eval()
            device = torch.device('cuda:0')
            disk = KF.DISK().to(device)
            pretrained_dict = torch.load('../params/depth-save.pth', map_location=device)
            disk.load_state_dict(pretrained_dict['extractor'])
            disk.eval()
        if LOCAL_FEATURE == 'Crop_KeyNetAffNetHardNet':
            num_feats = 512
            feature = KeyNetAffNetHardNet(num_feats, upright, device).to(device).eval()
        if not os.path.isdir(feature_dir):
            os.makedirs(feature_dir)
        kp_length = {'Class': 'First'}
        with h5py.File(f'{feature_dir}/lafs_{LOCAL_FEATURE}.h5', mode='w') as f_laf, \
                h5py.File(f'{feature_dir}/keypoints_{LOCAL_FEATURE}.h5', mode='w') as f_kp, \
                h5py.File(f'{feature_dir}/descriptors_{LOCAL_FEATURE}.h5', mode='w') as f_desc:
            for img_path in progress_bar(img_fnames):
                img_fname = img_path.split('/')[-1]
                key = img_fname
                with torch.inference_mode():
                    timg = load_torch_image(img_path, device=device)
                    H, W = timg.shape[2:]
                    if resize_small_edge_to is None:
                        timg_resized = timg
                    else:
                        timg_resized = K.geometry.resize(timg, resize_small_edge_to, antialias=True)
                        print(f'Resized {timg.shape} to {timg_resized.shape} (resize_small_edge_to={resize_small_edge_to})')
                    h, w = timg_resized.shape[2:]
                    if LOCAL_FEATURE == 'DISK':
                        # print("DISK-SIZE", timg_resized.shape)
                        features = disk(timg_resized, num_feats, pad_if_not_divisible=True)[0]
                        kps1, descs = features.keypoints, features.descriptors
                        # print(kps1[0])
                        lafs = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=device))
                    if LOCAL_FEATURE == 'KeyNetAffNetHardNet':
                        lafs, resps, descs = feature(K.color.rgb_to_grayscale(timg_resized))
                    if LOCAL_FEATURE == 'Crop_DISK' or LOCAL_FEATURE =='Crop_KeyNetAffNetHardNet':
                        with h5py.File(f'{feature_dir}/'+name, mode='r+') as f_kp_cropdisk:
                            f1 = f_kp_cropdisk[key][:]
                            print("第二步：计算半径为")
                            MinPts = 12
                            eps = radius(f1, MinPts)
                            # print("第三步：DBSCAN算法")
                            # types, sub_class = dbscan(f1, eps, MinPts)
                            #DASCAN划分后画图函数
                            #draw_DASCAN(f1, sub_class)

                            model = DBSCAN(eps=eps, min_samples=10)
                            model.fit(f1)
                            sub_class = model.fit_predict(f1)

                            # plt.figure()
                            # plt.scatter(f1[:, 0], f1[:, 1], c=sub_class)
                            # plt.show()


                            sub_class = np.array(sub_class)
                            sub_class =sub_class.reshape(-1)
                            #获取最多分类点的值
                            main_num =  Counter(sub_class).most_common(1)[0][0]
                            #在此类的点的序号
                            index = np.argwhere(sub_class == main_num)
                            f2 =[]
                            for i in index:
                                f2_member = [f1[i][0][0]*float(w) / float(W), f1[i][0][1]*float(h) / float(H)]
                                f2.append(f2_member)
                            f2 = np.array(f2)
                            f2 =torch.tensor(f2).view(len(index),2)
                            #计算上下左右四个边界坐标
                            r = int(torch.max(f2[:, 0]).item())
                            l = int(torch.min(f2[:, 0]).item())
                            u = int(torch.max(f2[:, 1]).item())
                            d = int(torch.min(f2[:, 1]).item())
                            # print("unresized", timg_resized.shape)
                            if ((u - d) > 100) or ((r - l) > 100):
                            #重新裁剪图像
                                timg_resized = timg_resized[:, :, d:u]
                                timg_resized = timg_resized[:, :, :, l:r]
                                # print("resized", timg_resized.shape)
                                # a = timg.contiguous().view(3, r-l, d-u)
                                # print("a", a.shape)
                                # a = a.permute(1, 2, 0)
                                # a = a.numpy()
                                #
                                # cv2.imshow("111", a)
                                # cv2.waitKey(0)
                                # cv2.destroyAllWindows()
                            # print("resized",timg_resized.shape)
                            # print("d", d)
                            # print("u", u)

                            if LOCAL_FEATURE == 'Crop_DISK':
                                features = disk(timg_resized, num_feats, pad_if_not_divisible=True)[0]
                                kps1, descs = features.keypoints, features.descriptors
                                lafs = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=device))
                            elif LOCAL_FEATURE == 'Crop_KeyNetAffNetHardNet':
                                lafs, resps, descs = feature(K.color.rgb_to_grayscale(timg_resized))

                      #如果是crop之后的则需要对keypoint的坐标加上l和d
                    if LOCAL_FEATURE == 'Crop_KeyNetAffNetHardNet' or LOCAL_FEATURE =='Crop_DISK':
                        lafs[:, :, 0, :] *= float(W) / float(w)
                        lafs[:, :, 1, :] *= float(H) / float(h)
                        desc_dim = descs.shape[-1]
                        kpts = KF.get_laf_center(lafs).reshape(-1, 2)
                        kpts[:, 0] = kpts[:, 0].add(l*float(W) / float(w))
                        kpts[:, 1] = kpts[:, 1].add(d*float(H) / float(h))
                        kpts = kpts.detach().cpu().numpy()
                        descs = descs.reshape(-1, desc_dim).detach().cpu().numpy()
                        f_laf[key] = lafs.detach().cpu().numpy()
                        f_kp[key] = kpts
                        f_desc[key] = descs

                    else:
                        lafs[:, :, 0, :] *= float(W) / float(w)
                        lafs[:, :, 1, :] *= float(H) / float(h)
                        desc_dim = descs.shape[-1]
                        kpts = KF.get_laf_center(lafs).reshape(-1, 2).detach().cpu().numpy()
                        descs = descs.reshape(-1, desc_dim).detach().cpu().numpy()
                        f_laf[key] = lafs.detach().cpu().numpy()
                        f_kp[key] = kpts
                        f_desc[key] = descs
    except:
        pass


    return


def get_unique_idxs(A, dim=0):
    # https://stackoverflow.com/questions/72001505/how-to-get-unique-elements-and-their-firstly-appeared-indices-of-a-pytorch-tenso
    unique, idx, counts = torch.unique(A, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]
    return first_indices


def match_features(img_fnames,
                   index_pairs,
                   feature_dir='.featureout',
                   device=torch.device('cpu'),
                   min_matches=0,
                   force_mutual=True,
                   matching_alg='smnn'
                   ):
    assert matching_alg in ['smnn', 'adalam']
    with h5py.File(f'{feature_dir}/lafs_{LOCAL_FEATURE}.h5', mode='r') as f_laf, \
            h5py.File(f'{feature_dir}/descriptors_{LOCAL_FEATURE}.h5', mode='r') as f_desc, \
            h5py.File(f'{feature_dir}/matches_{LOCAL_FEATURE}.h5', mode='w') as f_match:


        for pair_idx in progress_bar(index_pairs):
            try:
                idx1, idx2 = pair_idx
                fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
                key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
                lafs1 = torch.from_numpy(f_laf[key1][...]).to(device)
                lafs2 = torch.from_numpy(f_laf[key2][...]).to(device)
                desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
                desc2 = torch.from_numpy(f_desc[key2][...]).to(device)
                if matching_alg == 'adalam':
                    img1, img2 = cv2.imread(fname1), cv2.imread(fname2)
                    hw1, hw2 = img1.shape[:2], img2.shape[:2]
                    adalam_config = KF.adalam.get_adalam_default_config()
                    # adalam_config['orientation_difference_threshold'] = None
                    # adalam_config['scale_rate_threshold'] = None
                    adalam_config['force_seed_mnn'] = False
                    adalam_config['search_expansion'] = 16
                    adalam_config['ransac_iters'] = 128
                    adalam_config['device'] = device
                    dists, idxs = KF.match_adalam(desc1, desc2,
                                                  lafs1, lafs2,  # Adalam takes into account also geometric information
                                                  hw1=hw1, hw2=hw2,
                                                  config=adalam_config)  # Adalam also benefits from knowing image size
                else:
                    dists, idxs = KF.match_smnn(desc1, desc2, 0.92)
                if len(idxs) == 0:
                    continue
                # if len(idxs) >250:
                #     idxs = idxs[0:250]
                # Force mutual nearest neighbors
                if force_mutual:
                    first_indices = get_unique_idxs(idxs[:, 1])
                    idxs = idxs[first_indices]
                    dists = dists[first_indices]
                n_matches = len(idxs)
                if False:
                    print(f'{key1}-{key2}: {n_matches} matches')
                group = f_match.require_group(key1)
                if n_matches >= min_matches:
                    group.create_dataset(key2, data=idxs.detach().cpu().numpy().reshape(-1, 2))
            except:
                pass
    return

def match_superpoint(img_fnames,
                index_pairs,
                feature_dir='.featureout_loftr',
                device=torch.device('cpu'),
                name="None",
                min_matches=5, resize_to_=(800, 600)):
    resize = [-1, ]
    resize_float = True
    config = {
        "superpoint": {
            "nms_radius": 4,
            "keypoint_threshold": 0.005,
            "max_keypoints": 1024
        },
        "superglue": {
            "weights": "outdoor",
            "sinkhorn_iterations": 20,
            "match_threshold": 0.2,
        }
    }
    matcher = Matching(config).eval().to(device)
    with h5py.File(f'{feature_dir}/matches_superpoint.h5', mode='w') as f_match:
        for pair_idx in progress_bar(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]

            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
            timg1 = K.color.rgb_to_grayscale(load_torch_image(fname1, device=device))
            timg2 = K.color.rgb_to_grayscale(load_torch_image(fname2, device=device))
            f_cropname1 = './image1.jpg'
            f_cropname2 = './image2.jpg'
            # Load img1
            H1, W1 = timg1.shape[2:]
            if H1 < W1:
                resize_to = resize_to_[1], resize_to_[0]
            else:
                resize_to = resize_to_
            timg_resized1 = K.geometry.resize(timg1, resize_to, antialias=True)
            h1, w1 = timg_resized1.shape[2:]

            torch_resize = Resize(resize_to_)
            image_1 =torch_resize(load_torch_image(fname1, device=device))
            vutils.save_image(image_1, f_cropname1, normalize=True)


            # Load img2
            H2, W2 = timg2.shape[2:]
            if H2 < W2:
                resize_to2 = resize_to[1], resize_to[0]
            else:
                resize_to2 = resize_to_
            timg_resized2 = K.geometry.resize(timg2, resize_to2, antialias=True)
            h2, w2 = timg_resized2.shape[2:]

            image_2 = torch_resize(load_torch_image(fname2, device=device))
            vutils.save_image(image_2, f_cropname2, normalize=True)

            image_1, inp_1, scales_1 = read_image(f_cropname1, device, resize, 0, resize_float)
            image_2, inp_2, scales_2 = read_image(f_cropname2, device, resize, 0, resize_float)

            with torch.inference_mode():
                pred = matcher({"image0": inp_1, "image1": inp_2})
                pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
                kpts1, kpts2 = pred["keypoints0"], pred["keypoints1"]
                matches, conf = pred["matches0"], pred["matching_scores0"]
                valid = matches > -1
                mkpts1 = torch.from_numpy(kpts1[valid])
                mkpts2 = torch.from_numpy(kpts2[matches[valid]])
                mconf = torch.from_numpy(conf[valid])
            # a, idx1 = torch.sort(correspondences['confidence'], descending=True)  # descending为alse，升序，为True，降序
            idx1 = torch.gt(mconf, 0.6)
            idx = torch.nonzero(idx1).view(-1)
            if len(idx) < min_matches:
                a, idx1 = torch.sort(mconf, descending=True)  # descending为alse，升序，为True，降序
                idx = idx1[:min_matches-1]
            # if len(idx) > 200:
            #     a, idx1 = torch.sort(mconf, descending=True)  # descending为alse，升序，为True，降序
            #     idx = idx1[:200]
            # confidence = a[idx]
            keypoints0 = mkpts1[idx, :]
            keypoints1 = mkpts2[idx, :]
            mkpts1 = keypoints0
            mkpts2 = keypoints1
            # correspondences['confidence'] = confidence
            mkpts0 = mkpts1.cpu().numpy()
            mkpts1 = mkpts2.cpu().numpy()
            mkpts0[:, 0] = mkpts0[:, 0] * float(W1) / float(w1)
            mkpts0[:, 1] = mkpts0[:, 1] * float(H1) / float(h1)

            mkpts1[:, 0] = mkpts1[:, 0] * float(W2) / float(w2)
            mkpts1[:, 1] = mkpts1[:, 1] * float(H2) / float(h2)
            n_matches = len(mkpts1)

            group = f_match.require_group(key1)
            if n_matches >= min_matches:
                group.create_dataset(key2, data=np.concatenate([mkpts0, mkpts1], axis=1))
            print("one pair finished",pair_idx)

    kpts = defaultdict(list)
    match_indexes = defaultdict(dict)
    total_kpts = defaultdict(int)

    with h5py.File(f'{feature_dir}/matches_superpoint.h5', mode='r') as f_match:
        for k1 in f_match.keys():
            group = f_match[k1]
            for k2 in group.keys():
                matches = group[k2][...]
                total_kpts[k1]
                kpts[k1].append(matches[:, :2])
                kpts[k2].append(matches[:, 2:])
                current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
                current_match[:, 0] += total_kpts[k1]
                current_match[:, 1] += total_kpts[k2]
                total_kpts[k1] += len(matches)
                total_kpts[k2] += len(matches)
                match_indexes[k1][k2] = current_match

    for k in kpts.keys():
        kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
    unique_kpts = {}
    unique_match_idxs = {}
    out_match = defaultdict(dict)
    for k in kpts.keys():
        uniq_kps, uniq_reverse_idxs = torch.unique(torch.from_numpy(kpts[k]), dim=0, return_inverse=True)
        unique_match_idxs[k] = uniq_reverse_idxs
        unique_kpts[k] = uniq_kps.numpy()
    for k1, group in match_indexes.items():
        for k2, m in group.items():
            m2 = deepcopy(m)
            m2[:, 0] = unique_match_idxs[k1][m2[:, 0]]
            m2[:, 1] = unique_match_idxs[k2][m2[:, 1]]
            mkpts = np.concatenate([unique_kpts[k1][m2[:, 0]],
                                    unique_kpts[k2][m2[:, 1]],
                                    ],
                                   axis=1)
            unique_idxs_current = get_unique_idxs(torch.from_numpy(mkpts), dim=0)
            m2_semiclean = m2[unique_idxs_current]
            unique_idxs_current1 = get_unique_idxs(m2_semiclean[:, 0], dim=0)
            m2_semiclean = m2_semiclean[unique_idxs_current1]
            unique_idxs_current2 = get_unique_idxs(m2_semiclean[:, 1], dim=0)
            m2_semiclean2 = m2_semiclean[unique_idxs_current2]
            out_match[k1][k2] = m2_semiclean2.numpy()

    with h5py.File(f'{feature_dir}/keypoints_SUP.h5', mode='w') as f_kp:
        for k, kpts1 in unique_kpts.items():
            f_kp[k] = kpts1

    with h5py.File(f'{feature_dir}/matches_SUP.h5', mode='w') as f_match:
        for k1, gr in out_match.items():
            group = f_match.require_group(k1)
            for k2, match in gr.items():
                group[k2] = match
    return

def crop(file):
    f1 = file
    print("第二步：计算半径为")
    MinPts = 12
    eps = radius(f1, MinPts)
    # print("第三步：DBSCAN算法")
    # types, sub_class = dbscan(f1, eps, MinPts)
    # DASCAN划分后画图函数
    # draw_DASCAN(f1, sub_class)
    model = DBSCAN(eps=eps, min_samples=15)
    model.fit(f1)
    sub_class = model.fit_predict(f1)
    # plt.figure()
    # plt.scatter(f1[:, 0], f1[:, 1], c=sub_class)
    # plt.show()
    sub_class = np.array(sub_class)
    # draw_DASCAN(f1, sub_class)
    sub_class = sub_class.reshape(-1)

    # 获取最多分类点的值
    main_num = Counter(sub_class).most_common(1)[0][0]
    # 在此类的点的序号
    index = np.argwhere(sub_class == main_num)
    f1_crop = []
    for i in index:
        f1_crop_member = [f1[i][0][0], f1[i][0][1]]
        f1_crop.append(f1_crop_member)
    f1_crop = np.array(f1_crop)
    f1_crop = torch.tensor(f1_crop).view(len(index), 2)
    # 计算上下左右四个边界坐标
    r1 = int(torch.max(f1_crop[:, 0]).item())
    l1 = int(torch.min(f1_crop[:, 0]).item())
    u1 = int(torch.max(f1_crop[:, 1]).item())
    d1 = int(torch.min(f1_crop[:, 1]).item())
    return r1, l1, u1, d1
def crop_loftr(matcher, timg1, timg2, resize_to_, l1, d1, l2, d2):
    H1, W1 = timg1.shape[2:]
    if H1 < W1:
        resize_to = resize_to_[1], resize_to_[0]
    else:
        resize_to = resize_to_
    timg_resized1 = K.geometry.resize(timg1, resize_to, antialias=True)
    h1, w1 = timg_resized1.shape[2:]

    # Load img2
    H2, W2 = timg2.shape[2:]
    if H2 < W2:
        resize_to2 = resize_to[1], resize_to[0]
    else:
        resize_to2 = resize_to_
    timg_resized2 = K.geometry.resize(timg2, resize_to2, antialias=True)
    h2, w2 = timg_resized2.shape[2:]
    with torch.inference_mode():
        input_dict = {"image0": timg_resized1, "image1": timg_resized2}
        correspondences = matcher(input_dict)
    # a, idx1 = torch.sort(correspondences['confidence'], descending=True)  # descending为alse，升序，为True，降序
    idx1 = torch.gt(correspondences['confidence'], 0.6)
    idx = torch.nonzero(idx1).view(-1)
    if len(idx) < 5:
        a, idx1 = torch.sort(correspondences['confidence'], descending=True)  # descending为alse，升序，为True，降序
        idx = idx1[:4]
    # if len(idx) > 200:
    #     a, idx1 = torch.sort(correspondences['confidence'], descending=True)  # descending为alse，升序，为True，降序
    #     idx = idx1[:200]
    # confidence = a[idx]
    keypoints0 = correspondences['keypoints0'][idx, :]
    keypoints1 = correspondences['keypoints1'][idx, :]
    correspondences["keypoints0"] = keypoints0
    correspondences["keypoints1"] = keypoints1
    # correspondences['confidence'] = confidence
    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    mkpts0[:, 0] = mkpts0[:, 0] * float(W1) / float(w1) + l1
    mkpts0[:, 1] = mkpts0[:, 1] * float(H1) / float(h1) + d1

    mkpts1[:, 0] = mkpts1[:, 0] * float(W2) / float(w2) + l2
    mkpts1[:, 1] = mkpts1[:, 1] * float(H2) / float(h2) + d2
    return mkpts0, mkpts1

def crop_superpoint(matcher,timg1,timg2,fname1,fname2,resize_to_,l1,d1,l2,d2):
    resize = [-1, ]
    resize_float = True


    f_cropname1 = './image1_crop.jpg'
    f_cropname2 = './image2_crop.jpg'
    # Load img1
    torch_resize = Resize(resize_to_)
    image_1 = torch_resize(load_torch_image(fname1, device=device))
    vutils.save_image(image_1, f_cropname1, normalize=True)
    image_2 = torch_resize(load_torch_image(fname2, device=device))
    vutils.save_image(image_2, f_cropname2, normalize=True)

    H1, W1 = timg1.shape[2:]
    # if H1 < W1:
    #     resize_to = resize_to_[1], resize_to_[0]
    # else:
    resize_to = resize_to_
    timg_resized1 = K.geometry.resize(timg1, resize_to, antialias=True)
    h1, w1 = timg_resized1.shape[2:]




    # Load img2
    H2, W2 = timg2.shape[2:]
    # if H2 < W2:
    #     resize_to2 = resize_to_[1], resize_to_[0]
    # else:
    resize_to2 = resize_to_
    timg_resized2 = K.geometry.resize(timg2, resize_to2, antialias=True)
    h2, w2 = timg_resized2.shape[2:]



    image_1, inp_1, scales_1 = read_image(f_cropname1, device, resize, 0, resize_float)
    image_2, inp_2, scales_2 = read_image(f_cropname2, device, resize, 0, resize_float)
    with torch.inference_mode():
        pred = matcher({"image0": inp_1, "image1": inp_2})
        pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        kpts1, kpts2 = pred["keypoints0"], pred["keypoints1"]
        matches, conf = pred["matches0"], pred["matching_scores0"]
        valid = matches > -1
        mkpts1 = torch.from_numpy(kpts1[valid])
        mkpts2 = torch.from_numpy(kpts2[matches[valid]])
        mconf = torch.from_numpy(conf[valid])
    # a, idx1 = torch.sort(correspondences['confidence'], descending=True)  # descending为alse，升序，为True，降序
    idx1 = torch.gt(mconf, 0.6)
    idx = torch.nonzero(idx1).view(-1)
    if len(idx) < 5:
        a, idx1 = torch.sort(mconf, descending=True)  # descending为alse，升序，为True，降序
        idx = idx1[:4]
    # if len(idx) > 50:
    #     a, idx1 = torch.sort(mconf, descending=True)  # descending为alse，升序，为True，降序
    #     idx = idx1[:50]
    # confidence = a[idx]
    keypoints0 = mkpts1[idx, :]
    keypoints1 = mkpts2[idx, :]
    mkpts1 = keypoints0
    mkpts2 = keypoints1
    # correspondences['confidence'] = confidence
    m_kpts0 = mkpts1.cpu().numpy()
    m_kpts1 = mkpts2.cpu().numpy()
    m_kpts0[:, 0] = m_kpts0[:, 0] * float(W1) / float(w1) + l1
    m_kpts0[:, 1] = m_kpts0[:, 1] * float(H1) / float(h1) + d1

    m_kpts1[:, 0] = m_kpts1[:, 0] * float(W2) / float(w2) + l2
    m_kpts1[:, 1] = m_kpts1[:, 1] * float(H2) / float(h2) + d2
    return m_kpts0, m_kpts1


def match_crop(img_fnames,
                index_pairs,
                feature_dir='.featureout_loftr',
                device=torch.device('cpu'),
                name="None",
                min_matches=5, resize_to_=(600, 800)):
    resize = [-1, ]
    resize_float = True
    config = {
        "superpoint": {
            "nms_radius": 4,
            "keypoint_threshold": 0.005,
            "max_keypoints": 1024
        },
        "superglue": {
            "weights": "outdoor",
            "sinkhorn_iterations": 20,
            "match_threshold": 0.2,
        }
    }
    matcher_superpoint = Matching(config).eval().to(device)
    matcher_loftr = KF.LoFTR(pretrained=None)
    matcher_loftr.load_state_dict(torch.load('../params/loftr_outdoor.ckpt')['state_dict'])
    matcher_loftr = matcher_loftr.to(device).eval()
    match_ld =[]
    if not os.path.exists("./crop"):
        os.makedirs("./crop")
    with h5py.File(f'{feature_dir}/' + name, mode='r+') as f_kp_croploftr:
        for index,fname1 in enumerate(img_fnames):
            img_1 = load_torch_image(fname1, device=device)
            key1 = fname1.split('/')[-1]
            f1 = f_kp_croploftr[key1][:]
            f_cropname1 = "./crop/image_crop_%d.jpg"%index
            r1, l1, u1, d1 = crop(f1)
            img_1 = img_1[:, :, d1:u1]
            img_1 = img_1[:, :, :, l1:r1]
            vutils.save_image(img_1, f_cropname1, normalize=True)
            match_ld.append([l1, d1])

    with h5py.File(f'{feature_dir}/matches_Crop_temp.h5', mode='w') as f_match:
        for pair_idx in progress_bar(index_pairs):
            print(pair_idx)
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
            fname1, fname2 = "./crop/image_crop_%d.jpg"%idx1, "./crop/image_crop_%d.jpg"%idx2
            [l1, d1] = match_ld[idx1]
            [l2, d2] = match_ld[idx2]
            timg1 = K.color.rgb_to_grayscale(load_torch_image(fname1, device=device))
            timg2 = K.color.rgb_to_grayscale(load_torch_image(fname2, device=device))

            # resize_to_ = (960, 1280)
            # resize_to_ = (600, 800)
            # mkpts0_loftr_1280, mkpts1_loftr_1280 = crop_loftr(matcher_loftr, timg1, timg2, resize_to_, l1, d1, l2, d2)
            # resize_to_ = (630, 840)
            # mkpts0_loftr_840, mkpts1_loftr_840 = crop_loftr(matcher_loftr, timg1, timg2, resize_to_, l1, d1, l2, d2)
            # resize_to_ = (960, 1280)
            # mkpts0_sup_1280, mkpts1_sup_1280 = crop_superpoint(matcher_superpoint, timg1, timg2, fname1,
            #                                                    fname2, resize_to_, l1, d1, l2, d2)
            resize_to_ = (768, 1024)
            mkpts0_sup_1024, mkpts1_sup_1024 = crop_superpoint(matcher_superpoint, timg1, timg2, fname1,
                                                               fname2, resize_to_, l1, d1, l2, d2)
            resize_to_ = (1152, 1536)
            mkpts0_sup_1536, mkpts1_sup_1536 = crop_superpoint(matcher_superpoint, timg1, timg2, fname1,
                                                               fname2, resize_to_, l1, d1, l2, d2)

            mkpts0 = np.concatenate(
                (mkpts0_sup_1024, mkpts0_sup_1536), axis=0)
            mkpts1 = np.concatenate(
                (mkpts1_sup_1024, mkpts1_sup_1536), axis=0)
            # mkpts0 = np.concatenate((mkpts0_loftr_1280, mkpts0_loftr_840, mkpts0_sup_1280,mkpts0_sup_1024,mkpts0_sup_1536), axis=0)
            # mkpts1 = np.concatenate((mkpts1_loftr_1280, mkpts1_loftr_840, mkpts1_sup_1280,mkpts1_sup_1024,mkpts1_sup_1536), axis=0)
            # mkpts0 = mkpts0_sup_1024
            # mkpts1 = mkpts1_sup_1024
            n_matches = len(mkpts1)
            group = f_match.require_group(key1)
            if n_matches >= min_matches:
                group.create_dataset(key2, data=np.concatenate([mkpts0, mkpts1], axis=1))
            print("one pair finished", pair_idx)

    kpts = defaultdict(list)
    match_indexes = defaultdict(dict)
    total_kpts = defaultdict(int)

    with h5py.File(f'{feature_dir}/matches_Crop_temp.h5', mode='r') as f_match:
        for k1 in f_match.keys():
            group = f_match[k1]
            for k2 in group.keys():
                matches = group[k2][...]
                total_kpts[k1]
                kpts[k1].append(matches[:, :2])
                kpts[k2].append(matches[:, 2:])
                current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
                current_match[:, 0] += total_kpts[k1]
                current_match[:, 1] += total_kpts[k2]
                total_kpts[k1] += len(matches)
                total_kpts[k2] += len(matches)
                match_indexes[k1][k2] = current_match

    for k in kpts.keys():
        kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
    unique_kpts = {}
    unique_match_idxs = {}
    out_match = defaultdict(dict)
    for k in kpts.keys():
        uniq_kps, uniq_reverse_idxs = torch.unique(torch.from_numpy(kpts[k]), dim=0, return_inverse=True)
        unique_match_idxs[k] = uniq_reverse_idxs
        unique_kpts[k] = uniq_kps.numpy()
    for k1, group in match_indexes.items():
        for k2, m in group.items():
            m2 = deepcopy(m)
            m2[:, 0] = unique_match_idxs[k1][m2[:, 0]]
            m2[:, 1] = unique_match_idxs[k2][m2[:, 1]]
            mkpts = np.concatenate([unique_kpts[k1][m2[:, 0]],
                                    unique_kpts[k2][m2[:, 1]],
                                    ],
                                   axis=1)
            unique_idxs_current = get_unique_idxs(torch.from_numpy(mkpts), dim=0)
            m2_semiclean = m2[unique_idxs_current]
            unique_idxs_current1 = get_unique_idxs(m2_semiclean[:, 0], dim=0)
            m2_semiclean = m2_semiclean[unique_idxs_current1]
            unique_idxs_current2 = get_unique_idxs(m2_semiclean[:, 1], dim=0)
            m2_semiclean2 = m2_semiclean[unique_idxs_current2]
            out_match[k1][k2] = m2_semiclean2.numpy()


    with h5py.File(f'{feature_dir}/keypoints_Crop.h5', mode='w') as f_kp:
        for k, kpts1 in unique_kpts.items():
            f_kp[k] = kpts1

    with h5py.File(f'{feature_dir}/matches_Crop.h5', mode='w') as f_match:
        for k1, gr in out_match.items():
            group = f_match.require_group(k1)
            for k2, match in gr.items():
                group[k2] = match






def match_loftr(img_fnames,
                index_pairs,
                feature_dir='.featureout_loftr',
                device=torch.device('cpu'),
                name="None",
                min_matches=5, resize_to_=(800, 600)):
    # matcher = KF.LoFTR(pretrained="outdoor")
    # matcher = KF.LoFTR(pretrained=None)
    # matcher.load_state_dict(torch.load('/kaggle/input/loftr/pytorch/outdoor/1/loftr_outdoor.ckpt')['state_dict'])
    matcher = KF.LoFTR(pretrained=None)
    matcher.load_state_dict(torch.load('../params/loftr_outdoor.ckpt')['state_dict'])
    matcher = matcher.to(device).eval()
    # First we do pairwise matching, and then extract "keypoints" from loftr matches.
    with h5py.File(f'{feature_dir}/matches_loftr.h5', mode='w') as f_match:
        numm =0
        for pair_idx in progress_bar(index_pairs):
            try:

                numm =numm+1
                idx1, idx2 = pair_idx
                fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
                key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
                # Load img1
                img_a1 = load_torch_image(fname1, device=device)
                timg1 = K.color.rgb_to_grayscale(load_torch_image(fname1, device=device))
                timg2 = K.color.rgb_to_grayscale(load_torch_image(fname2, device=device))
                if LOCAL_FEATURE == 'Crop_LoFTR':
                    with h5py.File(f'{feature_dir}/' + name, mode='r+') as f_kp_croploftr:
                        f1 = f_kp_croploftr[key1][:]
                        print("第二步：计算半径为")
                        MinPts = 12
                        eps = radius(f1, MinPts)
                        # print("第三步：DBSCAN算法")
                        # types, sub_class = dbscan(f1, eps, MinPts)
                        # DASCAN划分后画图函数
                        # draw_DASCAN(f1, sub_class)
                        model = DBSCAN(eps=eps, min_samples=10)
                        model.fit(f1)
                        sub_class = model.fit_predict(f1)
                        # plt.figure()
                        # plt.scatter(f1[:, 0], f1[:, 1], c=sub_class)
                        # plt.show()
                        sub_class = np.array(sub_class)
                        sub_class = sub_class.reshape(-1)
                        # 获取最多分类点的值
                        main_num = Counter(sub_class).most_common(1)[0][0]
                        # 在此类的点的序号
                        index = np.argwhere(sub_class == main_num)
                        f1_crop = []
                        for i in index:
                            f1_crop_member = [f1[i][0][0] , f1[i][0][1]]
                            f1_crop.append(f1_crop_member)
                        f1_crop = np.array(f1_crop)
                        f1_crop = torch.tensor(f1_crop).view(len(index), 2)
                        # 计算上下左右四个边界坐标
                        r1 = int(torch.max(f1_crop[:, 0]).item())
                        l1 = int(torch.min(f1_crop[:, 0]).item())
                        u1 = int(torch.max(f1_crop[:, 1]).item())
                        d1 = int(torch.min(f1_crop[:, 1]).item())

                        # print("unresized", timg_resized.shape)
                        if ((u1 - d1) > 100) or ((r1 - l1) > 100):
                            # 重新裁剪图像
                            timg1 = timg1[:, :, d1:u1]
                            timg1 = timg1[:, :, :, l1:r1]


                        f2 = f_kp_croploftr[key2][:]
                        print("第二步：计算半径为")
                        MinPts = 12
                        eps = radius(f2, MinPts)
                        # print("第三步：DBSCAN算法")
                        # types, sub_class = dbscan(f1, eps, MinPts)
                        # DASCAN划分后画图函数
                        # draw_DASCAN(f1, sub_class)
                        model = DBSCAN(eps=eps, min_samples=10)
                        model.fit(f2)
                        sub_class = model.fit_predict(f2)
                        # plt.figure()
                        # plt.scatter(f1[:, 0], f1[:, 1], c=sub_class)
                        # plt.show()
                        sub_class = np.array(sub_class)
                        sub_class = sub_class.reshape(-1)
                        # 获取最多分类点的值
                        main_num = Counter(sub_class).most_common(1)[0][0]
                        # 在此类的点的序号
                        index = np.argwhere(sub_class == main_num)
                        f2_crop = []
                        for i in index:
                            f2_crop_member = [f2[i][0][0], f2[i][0][1]]
                            f2_crop.append(f2_crop_member)
                        f2_crop = np.array(f2_crop)
                        f2_crop = torch.tensor(f2_crop).view(len(index), 2)
                        # 计算上下左右四个边界坐标
                        r2 = int(torch.max(f2_crop[:, 0]).item())
                        l2 = int(torch.min(f2_crop[:, 0]).item())
                        u2 = int(torch.max(f2_crop[:, 1]).item())
                        d2 = int(torch.min(f2_crop[:, 1]).item())

                        # print("unresized", timg_resized.shape)
                        if ((u2 - d2) > 100) or ((r2 - l2) > 100):
                            # 重新裁剪图像
                            timg2 = timg2[:, :, d2:u2]
                            timg2 = timg2[:, :, :, l2:r2]


                H1, W1 = timg1.shape[2:]
                if H1 < W1:
                    resize_to = resize_to_[1], resize_to_[0]
                else:
                    resize_to = resize_to_
                timg_resized1 = K.geometry.resize(timg1, resize_to, antialias=True)
                h1, w1 = timg_resized1.shape[2:]

                # Load img2
                H2, W2 = timg2.shape[2:]
                if H2 < W2:
                    resize_to2 = resize_to[1], resize_to[0]
                else:
                    resize_to2 = resize_to_
                timg_resized2 = K.geometry.resize(timg2, resize_to2, antialias=True)
                h2, w2 = timg_resized2.shape[2:]
                with torch.inference_mode():
                    input_dict = {"image0": timg_resized1, "image1": timg_resized2}
                    correspondences = matcher(input_dict)
                #a, idx1 = torch.sort(correspondences['confidence'], descending=True)  # descending为alse，升序，为True，降序
                idx1 = torch.gt(correspondences['confidence'], 0.6)
                idx = torch.nonzero(idx1).view(-1)
                if len(idx)< min_matches:
                    a, idx1 = torch.sort(correspondences['confidence'], descending=True)  # descending为alse，升序，为True，降序
                    idx = idx1[:min_matches-1]
                # if len(idx) > 200:
                #     a, idx1 = torch.sort(correspondences['confidence'], descending=True)  # descending为alse，升序，为True，降序
                #     idx = idx1[:200]
                #confidence = a[idx]
                keypoints0 = correspondences['keypoints0'][idx, :]
                keypoints1 = correspondences['keypoints1'][idx, :]
                correspondences["keypoints0"] = keypoints0
                correspondences["keypoints1"] = keypoints1
                #correspondences['confidence'] = confidence
                mkpts0 = correspondences["keypoints0"].cpu().numpy()
                mkpts1 = correspondences["keypoints1"].cpu().numpy()

                if LOCAL_FEATURE == 'Crop_LoFTR':

                    mkpts0[:, 0] = mkpts0[:, 0]*float(W1) / float(w1)+l1
                    mkpts0[:, 1] = mkpts0[:, 1]*float(H1) / float(h1)+d1

                    mkpts1[:, 0] = mkpts1[:, 0]*float(W2) / float(w2)+l2
                    mkpts1[:, 1] = mkpts1[:, 1]*float(H2) / float(h2)+d2
                else:
                    mkpts0[:, 0] = mkpts0[:, 0] * float(W1) / float(w1)
                    mkpts0[:, 1] = mkpts0[:, 1] * float(H1) / float(h1)

                    mkpts1[:, 0] = mkpts1[:, 0] * float(W2) / float(w2)
                    mkpts1[:, 1] = mkpts1[:, 1]* float(H2) / float(h2)


                n_matches = len(mkpts1)

                group = f_match.require_group(key1)
                if n_matches >= min_matches:
                    group.create_dataset(key2, data=np.concatenate([mkpts0, mkpts1], axis=1))
                print("one pair finished", pair_idx)
            except:
                pass



    # Let's find unique loftr pixels and group them together.
    kpts = defaultdict(list)
    match_indexes = defaultdict(dict)
    total_kpts = defaultdict(int)

    with h5py.File(f'{feature_dir}/matches_loftr.h5', mode='r') as f_match:
        for k1 in f_match.keys():
            group = f_match[k1]
            for k2 in group.keys():
                matches = group[k2][...]
                total_kpts[k1]
                kpts[k1].append(matches[:, :2])
                kpts[k2].append(matches[:, 2:])
                current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
                current_match[:, 0] += total_kpts[k1]
                current_match[:, 1] += total_kpts[k2]
                total_kpts[k1] += len(matches)
                total_kpts[k2] += len(matches)
                match_indexes[k1][k2] = current_match


    for k in kpts.keys():
        kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
    unique_kpts = {}
    unique_match_idxs = {}
    out_match = defaultdict(dict)
    for k in kpts.keys():
        uniq_kps, uniq_reverse_idxs = torch.unique(torch.from_numpy(kpts[k]), dim=0, return_inverse=True)
        unique_match_idxs[k] = uniq_reverse_idxs
        unique_kpts[k] = uniq_kps.numpy()
    for k1, group in match_indexes.items():
        for k2, m in group.items():
            m2 = deepcopy(m)
            m2[:, 0] = unique_match_idxs[k1][m2[:, 0]]
            m2[:, 1] = unique_match_idxs[k2][m2[:, 1]]
            mkpts = np.concatenate([unique_kpts[k1][m2[:, 0]],
                                    unique_kpts[k2][m2[:, 1]],
                                    ],
                                   axis=1)
            unique_idxs_current = get_unique_idxs(torch.from_numpy(mkpts), dim=0)
            m2_semiclean = m2[unique_idxs_current]
            unique_idxs_current1 = get_unique_idxs(m2_semiclean[:, 0], dim=0)
            m2_semiclean = m2_semiclean[unique_idxs_current1]
            unique_idxs_current2 = get_unique_idxs(m2_semiclean[:, 1], dim=0)
            m2_semiclean2 = m2_semiclean[unique_idxs_current2]
            out_match[k1][k2] = m2_semiclean2.numpy()
    if LOCAL_FEATURE == 'Crop_LoFTR':

        with h5py.File(f'{feature_dir}/keypoints_CropLOF.h5', mode='w') as f_kp:
            for k, kpts1 in unique_kpts.items():
                f_kp[k] = kpts1

        with h5py.File(f'{feature_dir}/matches_CropLOF.h5', mode='w') as f_match:
            for k1, gr in out_match.items():
                group = f_match.require_group(k1)
                for k2, match in gr.items():
                    group[k2] = match
    else:
        with h5py.File(f'{feature_dir}/keypoints_LOF.h5', mode='w') as f_kp:
            for k, kpts1 in unique_kpts.items():
                f_kp[k] = kpts1

        with h5py.File(f'{feature_dir}/matches_LOF.h5', mode='w') as f_match:
            for k1, gr in out_match.items():
                group = f_match.require_group(k1)
                for k2, match in gr.items():
                    group[k2] = match
    return


def import_into_colmap(img_dir,
                       feature_dir='.featureout',
                       database_path='colmap.db',
                       img_ext='.jpg'):
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, img_dir, img_ext, 'simple-radial', single_camera)
    add_matches(
        db,
        feature_dir,
        fname_to_id,
    )

    db.commit()
    return


src = '../dataset'
# Get data from csv.

data_dict = {}


data_dict = {}
data_train ='train_mini'
train_path = os.path.join(src, data_train)
dataset_list = [dI for dI in os.listdir(train_path) if os.path.isdir(os.path.join(train_path,dI))]


for dataset in dataset_list:
    data_dict[dataset] = {}
    dataset_path = os.path.join(train_path, dataset)
    scene_list = [dI for dI in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path,dI))]
    for scene in scene_list:
        data_dict[dataset][scene] = [];
        scene_path = os.path.join(dataset_path, scene, 'images')
        image_list = os.listdir(scene_path)
        for images in image_list:
            image = os.path.join(dataset, scene ,'images', images)
            data_dict[dataset][scene].append(image)
                          
        
        
                          

                          
for dataset in data_dict:
    for scene in data_dict[dataset]:
        print(f' -> {len(data_dict[dataset][scene])} images')
out_results = {}
timings = {"shortlisting": [],
           "feature_detection": [],
           "feature_matching": [],
           "RANSAC": [],
           "Reconstruction": []}


# Function to create a submission file.
def create_submission(out_results, data_dict):
    with open(f'submission.csv', 'w') as f:
        f.write('image_path,dataset,scene,rotation_matrix,translation_vector\n')
        for dataset in data_dict:
            if dataset in out_results:
                res = out_results[dataset]
            else:
                res = {}
            for scene in data_dict[dataset]:
                if scene in res:
                    scene_res = res[scene]
                else:
                    scene_res = {"R": {}, "t": {}}
                for image in data_dict[dataset][scene]:
                    if image in scene_res:
                        R = scene_res[image]['R'].reshape(-1)
                        T = scene_res[image]['t'].reshape(-1)
                    else:
                        R = np.eye(3).reshape(-1)
                        T = np.zeros((3))
                    f.write(f'{image},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n')

def distance(data):
    m, n = np.shape(data)
    Mydist = np.mat(np.zeros((m, m)))
    for a in range(m):
        for z in range(a, m):
            # 计算a和z之间的欧式距离
            tmp = 0
            for k in range(n):
                tmp += (data[a, k] - data[z, k]) * (data[a, k] - data[z, k])
            Mydist[a, z] = np.sqrt(tmp)
            Mydist[z, a] = Mydist[a, z]
    return Mydist

def find_in_radius(distance, Myradius):
    X = []
    n = np.shape(distance)[1]
    for j in range(n):
        if distance[0, j] <= Myradius:
            X.append(j)
    return X


def radius(data, MinPts):
    m, n = np.shape(data)#m:样本个数；n:特征的维度
    Mymax_data = np.max(data, 0)#找到最大与最小的样本
    Mymin_data = np.min(data, 0)
    r = (((np.prod(Mymax_data - Mymin_data) * MinPts * math.gamma(0.5 * n + 1)) / (m * math.sqrt(math.pi ** n))) ** (1.0 / n))
    return r

def draw(fname1,fname2,name1,name2):
    img0_orig = cv2.cvtColor(cv2.imread(fname1), cv2.COLOR_BGR2RGB)
    img1_orig = cv2.cvtColor(cv2.imread(fname2), cv2.COLOR_BGR2RGB)
    key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
    exist = 0

    #print(key1)
    with h5py.File(f'{feature_dir}/'+name1, mode='r') as f_kp, \
         h5py.File(f'{feature_dir}/'+name2, mode='r') as f_match:
        try:
            #print('error1')
            kps0 = f_kp[key1][:]
            #print('error2')
            #print(kps0,type(kps0))
            kps1 = f_kp[key2][:]
            #print('---------')

            group = f_match[key1]

            idxs = group[key2][:]
            idxs1 = idxs[:, 0]
            idxs2 = idxs[:, 1]
            kps0 = kps0[idxs1, :]
            kps1 = kps1[idxs2, :]
        except:
            exist =1
        #print('idxs',idxs)
        #print(type(idxs))


    if exist == 0:
        # Pad the images if they are of different size, for visualization.
        h0, w0 = img0_orig.shape[:2]
        h1 = img1_orig.shape[0]
        if h0 > h1:
            img1_orig = np.pad(img1_orig, pad_width=((0, h0 - h1), (0, 0), (0, 0)))
        else:
            img0_orig = np.pad(img0_orig, pad_width=((0, h1 - h0), (0, 0), (0, 0)))

        img_matches = np.concatenate((img0_orig, img1_orig), axis=1)

        for kp0, kp1 in zip(kps0, kps1):
            img_matches = cv2.line(
                img_matches, tuple(map(int, (kp0[0], kp0[1]))), tuple(map(int, (w0 + kp1[0], kp1[1]))), color=(0, 0, 255),
                thickness=2
            )

        title = LOCAL_FEATURE+key1.split('\\')[-1]+' VS '+key2.split('\\')[-1]
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, img_matches)
        cv2.imwrite('./image_match/' +title, img_matches)
        #cv2.imwrite(r'C:\Users\dedll\Desktop\IMC\\together\\'+title,img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def draw_DASCAN(point_data, subclass):
    Myfig = plt.figure()
    axes = Myfig.add_subplot(111)
    length = len(point_data)
    for j in range(length):
        if subclass[j] == 1:
            axes.scatter(point_data[j, 0], point_data[j, 1], color='c', alpha=0.4)
        if subclass[j] == 2:
            axes.scatter(point_data[j, 0], point_data[j, 1], color='g', alpha=0.4)
        if subclass[j] == 3:
            axes.scatter(point_data[j, 0], point_data[j, 1], color='m', alpha=0.4)
        if subclass[j] == 4:
            axes.scatter(point_data[j, 0], point_data[j, 1], color='y', alpha=0.4)
        if subclass[j] == 5:
            axes.scatter(point_data[j, 0], point_data[j, 1], color='r', alpha=0.4)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('dbscan-JYZ')
    plt.show()

def concat_matches(name1, name2, name3,length):
    with h5py.File(f'{feature_dir}/'+name2, mode='r+') as f_matches_DISK, \
            h5py.File(f'{feature_dir}/'+name1, mode='r+') as f_matches_AffNet, \
            h5py.File(f'{feature_dir}/'+name3, mode='w') as f_matches:

        for key1 in f_matches_DISK.keys():
            group = f_matches.require_group(key1)
            for key2 in f_matches_DISK[key1]:
                pair_list = []
                for i, pair in enumerate(f_matches_DISK[key1][key2][:]):
                    pair_new = []
                    pair_new.append(pair[0] + length[key1])
                    pair_new.append(pair[1] + length[key2])
                    pair_list.append(list(pair_new))
                pair_list = torch.tensor(pair_list).reshape(-1, 2).detach().cpu().numpy()
                try:
                    try:
                        data = np.concatenate([f_matches_AffNet[key1][key2][:], np.array(pair_list)], axis=0)
                    except:
                        data = np.array(pair_list)
                    group[key2] = data
                    print(len(data))
                except:
                    continue
        for key1 in f_matches_AffNet.keys():
            group = f_matches.require_group(key1)
            for key2 in f_matches_AffNet[key1]:
                try:
                    exist = f_matches_DISK[key1][key2][:]
                except:
                    data = f_matches_AffNet[key1][key2][:]
                    group[key2] = data

def filter_OA(img_fnames,
              index_pairs,
              feature_dir='.featureout',
              device=torch.device('cpu')):
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='r+') as f_kp, \
            h5py.File(f'{feature_dir}/matches_bef.h5', mode='r+') as f_match, \
            h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_filter:
        for pair_idx in progress_bar(index_pairs):

            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
            print('filtername', key1, key2)

            kps0 = f_kp[key1][:]
            kps1 = f_kp[key2][:]
            group = f_match[key1]
            idxs = group[key2][:]
            idxs1 = idxs[:, 0]
            idxs2 = idxs[:, 1]
            kps0 = kps0[idxs1, :]
            kps1 = kps1[idxs2, :]
            print('滤前', f_match[key1][key2][:].shape)
            corr = np.concatenate((kps0, kps1), axis=1)
            corr_idx = idxs
            corr = torch.from_numpy(corr).to(device)
            corr_idx = torch.from_numpy(corr_idx).to(device)

            model_path = os.path.join('../OANetmaster/model/sift-gl3d/gl3d/sift-4000/model_best.pth')
            # img1_name, img2_name = 'test_img1.jpg', 'test_img2.jpg'
            lm = LearnedMatcher(model_path, inlier_threshold=-15000, use_ratio=0, use_mutual=0)
            filtermatches = lm.infer2(corr, corr_idx)

            group = f_filter.require_group(key1)
            group.create_dataset(key2, data=filtermatches)
            print('滤后',f_filter[key1][key2][:].shape)
            #draw(fname1, fname2, 1)
            #draw(fname1, fname2, 2)



def concat_keypoints(name1, name2, name3):
    length = {'Name': name1}
    with h5py.File(f'{feature_dir}/'+name2, mode='r') as f_kp_DISK, \
            h5py.File(f'{feature_dir}/'+name1, mode='r') as f_kp_AffNet, \
            h5py.File(f'{feature_dir}/'+name3, mode='w') as f_kp:
        for key in f_kp_DISK.keys():
            try:
                f1 = torch.from_numpy(f_kp_DISK[key][:])
                f2 = torch.from_numpy(f_kp_AffNet[key][:])
                length[key] = len(f2) #
                f_kp[key] = torch.cat((f2, f1), 0).reshape(-1, 2).detach().cpu().numpy()
            except:
                f_kp[key] = f_kp_DISK[key][:]
                length[key] = 0
        for key in f_kp_AffNet.keys():
            try:
                f1 = torch.from_numpy(f_kp_DISK[key][:])
                f2 = torch.from_numpy(f_kp_AffNet[key][:])
                length[key] = len(f2)
            except:
                f_kp[key] = f_kp_AffNet[key][:]
                length[key] = len(f_kp_AffNet[key][:])

        print("len", length)

    return length


gc.collect()
datasets = []
for dataset in data_dict:
    datasets.append(dataset)
for dataset in datasets:
    if dataset not in out_results:
        out_results[dataset] = {}
    for scene in data_dict[dataset]:
        img_dir = f'{src}/train_mini/{dataset}/{scene}/images'

        if not os.path.exists(img_dir):
            continue
        # Wrap the meaty part in a try-except block.

        out_results[dataset][scene] = {}
        img_fnames = [f'{src}/train_mini/{x}' for x in data_dict[dataset][scene]]
        print(f"Got {len(img_fnames)} images")
        feature_dir = f'featureout/{dataset}_{scene}'

        if not os.path.isdir(feature_dir):
            os.makedirs(feature_dir, exist_ok=True)
        t = time()

        #get image pairs
        index_pairs = get_image_pairs_shortlist(img_fnames,
                                                sim_th= 0.6,  # should be strict
                                                min_pairs=10,
                                                # we select at least min_pairs PER IMAGE with biggest similarity
                                                exhaustive_if_less=10,
                                                device=device)
        print(len(index_pairs))
        t = time() - t
        timings['shortlisting'].append(t)
        print(f'{len(index_pairs)}, pairs to match, {t:.4f} sec')
        gc.collect()
        t = time()

        #loftr,match features
        LOCAL_FEATURE = 'DISK'
        if LOCAL_FEATURE == 'DISK':
            detect_features(img_fnames,
                            2048,
                            feature_dir=feature_dir,
                            upright=True,
                            device=device,
                            resize_small_edge_to=600
                            )
            gc.collect()
            t = time() - t
            timings['feature_detection'].append(t)
            print(f'Features detected in  {t:.4f} sec')
            t = time()
            match_features(img_fnames, index_pairs, feature_dir=feature_dir, device=device)
            # for Z in index_pairs:
            #     print(img_fnames[Z[0]])
            #     draw(img_fnames[Z[0]], img_fnames[Z[1]],"keypoints_DISK.h5","matches_DISK.h5")

        #
        #
        # LOCAL_FEATURE = 'KeyNetAffNetHardNet'
        # if LOCAL_FEATURE == 'KeyNetAffNetHardNet':
        #     detect_features(img_fnames,
        #                     1024,
        #                     feature_dir=feature_dir,
        #                     upright=True,
        #                     device=device,
        #                     resize_small_edge_to=600
        #                     )
        #     gc.collect()
        #     t = time() - t
        #     timings['feature_detection'].append(t)
        #     print(f'Features detected in  {t:.4f} sec')
        #     t = time()
        #     match_features(img_fnames, index_pairs, feature_dir=feature_dir, device=device)
        # #拼接顺序先affnet再disk
        #     klen1 =concat_keypoints("keypoints_KeyNetAffNetHardNet.h5", "keypoints_DISK.h5", "keypoints_temp01.h5")
        #     concat_matches("matches_KeyNetAffNetHardNet.h5", "matches_DISK.h5", "matches_temp01.h5", klen1)

            # for Z in index_pairs:
            #     print(img_fnames[Z[0]])
            #     draw(img_fnames[Z[0]], img_fnames[Z[1]],"keypoints_temp01.h5","matches_temp01.h5")
        # LOCAL_FEATURE = 'superpoint'
        # if LOCAL_FEATURE == 'superpoint':
        #     match_superpoint(img_fnames, index_pairs, feature_dir=feature_dir, device=device, resize_to_=(600, 800))
        #     klen2 = concat_keypoints("keypoints_DISK.h5", "keypoints_SUP.h5", "keypoints.h5")
        #     concat_matches("matches_DISK.h5", "matches_SUP.h5", "matches_bef.h5", klen2)
        #     gc.collect()
            # for Z in index_pairs:
            #     print(img_fnames[Z[0]])
            #     draw(img_fnames[Z[0]], img_fnames[Z[1]],"keypoints_SUP.h5","matches_SUP.h5")


        LOCAL_FEATURE = 'LoFTR'
        if LOCAL_FEATURE == 'LoFTR':
            match_loftr(img_fnames, index_pairs, feature_dir=feature_dir, device=device, resize_to_=(800, 600))
            klen2 = concat_keypoints("keypoints_DISK.h5", "keypoints_LOF.h5", "keypoints_temp03.h5")
            concat_matches("matches_DISK.h5", "matches_LOF.h5", "matches_temp03.h5", klen2)
            gc.collect()



        # #画图函数
        #     for Z in index_pairs:
        #         print(img_fnames[Z[0]])
        #         draw(img_fnames[Z[0]], img_fnames[Z[1]],"keypoints_LOF.h5","matches_LOF.h5")

        #second stage, 一边裁剪一遍特征提取
        # LOCAL_FEATURE = 'Crop_DISK'
        # if LOCAL_FEATURE == 'Crop_DISK':
        #     detect_features(img_fnames,
        #                     2048,
        #                     feature_dir=feature_dir,
        #                     upright=True,
        #                     device=device,
        #                     resize_small_edge_to=600,
        #                     name="keypoints_temp02.h5"
        #                     )
        #     gc.collect()
        #     t = time() - t
        #     timings['feature_detection'].append(t)
        #     print(f'Features detected in  {t:.4f} sec')
        #     t = time()
        #     match_features(img_fnames, index_pairs, feature_dir=feature_dir, device=device)
        #     #先crop后的再crop前的
        #
        #     #klen3 = concat_keypoints("keypoints_Crop_DISK.h5", "keypoints_temp02.h5", "keypoints_temp03.h5")
        #     klen3 = concat_keypoints("keypoints_temp02.h5", "keypoints_Crop_DISK.h5", "keypoints_temp03.h5")
        #     # concat_matches("matches_Crop_DISK.h5", "matches_temp02.h5", "matches_temp03.h5", klen3)
        #     concat_matches("matches_temp02.h5", "matches_Crop_DISK.h5", "matches_temp03.h5", klen3)
        #
        #
        # LOCAL_FEATURE = 'Crop_KeyNetAffNetHardNet'
        # if LOCAL_FEATURE == 'Crop_KeyNetAffNetHardNet':
        #     detect_features(img_fnames,
        #                     2048,
        #                     feature_dir=feature_dir,
        #                     upright=True,
        #                     device=device,
        #                     resize_small_edge_to=600,
        #                     name="keypoints_temp03.h5"
        #                     )
        #     gc.collect()
        #     t = time() - t
        #     timings['feature_detection'].append(t)
        #     print(f'Features detected in  {t:.4f} sec')
        #     t = time()
        #     match_features(img_fnames, index_pairs, feature_dir=feature_dir, device=device)
        #     #先crop后的再crop前的
        #
        #     #klen4 = concat_keypoints("keypoints_Crop_KeyNetAffNetHardNet.h5", "keypoints_temp03.h5", "keypoints.h5")
        #     klen4 = concat_keypoints("keypoints_temp03.h5", "keypoints_Crop_KeyNetAffNetHardNet.h5", "keypoints_temp04.h5")
        #     #concat_matches("matches_Crop_KeyNetAffNetHardNet.h5", "matches_temp03.h5", "matches.h5", klen4)
        #     concat_matches("matches_temp03.h5", "matches_Crop_KeyNetAffNetHardNet.h5", "matches_temp04.h5", klen4)

            # for Z in index_pairs:
            #     print(img_fnames[Z[1]])
            #     draw(img_fnames[Z[0]], img_fnames[Z[1]], "keypoints_Crop_KeyNetAffNetHardNet.h5", "matches_Crop_KeyNetAffNetHardNet.h5")

        LOCAL_FEATURE = 'Crop'
        if LOCAL_FEATURE == 'Crop':
            match_crop(img_fnames, index_pairs, feature_dir=feature_dir, device=device, resize_to_=(800, 600),name='keypoints_temp03.h5')
            klen5 = concat_keypoints("keypoints_temp03.h5", "keypoints_Crop.h5", "keypoints.h5")
            concat_matches("matches_temp03.h5", "matches_Crop.h5", "matches_bef.h5", klen5)
            gc.collect()
        #     for Z in index_pairs:
        #         print(img_fnames[Z[1]])
        #         draw(img_fnames[Z[0]], img_fnames[Z[1]], "keypoints_Crop.h5", "matches_Crop.h5")
        # #


#match文件格式可能有点小问题？？？

        filter_OA(img_fnames, index_pairs, feature_dir=feature_dir,device=device)
        for Z in index_pairs:
            print(img_fnames[Z[1]])
            draw(img_fnames[Z[0]], img_fnames[Z[1]], "keypoints.h5", "matches.h5")

#         t = time() - t
#         timings['feature_matching'].append(t)
#         print(f'Features matched in  {t:.4f} sec')
#         database_path = f'{feature_dir}/colmap.db'
#         if os.path.isfile(database_path):
#             os.remove(database_path)
#         gc.collect()
#
#         import_into_colmap(img_dir, feature_dir=feature_dir, database_path=database_path)
#         output_path = f'{feature_dir}/colmap_rec_{LOCAL_FEATURE}'
#
#         t = time()
#         pycolmap.match_exhaustive(database_path)
#         t = time() - t
#         timings['RANSAC'].append(t)
#         print(f'RANSAC in  {t:.4f} sec')
#
#         t = time()
#         # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
#         mapper_options = pycolmap.IncrementalMapperOptions()
#         mapper_options.min_model_size = 3
#         os.makedirs(output_path, exist_ok=True)
#         maps = pycolmap.incremental_mapping(database_path=database_path, image_path=img_dir,
#                                             output_path=output_path, options=mapper_options)
#         # clear_output(wait=False)
#         t = time() - t
#         timings['Reconstruction'].append(t)
#         print(f'Reconstruction done in  {t:.4f} sec')
#         imgs_registered = 0
#         best_idx = None
#         print("Looking for the best reconstruction")
#         if isinstance(maps, dict):
#             for idx1, rec in maps.items():
#                 print(idx1, rec.summary())
#                 if len(rec.images) > imgs_registered:
#                     imgs_registered = len(rec.images)
#                     best_idx = idx1
#         if best_idx is not None:
#             print(maps[best_idx].summary())
#             for k, im in maps[best_idx].images.items():
#                 key1 = f'{dataset}/{scene}/images/{im.name}'
#                 out_results[dataset][scene][key1] = {}
#                 out_results[dataset][scene][key1]["R"] = deepcopy(im.rotmat())
#                 out_results[dataset][scene][key1]["t"] = deepcopy(np.array(im.tvec))
#         print(f'Registered: {dataset} / {scene} -> {len(out_results[dataset][scene])} images')
#         print(f'Total: {dataset} / {scene} -> {len(data_dict[dataset][scene])} images')
#         create_submission(out_results, data_dict)
#         gc.collect()
#
#
# create_submission(out_results, data_dict)