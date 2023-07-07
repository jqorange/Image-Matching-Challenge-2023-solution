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
matplotlib.use('TkAgg')
print('Kornia version', K.__version__)
# print('Pycolmap version', pycolmap.__version__)

def load_image(src):
    data_dict = {}
    data_train = 'train_mini'
    train_path = os.path.join(src, data_train)
    dataset_list = [dI for dI in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, dI))]

    for dataset in dataset_list:
        data_dict[dataset] = {}
        dataset_path = os.path.join(train_path, dataset)
        scene_list = [dI for dI in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, dI))]
        for scene in scene_list:
            data_dict[dataset][scene] = [];
            scene_path = os.path.join(dataset_path, scene, 'images')
            image_list = os.listdir(scene_path)
            for images in image_list:
                image = os.path.join(dataset, scene, 'images', images)
                data_dict[dataset][scene].append(image)

    for dataset in data_dict:
        for scene in data_dict[dataset]:
            print(f' -> {len(data_dict[dataset][scene])} images')
    return data_dict

def load_torch_image(fname, device=torch.device('cpu')):
    img = K.image_to_tensor(cv2.imread(fname), False).float() / 255.
    img = K.color.bgr_to_rgb(img.to(device))
    return img