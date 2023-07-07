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
import os, argparse, h5py, warnings
import numpy as np
from tqdm import tqdm
from PIL import Image, ExifTags
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

LOCAL_FEATURE = 'DISK'
device = torch.device('cuda')




