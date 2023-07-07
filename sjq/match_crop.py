import torch
from superglue.models.matching import Matching
import os
import kornia.feature as KF
import h5py
from torchvision import utils as vutils
from fastprogress import progress_bar
import numpy as np
from collections import defaultdict
import kornia as K
from copy import deepcopy
from load_image import load_torch_image
from collections import Counter
from DBSCAN import radius
from sklearn.cluster import DBSCAN
from match_tools import get_unique_idxs
from superpoint import crop_superpoint
from loftr import crop_loftr

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
            print(key1)
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
            resize_to_ = (630, 840)
            mkpts0_loftr_840, mkpts1_loftr_840 = crop_loftr(matcher_loftr, timg1, timg2, resize_to_, l1, d1, l2, d2)
            # resize_to_ = (960, 1280)
            # mkpts0_sup_1280, mkpts1_sup_1280 = crop_superpoint(matcher_superpoint, timg1, timg2, fname1,
            #                                                    fname2, resize_to_, l1, d1, l2, d2)
            # resize_to_ = (768, 1024)
            # mkpts0_sup_1024, mkpts1_sup_1024 = crop_superpoint(matcher_superpoint, timg1, timg2, fname1,
            #                                                    fname2, resize_to_, l1, d1, l2, d2)
            resize_to_ = (1152, 1536)
            mkpts0_sup_1536, mkpts1_sup_1536 = crop_superpoint(matcher_superpoint, timg1, timg2, fname1,
                                                               fname2, resize_to_, l1, d1, l2, d2)

            mkpts0 = np.concatenate(
                (mkpts0_loftr_840, mkpts0_sup_1536), axis=0)
            mkpts1 = np.concatenate(
                (mkpts1_loftr_840, mkpts1_sup_1536), axis=0)
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

