from torchvision.transforms import Resize
from superglue.models.utils import (read_image)
import torch
from superglue.models.matching import Matching
import h5py
from torchvision import utils as vutils
from fastprogress import progress_bar
import numpy as np
from collections import defaultdict
import kornia as K
from copy import deepcopy
from load_image import load_torch_image
from match_tools import get_unique_idxs
device = torch.device('cuda')

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
            # if H1 < W1:
            #     resize_to = resize_to_[1], resize_to_[0]
            # else:
            resize_to = resize_to_
            timg_resized1 = K.geometry.resize(timg1, resize_to, antialias=True)
            h1, w1 = timg_resized1.shape[2:]

            torch_resize = Resize(resize_to_)
            image_1 =torch_resize(load_torch_image(fname1, device=device))
            vutils.save_image(image_1, f_cropname1, normalize=True)


            # Load img2
            H2, W2 = timg2.shape[2:]
            # if H2 < W2:
            #     resize_to2 = resize_to[1], resize_to[0]
            # else:
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