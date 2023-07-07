import h5py
import torch
from fastprogress import progress_bar
from OANetmaster.demo.learnedmatcher import LearnedMatcher
import numpy as np
import os
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
