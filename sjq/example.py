# General utilities
from time import time
from filter_OAnet import filter_OA
from concat_h5 import concat_matches, concat_keypoints
import gc
from draw import draw
from Create_submission import create_submission
from detect_features import detect_features
from match_features import match_features
import matplotlib
from import_into_colmap import import_into_colmap
from match_crop import match_crop
from loftr import match_loftr
from superpoint import match_superpoint
from load_image import load_image
import numpy as np
from IPython.display import clear_output
from copy import deepcopy
# CV/ML
from get_image_pairs import get_image_pairs_shortlist
import torch
import kornia as K

# 3D reconstruction
# import pycolmap
import sys
sys.path.append('..')
matplotlib.use('TkAgg')
print('Kornia version', K.__version__)
# print('Pycolmap version', pycolmap.__version__)

LOCAL_FEATURE = 'DISK'
device = torch.device('cuda')

import os



src = '../dataset'
# Get data from csv.
data_dict = load_image(src)
out_results = {}
timings = {"shortlisting": [],
           "feature_detection": [],
           "feature_matching": [],
           "RANSAC": [],
           "Reconstruction": []}

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
                            resize_small_edge_to=600,
                            LOCAL_FEATURE = LOCAL_FEATURE
                            )
            gc.collect()
            t = time() - t
            timings['feature_detection'].append(t)
            print(f'Features detected in  {t:.4f} sec')
            t = time()
            match_features(img_fnames, index_pairs, feature_dir=feature_dir, device=device, LOCAL_FEATURE=LOCAL_FEATURE)
        #     for Z in index_pairs:
        #         print(img_fnames[Z[0]])
        #         draw(img_fnames[Z[0]], img_fnames[Z[1]],"keypoints_DISK.h5","matches_DISK.h5",feature_dir,LOCAL_FEATURE)

        #
        #
        LOCAL_FEATURE = 'KeyNetAffNetHardNet'
        if LOCAL_FEATURE == 'KeyNetAffNetHardNet':
            detect_features(img_fnames,
                            1024,
                            feature_dir=feature_dir,
                            upright=True,
                            device=device,
                            resize_small_edge_to=600,
                            LOCAL_FEATURE = LOCAL_FEATURE
                            )
            gc.collect()
            t = time() - t
            timings['feature_detection'].append(t)
            print(f'Features detected in  {t:.4f} sec')
            t = time()
            match_features(img_fnames, index_pairs, feature_dir=feature_dir, device=device, LOCAL_FEATURE=LOCAL_FEATURE)
        #拼接顺序先affnet再disk
            klen1 =concat_keypoints("keypoints_KeyNetAffNetHardNet.h5", "keypoints_DISK.h5", "keypoints_temp01.h5",feature_dir)
            concat_matches("matches_KeyNetAffNetHardNet.h5", "matches_DISK.h5", "matches_temp01.h5", klen1,feature_dir)
        #
        #     for Z in index_pairs:
        #         print(img_fnames[Z[0]])
        #         draw(img_fnames[Z[0]], img_fnames[Z[1]],"keypoints_KeyNetAffNetHardNet.h5","matches_KeyNetAffNetHardNet.h5",feature_dir,LOCAL_FEATURE)
        LOCAL_FEATURE = 'superpoint'
        if LOCAL_FEATURE == 'superpoint':
            match_superpoint(img_fnames, index_pairs, feature_dir=feature_dir, device=device, resize_to_=(600, 800))
            klen2 = concat_keypoints("keypoints_temp01.h5","keypoints_SUP.h5", "keypoints_temp02.h5",feature_dir)
            concat_matches("matches_temp01.h5", "matches_SUP.h5", "matches_temp02.h5", klen2,feature_dir)
            gc.collect()
            # for Z in index_pairs:
            #     print(img_fnames[Z[0]])
            #     draw(img_fnames[Z[0]], img_fnames[Z[1]],"keypoints_SUP.h5","matches_SUP.h5",feature_dir,LOCAL_FEATURE)


        LOCAL_FEATURE = 'LoFTR'
        if LOCAL_FEATURE == 'LoFTR':
            match_loftr(img_fnames, index_pairs, feature_dir=feature_dir, device=device, resize_to_=(800, 600))
            klen2 = concat_keypoints("keypoints_temp02.h5", "keypoints_LOF.h5", "keypoints_temp03.h5",feature_dir)
            concat_matches("matches_temp02.h5", "matches_LOF.h5", "matches_temp03.h5", klen2,feature_dir)
            gc.collect()



        #画图函数
            # for Z in index_pairs:
            #     print(img_fnames[Z[0]])
            #     draw(img_fnames[Z[0]], img_fnames[Z[1]],"keypoints_LOF.h5","matches_LOF.h5",feature_dir,LOCAL_FEATURE)

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
        #     match_features(img_fnames, index_pairs, feature_dir=feature_dir, device=device, LOCAL_FEATURE=LOCAL_FEATURE)
        #     #先crop后的再crop前的
        #
        #     #klen3 = concat_keypoints("keypoints_Crop_DISK.h5", "keypoints_temp02.h5", "keypoints_temp03.h5",feature_dir)
        #     klen3 = concat_keypoints("keypoints_temp02.h5", "keypoints_Crop_DISK.h5", "keypoints_temp03.h5",feature_dir)
        #     # concat_matches("matches_Crop_DISK.h5", "matches_temp02.h5", "matches_temp03.h5", klen3,feature_dir)
        #     concat_matches("matches_temp02.h5", "matches_Crop_DISK.h5", "matches_temp03.h5", klen3,feature_dir)
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
        #     match_features(img_fnames, index_pairs, feature_dir=feature_dir, device=device, LOCAL_FEATURE=LOCAL_FEATURE)
        #     #先crop后的再crop前的
        #
        #     #klen4 = concat_keypoints("keypoints_Crop_KeyNetAffNetHardNet.h5", "keypoints_temp03.h5", "keypoints.h5",feature_dir)
        #     klen4 = concat_keypoints("keypoints_temp03.h5", "keypoints_Crop_KeyNetAffNetHardNet.h5", "keypoints_temp04.h5",feature_dir)
        #     #concat_matches("matches_Crop_KeyNetAffNetHardNet.h5", "matches_temp03.h5", "matches.h5", klen4,feature_dir
        #     concat_matches("matches_temp03.h5", "matches_Crop_KeyNetAffNetHardNet.h5", "matches_temp04.h5", klen4,feature_dir)

            # for Z in index_pairs:
            #     print(img_fnames[Z[1]])
            #     draw(img_fnames[Z[0]], img_fnames[Z[1]], "keypoints_Crop_KeyNetAffNetHardNet.h5", "matches_Crop_KeyNetAffNetHardNet.h5",feature_dir,feature_dir,LOCAL_FEATURE)

        LOCAL_FEATURE = 'Crop'
        if LOCAL_FEATURE == 'Crop':
            match_crop(img_fnames, index_pairs, feature_dir=feature_dir, device=device, resize_to_=(800, 600),name='keypoints_temp03.h5')
            klen5 = concat_keypoints("keypoints_temp03.h5", "keypoints_Crop.h5", "keypoints.h5", feature_dir)
            concat_matches("matches_temp03.h5", "matches_Crop.h5", "matches_bef.h5", klen5, feature_dir)
            gc.collect()
            # for Z in index_pairs:
            #     print(img_fnames[Z[1]])
            #     draw(img_fnames[Z[0]], img_fnames[Z[1]], "keypoints.h5", "matches_bef.h5",feature_dir,LOCAL_FEATURE)

        #
        filter_OA(img_fnames, index_pairs, feature_dir=feature_dir,device=device)
        for Z in index_pairs:
            print(img_fnames[Z[1]])
            draw(img_fnames[Z[0]], img_fnames[Z[1]], "keypoints.h5", "matches.h5",feature_dir,LOCAL_FEATURE)

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