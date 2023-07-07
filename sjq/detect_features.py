import torch
import os
import kornia.feature as KF
import h5py
from fastprogress import progress_bar
import numpy as np
import kornia as K
from load_image import load_torch_image
from collections import Counter
from DBSCAN import radius
from sklearn.cluster import DBSCAN
from Affnet import KeyNetAffNetHardNet
def detect_features(img_fnames,
                    num_feats=256,
                    upright=False,
                    device=torch.device('cpu'),
                    feature_dir='.featureout',
                    resize_small_edge_to=600,name = 'None', LOCAL_FEATURE = 'DISK'):
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