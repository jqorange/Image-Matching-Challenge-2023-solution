import h5py
import torch
import numpy as np
def concat_matches(name1, name2, name3,length,feature_dir):
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


def concat_keypoints(name1, name2, name3, feature_dir):
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