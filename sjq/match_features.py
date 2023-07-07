
import torch
import h5py
from fastprogress import progress_bar
def match_features(img_fnames,
                   index_pairs,
                   feature_dir='.featureout',
                   device=torch.device('cpu'),
                   min_matches=0,
                   force_mutual=True,
                   matching_alg='smnn',
                   LOCAL_FEATURE = 'DISK'
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
                    dists, idxs = KF.match_smnn(desc1, desc2, 0.9)
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
def get_unique_idxs(A, dim=0):
    # https://stackoverflow.com/questions/72001505/how-to-get-unique-elements-and-their-firstly-appeared-indices-of-a-pytorch-tenso
    unique, idx, counts = torch.unique(A, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]
    return first_indices