import cv2
import h5py
import numpy as np
def draw(fname1,fname2,name1,name2,feature_dir,LOCAL_FEATURE):
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
        # cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        # cv2.imshow(title, img_matches)
        cv2.imwrite('./image_match/' +title, img_matches)
        #cv2.imwrite(r'C:\Users\dedll\Desktop\IMC\\together\\'+title,img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
