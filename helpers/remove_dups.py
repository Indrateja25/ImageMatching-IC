import cv2
import numpy as np
import os
import pandas as pd
from skimage.metrics import structural_similarity as ssim

def compute_SSIM(img1_path, img2_path):
    image1 = cv2.imread(img1_path)
    image2 = cv2.imread(img2_path)
    image1 = cv2.resize(image1,(2200, 1700),interpolation = cv2.INTER_LINEAR)
    image2 = cv2.resize(image2,(2200, 1700),interpolation = cv2.INTER_LINEAR)
    # print(image1.shape, image2.shape)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    (score, diff) = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")
    return score


def remove_dups(input_folder, label_folder):
    aoi = os.listdir(input_folder)
    aoi_images = [item for item in aoi if os.path.isfile(os.path.join(input_folder, item)) 
                  and (item.lower().endswith('.png') or item.lower().endswith('.jpeg')  or item.lower().endswith('.jpg') )]
    aoi_image_paths = sorted([os.path.join(input_folder, file_name) for file_name in aoi_images])
    label_image_paths = sorted([os.path.join(label_folder, file_name) for file_name in aoi_images])

    df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([label_image_paths[0]])], ignore_index=True)
    prev = label_image_paths[0]
    for idx,label_path in enumerate(label_image_paths):
        if compute_SSIM(prev, label_path) < 0.95:
            df = pd.concat([df, pd.DataFrame([label_path])], ignore_index=True)
            prev = label_path
        if idx%50 == 0:
            print(idx, len(df))
    print("adj unique found:", len(df))

    # df = pd.read_csv('unique_docs.csv')

    idxlist = []
    dups = set()
    for idx,row in df.iterrows():
        if idx in dups:
            continue
        prev = row.values[0]
        for i,row2 in df.iloc[idx+1:,:].iterrows():
            curr = row2.values[0]
            if compute_SSIM(prev, curr) < 0.95:
                idxlist.append(idx)
            else:
                print(idx, i, prev, curr)
                dups.add(i)

    idxlist = list(set(idxlist))
    print(idxlist, dups)
    df = df.iloc[(idxlist)]
    print("unique found:", len(df))
    df.to_csv('./unique_docs.csv',index=False)

input_folder = './final_cropped_aoi'
label_folder = './final_optimal_gt'
remove_dups(input_folder, label_folder)