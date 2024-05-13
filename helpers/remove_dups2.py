import cv2
import numpy as np
import os
import pandas as pd
from skimage.metrics import structural_similarity as ssim

def remove_dups(label_folder):
    aoi = os.listdir(label_folder)
    aoi_images = [item for item in aoi if os.path.isfile(os.path.join(label_folder, item)) 
                  and (item.lower().endswith('.png') or item.lower().endswith('.jpeg')  or item.lower().endswith('.jpg') )]
    label_image_paths = sorted([os.path.join(label_folder, file_name) for file_name in aoi_images])

    imgs = set()
    df = pd.DataFrame()
    for idx,label_path in enumerate(label_image_paths):
        if idx%50 == 0:
            print(idx, len(df))
        img = cv2.imread(label_path)
        timg = tuple(list(img.reshape(-1)))
        if timg in imgs:
            continue
        else:
            imgs.add(timg)
            df = pd.concat([df, pd.DataFrame([label_path])], ignore_index=True)
    print(df)
    print("unique found:", len(df))
    df.to_csv('./unique_docs2.csv',index=False)

label_folder = './final_optimal_gt'
remove_dups(label_folder)