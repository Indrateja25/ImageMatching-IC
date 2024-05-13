import os
import random
import numpy as np
import pandas as pd
import datetime
from PIL import Image
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
import torchvision.transforms as T

def Customtransform(img, idx, p_path):

    k = random.uniform(2.5,5) #[25%-50%]
    crop_size = int(min(int(img.size[1]/k), int(img.size[0]/k)))
    center_crop = T.RandomCrop(size=crop_size)
    cc = center_crop(img)
    augmenter = T.RandAugment()
    patch = augmenter(cc)

    path = p_path+'/query_{}.png'.format(idx)
    patch.save(path) 
    return path

def generate_queries(doc_images_path, n_augs):
    all_items = os.listdir(doc_images_path)
    doc_items = [item for item in all_items if os.path.isfile(os.path.join(doc_images_path, item)) and item.lower().endswith('.png')]
    docs = [os.path.join(doc_images_path, file_name) for file_name in doc_items]

    p_path = './data/queries'
    if os.path.exists(p_path) is False:
        os.mkdir(p_path)

    start = datetime.datetime.now()
    df = pd.DataFrame(columns=['PatchPath', 'doc'])
    print("images found:", len(docs))
    for idx, filepath in enumerate(docs):
        try:
            img = Image.open(filepath)
            patch_path = Customtransform(img, idx, p_path)
            doc,page = doc_items[idx][:-4].split('_')
            # print(filepath, doc_items[idx], int(doc), int(page))
            df = df.append({'PatchPath':patch_path, 'doc_label':int(doc), 'page':int(page)}, ignore_index=True)
            if idx%100 == 0:
                print("sample-pairs extracted:",df.shape[0],":=:" ,np.round(idx/len(docs)*100, 2),"%. done")
                df.to_csv(p_path+'/query_index.csv',index=False)
        except Exception as e:
            print(filepath, e)
    df.to_csv(p_path+'/query_index.csv',index=False)
    end = datetime.datetime.now()
    print("total samples:",len(df))
    print("time taken:", end-start)

# docpath = ("/home/csgrad/indratej/my_projects/incidental_capture/mobile_retriever/data/docs/")
docpath = '/home/csgrad/indratej/my_projects/thesis/data/pdfs/imgs/'
generate_queries(docpath,n_augs=8)