import os
import random
import numpy as np
import pandas as pd
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchsummary import summary
from torch.utils.data import SubsetRandomSampler,Subset,DataLoader
from vit_pytorch import ViT 

# 1crop - n_augs Transforms
# 1crop - n_augs Transforms
def Customtransforms(img, idx, p_path, n_augs=4):
      
      k = random.uniform(2.5,4) #[25%-40%]
      crop_size = int(min(int(img.size[1]/k), int(img.size[0]/k)))
      center_crop = T.RandomCrop(size=crop_size)
      cc = center_crop(img)
      augmenter = T.RandAugment()
      augs = [augmenter(cc) for _ in range(n_augs)]

      patches = [cc] + augs   
      paths = []     
      for i,patch in enumerate(patches):
            path = p_path+'/{}_{}.png'.format(idx, i)
            paths.append(path)
            patch.save(path) 
      return paths

def generate_samples_index(doc_images_path, n_augs):
    all_items = os.listdir(doc_images_path)
    doc_items = [item for item in all_items if os.path.isfile(os.path.join(doc_images_path, item)) and item.lower().endswith('.png')]
    docs = [os.path.join(doc_images_path, file_name) for file_name in doc_items]

    p_path = './data/patches'
    if os.path.exists(p_path) is False:
        os.mkdir(p_path)

    start = datetime.datetime.now()
    df = pd.DataFrame(columns=['ImagePath', 'PatchPath'])
    print("images found:", len(docs))
    for idx, filepath in enumerate(docs):
        try:
            img = Image.open(filepath)
            patch_paths = Customtransforms(img, idx, p_path, n_augs=n_augs)
            for patch_path in patch_paths:
                df = df.append({'ImagePath':filepath, 'PatchPath':patch_path}, ignore_index=True)
            if idx%100 == 0:
                print("sample-pairs extracted:",df.shape[0],":=:" ,np.round(idx/len(docs)*100, 2),"%. done")
                df.to_csv(p_path+'/patches_index.csv',index=False)
        except Exception as e:
            print(filepath, e)
    df.to_csv(p_path+'/patches_index.csv',index=False)
    end = datetime.datetime.now()
    print("total samples:",len(df))
    print("time taken:", end-start)

# docpath = ("/home/csgrad/indratej/my_projects/incidental_capture/mobile_retriever/data/docs/")
docpath = '/home/csgrad/indratej/my_projects/thesis/data/pdfs/imgs/'
generate_samples_index(docpath,n_augs=8)
#[1] 2311847
