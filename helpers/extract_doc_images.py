# conda install -c conda-forge poppler
# pip install pdf2image

import os
import datetime
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from pdf2image import convert_from_path

import warnings
warnings.filterwarnings('error')
Image.MAX_IMAGE_PIXELS = 1000000000 



def PDFtoImages(pdf_folder_path,target_folder_path=None,dpi=500):
    
    start = datetime.datetime.now()

    pdf_files = [file for file in os.listdir(pdf_folder_path) if file.endswith('.pdf')] #filter only pdf files
    pdf_df = pd.DataFrame(columns=['pdf_name','pdf_path','pdf_id'])
    img_df = pd.DataFrame(columns=['img_name','img_path','img_id', 'pdf_id'])
    if target_folder_path is None:
        target_folder_path = pdf_folder_path+'imgs/'
        if os.path.exists(target_folder_path) is False:
            os.mkdir(target_folder_path) 
    
    for idx,file_name in enumerate(pdf_files):
        pdf_path = os.path.join(pdf_folder_path, file_name)
        try:
            pages = convert_from_path(pdf_path,dpi=dpi)
            pdf_df = pdf_df.append({'pdf_name':file_name, 'pdf_path':pdf_path,'pdf_id':int(idx)}, ignore_index=True)
            for count, page in enumerate(pages):
                img_name = '{}_{}.png'.format(idx, count)
                img_path = target_folder_path+img_name
                page.save(img_path, 'PNG')
                img_df = img_df.append({'img_name':img_name, 'img_path':img_path,'img_id':int(count),'pdf_id':int(idx)}, ignore_index=True)
        except Exception as e:
            print(idx, file_name, e)

    pdf_df.to_csv(pdf_folder_path+'pdf_filenames_index.csv')
    img_df.to_csv(pdf_folder_path+'img_filenames_index.csv')
    end = datetime.datetime.now()

    print("saved pdf file-index at:",pdf_folder_path+'pdf_filenames_index.csv')
    print("saved img file-index at:",pdf_folder_path+'img_filenames_index.csv')
    print("total no of pdf's available:", len(pdf_files))
    print("total no of pdf's able to crawl:", pdf_df.shape[0])
    print("total no of images saved:", img_df.shape[0])
    print("total time taken",end-start)

def main():
    pdf_path = './data/pdfs/'
    PDFtoImages(pdf_path)

if __name__ == '__main__':
    main()