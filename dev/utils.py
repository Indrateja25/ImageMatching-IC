import os
import cv2
import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift

doc_image_size = (1200,1600)
qry_image_size = (600, 800)
pixel_row_gap = 10
k_neigh = 5
min_ar, max_ar = 2.0, 7.0

def binarize(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,129,4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    result = 255 - opening
    return result

def generate_stats(totalLabels,stats, centroids):
    columns = ['label_id', 'left','top','width','height','area','centroid_x','centroid_y']
    df = pd.DataFrame(columns=columns)
    df['label_id'] = [i for i in range(1,totalLabels+1)]
    df['left'] = stats[:,0]
    df['top'] = stats[:,1]
    df['width'] = stats[:,2]
    df['height'] = stats[:,3]
    df['area'] = stats[:,4]
    df['centroid_x'] = centroids[:,0]
    df['centroid_y'] = centroids[:,1]

    df['aspect_ratio'] = np.round(df['width']/df['height'],2)
    df = df.sort_values(by=['top', 'left']).reset_index(drop=True) #sort-by occurence
    
    #calculate rowID, colID
    df['rowID'] = 1
    row_id = 1
    prev_val = df.loc[0, 'top']
    for idx in range(1, df.shape[0]):
        curr_val = df.loc[idx, 'top']
        if curr_val - prev_val > pixel_row_gap:
            row_id += 1
        prev_val = curr_val
        df.at[idx, 'rowID'] = int(row_id)
    df['columnID'] = df.groupby('rowID')['left'].rank().astype(int)
    return df
  
def build_single_lexicon(label_id, df):
    if label_id not in df['label_id'].values:
        print("unknown component")
        return
    
    ar = df[df['label_id'] == label_id]['aspect_ratio'].values[0]
    if ar < min_ar or ar > max_ar:
        return

    #filter by current component and extract candidate neighbors
    x,y,w,h,r,c = df[df['label_id'] == label_id][['centroid_x','centroid_y','width','height','rowID','columnID']].values[0]
    r1 = (df['rowID'] >= r-1)
    r2 = (df['rowID'] <= r+1)    
    temp_df = df[r1 & r2]
    temp_df = temp_df[temp_df['label_id'] != label_id]
    
    #rotation-invariance - TBD during preprocessing
    
    #scale-invariance
    temp_df['new_centroid_x'] = (temp_df['centroid_x']-x)/w
    temp_df['new_centroid_y'] = (y-temp_df['centroid_y'])/h
    temp_df['new_left'] = (temp_df['left']-x)/w
    temp_df['new_top'] = (y-temp_df['top'])/h
    temp_df['new_width'] = temp_df['width']/w
    temp_df['new_height'] = temp_df['height']/h
    
    #calculate distance & angle
    temp_df['Euclidean'] = np.sqrt(temp_df['new_centroid_y']**2 + temp_df['new_centroid_x']**2)
    temp_df['theta'] = np.degrees(np.arctan2(temp_df['new_centroid_y'], temp_df['new_centroid_x']))
    temp_df['theta'] = (temp_df['theta'] + 360) % 360 
    temp_df.loc[temp_df['theta'] > 350, 'theta'] = 0 #heuristic
    temp_df.loc[temp_df['theta'] < 2, 'theta'] = 0 #heuristic
    temp_df['quadrant'] = pd.cut(temp_df['theta'], 8, labels=range(1,9))
    
    #sort and retreive top-K, format required coordinates.
    res_df = temp_df.sort_values(by=['Euclidean','theta'])[:k_neigh] #sort and get top-k neighbors
    res_df = res_df.sort_values(by=['theta'])
    res_df['tl_corner'] = res_df.apply(lambda row: (row['new_left'], row['new_top']), axis=1)
    res_df['br_corner'] = res_df.apply(lambda row: (row['new_left']+row['new_width'], row['new_top']+row['new_height']), axis=1)           
    return res_df

def build_context_layout(img):
    #preprocess and transform image
    thresh = binarize(img)
    totalLabels, labels, stats, centroid = cv2.connectedComponentsWithStats(thresh,4,cv2.CV_32S)
    df = generate_stats(totalLabels,stats, centroid)
    #plot all identified word components
    output = img.copy() 
    for comp in df['label_id'].values:
        x, y, w, h, area = df[df['label_id'] == comp][['left','top','width','height','area']].values[0]
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #extract context for each word component
    context_labels, context_vectors,context_coordinates = [],[],[]
    for label in df.label_id.values:
        try:
            res = build_single_lexicon(label, df)
            if(res is not None and res.shape[0] >= k_neigh):
                context_vectors.append(res[['tl_corner','br_corner']].values)
                coords = df[df['label_id'] == label][['centroid_x','centroid_y']].values
                context_coordinates.append(coords)
                context_labels.append(label)
        except Exception as e:
            print(e)
    
    #plot extracted feature word components
    output2 = img.copy() 
    for comp in context_labels:
        x, y, w, h, area = df[df['label_id'] == comp][['left','top','width','height','area']].values[0]
        cv2.rectangle(output2, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # format into a set of 4/quadrapules
    flattened_tuples = [np.array([item for sublist in t for item in sublist]) for t in context_vectors]
    array_of_tuples = np.array(flattened_tuples)
    context_vectors = array_of_tuples.reshape(len(context_vectors), k_neigh, 4) # 4 is for (Tl-x,TL-y,BR-x,BR-y )
    context_vectors = context_vectors.round(4)
    #print("total no of contexts extracted:",context_vectors.shape[0])
    return context_vectors,context_coordinates,[thresh, output, output2]

class DocumentDetails():
    def __init__(self,img_path, feature_vectors,label_coordinates):
        self.img_path = img_path
        self.feature_vectors = feature_vectors 
        self.label_coordinates = label_coordinates #saves each word coordinates indexed wrt to feature_vectors
        
class DocumentVectors():
    def __init__(self, img_folder_path):
        
        img_files = [file for file in os.listdir(img_folder_path) if file.endswith('.png') or file.endswith('.jpeg') or file.endswith('.jpg')] 
        img_file_paths = [os.path.join(img_folder_path, file_name) for file_name in img_files]
        self.img_file_paths = img_file_paths
        self.docs = {} 
        self.context_index = {}
        self.labels = []

    def extract_context_vectors(self):
        print("found {} images ".format(len(self.img_file_paths)))
        labels_df = pd.DataFrame()
        for id,img_path in enumerate(self.img_file_paths):
            img = cv2.imread(img_path)
            img = cv2.resize(img,doc_image_size,interpolation = cv2.INTER_LINEAR)
            img_name = img_path.split('/')[-1]
            feature_vectors,label_coordinates,_ = build_context_layout(img)
            print(img_name,img_path,id, img.shape, feature_vectors.shape,len(label_coordinates))
            self.docs[id] = DocumentDetails(img_path,feature_vectors,label_coordinates)
            labels_df = pd.concat([labels_df, pd.DataFrame([[img_name,id]])], ignore_index=True)
        labels_df.columns = ['filename', 'label']
        labels_df.to_csv('./query_labels.csv',index=False)
        self.labels = labels_df
    
    def build_context_index(self, ms=True, max_features=500, max_sample_points=10000):
        index = {}
        for id in self.docs:
            cv_all = tuple(self.docs[id].feature_vectors)
            for cv in cv_all:
                index[tuple(list(cv.reshape(-1)))] = id
        print("extracted all context vectors, in total: ",len(index))
        if ms: 
            layouts = list(index.keys())
            meanshift = MeanShift(n_jobs=-1)
            for i in range(0, len(index), max_sample_points):
                minibatch = layouts[i:i+max_sample_points]
                meanshift.fit(minibatch)
            cluster_centers = meanshift.cluster_centers_
            print("MeanShift() model fitting, found {} clusters: ".format(len(cluster_centers)))

            reduced_index = {}
            for i,p in enumerate(layouts):
                p1 = [tuple(np.array(p))]
                cluster = meanshift.predict(p1)
                centroid = tuple(cluster_centers[cluster[0]])
                doc = index[layouts[i]]
                if centroid in reduced_index:
                    reduced_index[centroid].add(doc)
                else:
                    reduced_index[centroid] = set([doc])
                if i%max_sample_points==0:
                    print("building inverted index, completed {}%".format((i+1)/len(layouts)))

            self.context_index = reduced_index
            print("built mean-shift reduced-index, in total: ",len(reduced_index))
        else:
            self.context_index = index
            print("built full-index, in total: ",len(index))
