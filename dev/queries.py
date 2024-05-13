import os
import cv2
import pickle
import numpy as np
import pandas as pd
import datetime
import itertools
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew
from scipy.spatial.distance import cdist
from utils import build_context_layout,qry_image_size,doc_image_size

min_angle=-5
max_angle=5

def extract_ROI(image):
    
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, 0, -1)

    # Dilate to merge into a single contour
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,30))
    dilate = cv2.dilate(thresh, vertical_kernel, iterations=3)

    # Find contours, sort for largest contour and extract ROI
    cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:-1]
    if len(cnts) == 0:
        return image
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 4)
        return image[y:y+h, x:x+w]
        
def deskew(image):
    grayscale = rgb2gray(image)
    angle = determine_skew(grayscale)
    print("angle:", angle)
    if angle >= max_angle or angle < min_angle:
        return image
    rotated = (rotate(image, angle, resize=True) * 255).astype(np.uint8)
    # rotated_norm = cv2.resize(rotated,qry_image_size,interpolation = cv2.INTER_LINEAR)
    # ar = rotated_norm.shape[0]/rotated.shape[1]
    # if ar < 1:
    #     rotated_norm = (rotate(rotated_norm, 90, resize=True) * 255).astype(np.uint8)
    rotated_norm = (rotate(rotated, 180, resize=True) * 255).astype(np.uint8)
    return rotated_norm

def preprocess(img):
    # img = extract_ROI(img)
    img = deskew(img)
    return img

def custom_plot_matches(q, doc_path, score, matches, id, plots_path):   
        matches = np.array(matches)
        m1 = np.round(matches[:,0,:],2)
        m2 = np.round(matches[:,1,:],2)
        doc = cv2.imread(doc_path)
        d = cv2.resize(doc,doc_image_size,interpolation = cv2.INTER_LINEAR)
        h = max(q.shape[0],d.shape[0])
        w = q.shape[1]+d.shape[1]
        point_img = np.full((h,w,3),255) #draw and image of 2* doc-image
        point_img[:q.shape[0],:q.shape[1],:] = q #fill top-left with query
        point_img[:d.shape[0],q.shape[1]:,:] = d #fill top-right with document
        point_img = point_img.astype('uint8')
        output_image = point_img.copy()
        
        for match1, match2 in zip(m1, m2):
            #difference between a match should be in range of (mean_centroid-threshold, mean_centroid+threshold)
            (x1, y1) = match1
            (x2, y2) = match2
            x2 += q.shape[1]
            color = (0, 255, 0)
            cv2.circle(output_image, (int(x1),int(y1)), radius=8, color=color, thickness=5)
            cv2.circle(output_image, (int(x2),int(y2)), radius=8, color=color, thickness=5)
            cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        path = plots_path+'/point_match_{}.png'.format(id)
        plt.figure()
        plt.title('id:{}-score:{}'.format(id,score))
        plt.imshow(output_image)
        plt.savefig(path)
        plt.close()
        return path

def plot_interims(interims, id, transforms_path):
    path = transforms_path+'/query_transforms_{}.png'.format(id)
    fig, axs = plt.subplots(1, 5, figsize=(20, 6))
    axs[0].imshow(interims[0])
    axs[0].set_title("image")
    axs[1].imshow(interims[1])
    axs[1].set_title("preprocessed")
    axs[2].imshow(interims[2],cmap='gray')
    axs[2].set_title("thresh")
    axs[3].imshow(interims[3])
    axs[3].set_title("components")
    axs[4].imshow(interims[4])
    axs[4].set_title("feature word components")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.2)
    fig.savefig(path)
    plt.close()
    return path

def verification(dv, candidates, query_vectors, query_coordinates, m_matches=30):
    def get_orientation(matrix):
            determinant = np.linalg.det(matrix)
            orientation = np.sign(determinant) 
            return orientation
    def verfiy_orientation(P1,P2):
            s1 = get_orientation(P1)
            s2 = get_orientation(P2)
            verif = s1 * s2
            return int(verif)
    triplet_scores= {}
    query_coordinates = np.array(query_coordinates).reshape(len(query_coordinates),2).round(2)
    curr_score = -1
    curr_matches = None
    res_path = None
    #for each candidate doc, build triplet score
    for cd,_ in candidates: 
        dc,doc_vectors, path = dv.docs[cd].label_coordinates, dv.docs[cd].feature_vectors, dv.docs[cd].img_path
        doc_coords = np.array(dc).reshape(len(dc),2).round(2)
        doc_vectors = np.array(doc_vectors).reshape(len(doc_vectors),20).round(2)
        distances = np.round(cdist(query_vectors, doc_vectors, metric='euclidean'),2) 
        edges_dict = {(tuple(query_coordinates[i]), tuple(doc_coords[j])): distances[i, j] for i in range(query_coordinates.shape[0]) for j in range(doc_coords.shape[0])}
        min_edges = sorted(edges_dict.items(), key=lambda x: x[1])
        triplets,count = [],1
        p1_set,p2_set = set(),set()
        for edge in min_edges:
            P1,P2,d = edge[0][0],edge[0][1], edge[1]
            if(P1 not in p1_set) and (P2 not in p2_set):
                p1_set.add(P1)
                p2_set.add(P2)
                triplets.append((P1,P2))
                count += 1
            if count >= m_matches:
                break
        triplet_combinations = np.array(list(itertools.combinations(triplets, 3)))
        matches_score = 0
        for elm in triplet_combinations:
            s1 = np.concatenate([elm[:,0,:], np.ones((elm[:,0,:].shape[0], 1))], axis=1) #2,3 => 3,3 matrix
            s2 = np.concatenate([elm[:,1,:], np.ones((elm[:,1,:].shape[0], 1))], axis=1) #2,3 => 3,3 matrix
            matches_score += verfiy_orientation(s1,s2) #
        triplet_scores[cd] = matches_score # final triplet matches score for candidate cd
        if(matches_score > curr_score):
                curr_score = matches_score
                curr_matches = triplets
                res_path = path
    # print("triplet verification scores:",sorted(triplet_scores.items(), key=lambda x:x[1], reverse=True))
    results = sorted(triplet_scores.items(), key=lambda x:x[1], reverse=True)
    max_score = results[0][1]
    doc_id = results[0][0]
    return results, doc_id,max_score, curr_matches, res_path
    
def query(dv, query_img,qry_id,matches_path, max_candidates=30,min_qry_vectors=10,min_score=1000):
        query_vectors,query_coordinates,interim = build_context_layout(query_img)
        if len(query_vectors)<min_qry_vectors: #return if <10 query vectors found
            print("found only",len(query_vectors),"query vectors. cannot query with this img")
            return
        query_vectors = query_vectors.reshape(len(query_vectors),20)
        index = np.array(list(dv.context_index.keys()))
        docs = np.array(list(dv.context_index.values()))
        ud = [list(i)for i in docs]
        unique_docs = [item for row in ud for item in row]
        coverage_scores = {}
        distances = cdist(query_vectors, index) #extract distances between each pair of index-vectors,query-vectors 
        best_match_indices = np.argmin(distances, axis=1) #filter best matches of pairs
        for bm in best_match_indices: #build coverage scores(no of matched index-vectors) for each document.
            for elm in set(docs[bm]):
                if elm in coverage_scores:
                    coverage_scores[elm] += 1
                else:
                    coverage_scores[elm] = 1
        candidates = sorted(coverage_scores.items(), key=lambda x:x[1], reverse=True)#[:max_candidates]
        finalists, doc_id, verif_score, matches, doc_path = verification(dv,candidates, query_vectors, query_coordinates)
        doc_path = dv.docs[doc_id].img_path
        plot_path = custom_plot_matches(query_img,doc_path,verif_score,matches,qry_id,matches_path)    
        exists = False
        if verif_score > min_score:
            exists = True
        return exists,doc_id, doc_path,plot_path,[candidates, finalists],interim

def inference(dv , test_folder, matches_path, transforms_path, df_path):
    # qry_files = [file for file in os.listdir(test_folder) if file.endswith('.png') or file.endswith('.jpeg') or file.endswith('.jpg')] #filter only image files
    # qry_file_paths = [os.path.join(test_folder, file_name) for file_name in qry_files] 
    labels_df = pd.read_csv(test_folder+'query_labels.csv')
    df = pd.DataFrame()
    excep_count = []
    for idx,row in labels_df.iterrows():
        try:
            interims = []
            start = datetime.datetime.now()
            label = row.values[1]
            img_path = row.values[0]
            qry = cv2.imread(img_path)
            interims.append(qry)
            print("1", img_path)
            qry = preprocess(qry)
            interims.append(qry)
            print("2")
            exists,doc_id,doc_path,custom_plot,scores,interims2 = query(dv,qry,idx,matches_path)
            # print("20", scores[0][0], scores[1][0])
            # if exists is False:
            #     qry_90 = (rotate(qry, angle=90, resize=True) * 255).astype(np.uint8)
            #     exists,doc_id,doc_path,custom_plot,scores,interims2 = query(dv,qry_90,idx,matches_path)
            #     print("21", scores[0][0], scores[1][0])
            #     if exists is False:
            #         qry_180 = (rotate(qry_90, angle=90, resize=True) * 255).astype(np.uint8)
            #         exists,doc_id,doc_path,custom_plot,scores,interims2 = query(dv,qry_180,idx,matches_path)
            #         print("22", scores[0][0], scores[1][0])
            #         if exists is False:
            #             qry_270 = (rotate(qry_180, angle=90, resize=True) * 255).astype(np.uint8)
            #             exists,doc_id,doc_path,custom_plot,scores,interims2 = query(dv,qry_270,idx,matches_path)
            #             print("23", scores[0][0], scores[1][0])
            #             if exists is False:
            #                 qry_360 = (rotate(qry_270, angle=90, resize=True) * 255).astype(np.uint8)
            #                 exists,doc_id,doc_path,custom_plot,scores,interims2 = query(dv,qry_360,idx,matches_path)
            #                 print("24", scores[0][0], scores[1][0])
            print("3")
            index_scores, triplet_scores = scores
            label = doc_id
            idx_rank,idx_score = next(((i+1, item[1])for i, item in enumerate(index_scores) if item[0] == label), -1)
            verif_rank,verif_score = next(((i+1, item[1]) for i, item in enumerate(triplet_scores) if item[0] == label), -1)
            end = datetime.datetime.now()
            interims = interims + interims2
            print("4", len(interims))
            transforms_plot = plot_interims(interims,idx,transforms_path)
            new_row = [img_path,exists,doc_path,int(doc_id),transforms_plot,custom_plot,idx_rank,idx_score,verif_rank,verif_score, int(label), end-start]
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            print(new_row)
            print("\nlabel:",label,"\n")
            print("index_scores:",index_scores,"\n")
            print("idx_rank:",idx_rank,"idx_score:",idx_score,"\n")
            print("triplet_scores:",triplet_scores,"\n")
            print("verif_rank:",verif_rank,"verif_score:",verif_score,"\n")
            print("\n\n")
        except Exception as e:
            print("Error at Query:",idx,img_path,e)
            excep_count.append(row.values[0])
        if idx > 1:
            break
    if len(df) > 0:
        df.columns = ['qry_file_paths','exists','doc_file_paths','return_doc_id','transforms_plot','point_matching_path',
                      'idx_rank','idx_score','verif_rank','verif_score', 'label', 'time_taken']
        df.to_csv(df_path,index=False)
        if excep_count:
            print('couldnt query {} imgs. {}'.format(len(excep_count),excep_count))
        print("saved retreival results at",df_path, len(df))
        
    return df

def main():
    query_path = '/home/csgrad/indratej/my_projects/thesis/data/pdfs/queries/'
    index_path = '/home/csgrad/indratej/my_projects/thesis/2k-cropset/models/instance_04_04_2024_19:18:04.pkl'
    res_path = './results'
    matches_path = res_path+'/matches'
    transforms_path = res_path+'/transforms'
    df_path = res_path +'/query_results.csv'

    if os.path.exists(res_path) is False:
        os.mkdir(res_path)
        print("created directory at",res_path)

    if os.path.exists(matches_path) is False:
        os.mkdir(matches_path)
        print("created directory at",matches_path)

    if os.path.exists(transforms_path) is False:
        os.mkdir(transforms_path)
        print("created directory at",transforms_path)

    with open(index_path, "rb") as file:
        dv = pickle.load(file)
        print(f'index object successfully loaded from "{index_path}"')

    inference(dv , query_path, matches_path, transforms_path, df_path)

if __name__ == "__main__":
   main()
