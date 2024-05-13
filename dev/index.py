import os
import pickle
import datetime
import numpy as np
import datetime as dt
from utils import DocumentVectors


def main():
    docs_path = '/home/csgrad/indratej/my_projects/thesis/data/pdfs/imgs/'
    models_path = './models'
    version = datetime.datetime.now().strftime("%m_%d_%Y_%H:%M:%S")
    obj_path = models_path+'/instance_'+version+'.pkl'

    if os.path.exists(models_path) is False:
        os.mkdir(models_path)
        print("created directory at", models_path)

    if os.path.exists(obj_path):
        with open(obj_path, "rb") as file:
            dv = pickle.load(file)
            print(f'Object successfully loaded from "{obj_path}"')
    else:
        start = dt.datetime.now()
        dv = DocumentVectors(docs_path)
        print(docs_path)
        dv.extract_context_vectors()
        dv.build_context_index()
        with open(obj_path, 'wb') as file:
            pickle.dump(dv, file)
            print(f'Object successfully saved at "{obj_path}"')
        end = dt.datetime.now()
        print("total time taken:", (end-start))

if __name__ == "__main__":
   main()




