# ImageMatching-IC
this project has been performed as Camera Captured Image Matching for Incidental Capture. expts contains all the experiments performed. helpers contains python helper scripts from experiments. dev contains the only required development code to reproduce the project/task.
rest explain themselves. all the code is python 3+.

steps to reproduce.
1. place ground-truth images in a folder. 
2. run index.py(dev/index.py)[change the foldername in index.py]
3. place query images in a seperate folder.
4. run queries.py(dev/queries.py)[change the foldername in queries.py]
5. run diagnostics.ipynb to obtain results.

results contain accuracy, precision, indexing & verification plots for the given queries.

