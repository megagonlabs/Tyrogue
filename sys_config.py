import os

import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("Cuda available:", torch.cuda.is_available())

# print("torch:", torch.__version__)
# print("Cuda:", torch.backends.cudnn.cuda)
# print("CuDNN:", torch.backends.cudnn.version())

glue_datasets = ['sst-2', 'sst-2-imbalance', 'qnli', "qqp"]
available_datasets = glue_datasets + ["ag_news", "dbpedia", "imdb", "pubmed", "20ng"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

# DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_DIR = '/nfs/shared/labeler_al/data'

# EXP_DIR = os.path.join(BASE_DIR, 'experiments')
EXP_DIR = '/nfs/shared/labeler_al/experiments'

# CACHE_DIR = os.path.join(BASE_DIR, 'cache')
CACHE_DIR = '/nfs/shared/labeler_al/cache'

ANALYSIS_DIR = os.path.join(BASE_DIR, 'analysis')

acquisition_functions = ["random", "entropy", "alps", "FTbertKM", "badge", "cal", "naive", "rand_clus_unc", "rand_unc_clus", "rand_unc_clus_adapt"]
