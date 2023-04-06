import collections
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.dist_metrics import DistanceMetric
from torch import nn
from torch.nn.functional import normalize
from tqdm import tqdm

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from transformers import set_seed, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from utilities.preprocessors import output_modes

# import faiss

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utilities.data_loader import get_glue_tensor_dataset
from utilities.preprocessors import processors
from utilities.trainers import my_evaluate, test_transformer_model
from acquisition.uncertainty import margin_of_confidence, entropy

# from acquisition.bertscorer import calculate_bertscore


logger = logging.getLogger(__name__)

## random -> diversity (clustering) -> uncertainty (entropy)
def tyrogue(args, acq_size, X_original, y_original,
                            labeled_inds, dpool_inds, discarded_inds, original_inds,
                            tokenizer=None,
                            results_dpool=None, logits_dpool=None,
                            bert_representations=None,
                            candidate_scale=None,
                            model=None,
                            init=False):
    print("tyrogue acquisition!")
    embs = 'bert_cls'

    if init: ## For initial acquisition, the method does not consider the output of the BERT model
        r = 1
    else:
        r = args.r

    ## random filter
    if len(dpool_inds) > args.Srand:
        rand = np.random.choice(range(len(dpool_inds)), args.Srand, replace=False)
        rand_inds = np.array(dpool_inds)[rand]
    else:
        rand_inds = np.array(dpool_inds)

    if args.Srand == acq_size:
        sampled_ind = list(rand_inds)
    elif args.Srand > acq_size:
    ## model inference on random filtered samples
        args.task_name = args.task_name.lower()
        args.output_mode = output_modes[args.task_name]
        ori_dataset = get_glue_tensor_dataset(rand_inds, args, args.task_name, tokenizer, train=True)

        if model == None: # For initial selection
            bert_config = AutoConfig.from_pretrained(
                # args.config_name if args.config_name else args.model_name_or_path,
                'bert-base-cased',
                num_labels=args.num_classes,
                finetuning_task=args.task_name,
                cache_dir=args.cache_dir,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                # args.model_name_or_path,
                'bert-base-cased',
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=bert_config,
                cache_dir=args.cache_dir,
            )
        model.to(args.device)

        _, logits_rand, results = test_transformer_model(args, dataset=ori_dataset, model=model, return_cls=True)

        ## clustering filter
        if args.Srand > acq_size*r:
            random_cls = normalize(results[embs]).detach().cpu().numpy()
            centers = kmeans(torch.tensor(random_cls), acq_size*r)
            centers = np.array(centers)
            cand_inds = np.array(rand_inds)[centers]
            logits_cand = torch.tensor(logits_rand[centers])
        else: ## skipping clustering filter
            cand_inds = np.array(rand_inds)
            logits_cand = torch.tensor(logits_rand)

        if r != 1:
            ## uncertain filter
            uncertainty_scores = entropy(logits_cand)
            acq_inds = np.argpartition(uncertainty_scores, -acq_size)[-acq_size:]
            ## map filtered inds to original inds
            sampled_ind = list(cand_inds[acq_inds])
        else: ## skipping uncertainty filter
            sampled_ind = list(cand_inds)

    ########### Calculating statistics of samples (not main function of acquisition) ##############

    y_lab = np.asarray(y_original, dtype='object')[labeled_inds]
    X_unlab = np.asarray(X_original, dtype='object')[dpool_inds]
    y_unlab = np.asarray(y_original, dtype='object')[dpool_inds]

    labels_list_previous = list(y_lab)
    c = collections.Counter(labels_list_previous)
    stats_list_previous = [(i, c[i] / len(labels_list_previous) * 100.0) for i in c]

    new_samples = np.asarray(X_original, dtype='object')[sampled_ind]
    new_labels = np.asarray(y_original, dtype='object')[sampled_ind]

    # Mean and std of length of selected sequences
    if args.task_name in ['sst-2', 'sst-2-imbalance', 'ag_news', 'dbpedia', 'trec-6', 'imdb', 'pubmed', 'sentiment', '20ng']: # single sequence
        l = [len(x.split()) for x in new_samples]
    elif args.dataset_name in ['mrpc', 'mnli', 'qnli', 'cola', 'rte', 'qqp', 'paws-qqp', 'nli']:
        l = [len(sentence[0].split()) + len(sentence[1].split()) for sentence in new_samples]  # pairs of sequences
    assert type(l) is list, "type l: {}, l: {}".format(type(l), l)
    length_mean = np.mean(l)
    length_std = np.std(l)
    length_min = np.min(l)
    length_max = np.max(l)

    # Percentages of each class
    labels_list_selected = list(np.array(y_original)[sampled_ind])
    c = collections.Counter(labels_list_selected)
    stats_list = [(i, c[i] / len(labels_list_selected) * 100.0) for i in c]

    labels_list_after = list(new_labels) + list(y_lab)
    c = collections.Counter(labels_list_after)
    stats_list_all = [(i, c[i] / len(labels_list_after) * 100.0) for i in c]

    assert len(set(sampled_ind)) == len(sampled_ind)
    assert bool(not set(sampled_ind) & set(labeled_inds))
    assert len(new_samples) == acq_size, 'len(new_samples)={}, annotatations_per_it={}'.format(
        len(new_samples),
        acq_size)

    stats = {'length': {'mean': float(length_mean),
                        'std': float(length_std),
                        'min': float(length_min),
                        'max': float(length_max)},
             'class_selected_samples': stats_list,
             'class_samples_after': stats_list_all,
             'class_samples_before': stats_list_previous,
             }

    return sampled_ind, stats

## Swapping filters: random -> uncertainty (entropy) -> diversity (clustering)
def rand_unc_clus(args, acq_size, X_original, y_original,
                            labeled_inds, dpool_inds, discarded_inds, original_inds,
                            tokenizer=None,
                            results_dpool=None, logits_dpool=None,
                            bert_representations=None,
                            candidate_scale=None,
                            model=None,
                            init=False):
    print("rand_unc_clus (swapped) acquisition!")
    embs = 'bert_cls'

    # if init:
    #     r = 1
    # else:
    #     r = args.r
    r = args.r

## random filter
    if len(dpool_inds) > args.Srand:
        rand = np.random.choice(range(len(dpool_inds)), args.Srand, replace=False)
        rand_inds = np.array(dpool_inds)[rand]
    else:
        rand_inds = np.array(dpool_inds)

    if args.Srand == acq_size:
        sampled_ind = list(rand_inds)
    elif args.Srand > acq_size:
    ## model inference on random filtered samples
        args.task_name = args.task_name.lower()
        args.output_mode = output_modes[args.task_name]
        ori_dataset = get_glue_tensor_dataset(rand_inds, args, args.task_name, tokenizer, train=True)

        if model == None: # For initial selection
            bert_config = AutoConfig.from_pretrained(
                # args.config_name if args.config_name else args.model_name_or_path,
                'bert-base-cased',
                num_labels=args.num_classes,
                finetuning_task=args.task_name,
                cache_dir=args.cache_dir,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                # args.model_name_or_path,
                'bert-base-cased',
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=bert_config,
                cache_dir=args.cache_dir,
            )
        model.to(args.device)

        _, logits_rand, results = test_transformer_model(args, dataset=ori_dataset, model=model, return_cls=True)

        ## uncertain filter
        if args.Srand > acq_size*r and not init:
            uncertainty_scores = entropy(logits_rand)
            unc_inds = np.argpartition(uncertainty_scores, -acq_size*r)[-acq_size*r:]

            ## map filtered inds to original inds
            # cand_inds = np.array(rand_inds[unc_inds])
        else: # no filtering
            unc_inds = np.array(range(len(rand_inds)))

        ## clustering filter
        # if r != 1:
        if len(unc_inds) > acq_size:
            # cand_inds = np.array(rand_inds)[unc_inds]
            cand_cls = normalize(results[embs]).detach().cpu().numpy()[unc_inds]
            centers = kmeans(torch.tensor(cand_cls), acq_size)
            centers = np.array(centers)
            sampled_ind = list(np.array(rand_inds)[centers])
            # logits_cand = torch.tensor(logits_rand[centers])
        else: # no filtering
            sampled_ind = list(np.array(rand_inds)[unc_inds])
            # logits_cand = torch.tensor(logits_rand)
            
    ########### Calculating statistics of samples (not main function of acquisition) ##############
    y_lab = np.asarray(y_original, dtype='object')[labeled_inds]
    X_unlab = np.asarray(X_original, dtype='object')[dpool_inds]
    y_unlab = np.asarray(y_original, dtype='object')[dpool_inds]

    labels_list_previous = list(y_lab)
    c = collections.Counter(labels_list_previous)
    stats_list_previous = [(i, c[i] / len(labels_list_previous) * 100.0) for i in c]

    new_samples = np.asarray(X_original, dtype='object')[sampled_ind]
    new_labels = np.asarray(y_original, dtype='object')[sampled_ind]

    # Mean and std of length of selected sequences
    if args.task_name in ['sst-2', 'sst-2-imbalance', 'ag_news', 'dbpedia', 'trec-6', 'imdb', 'pubmed', 'sentiment', '20ng']: # single sequence
        l = [len(x.split()) for x in new_samples]
    elif args.dataset_name in ['mrpc', 'mnli', 'qnli', 'cola', 'rte', 'qqp','paws-qqp', 'nli']:
        l = [len(sentence[0].split()) + len(sentence[1].split()) for sentence in new_samples]  # pairs of sequences
    assert type(l) is list, "type l: {}, l: {}".format(type(l), l)
    length_mean = np.mean(l)
    length_std = np.std(l)
    length_min = np.min(l)
    length_max = np.max(l)

    # Percentages of each class
    labels_list_selected = list(np.array(y_original)[sampled_ind])
    c = collections.Counter(labels_list_selected)
    stats_list = [(i, c[i] / len(labels_list_selected) * 100.0) for i in c]

    labels_list_after = list(new_labels) + list(y_lab)
    c = collections.Counter(labels_list_after)
    stats_list_all = [(i, c[i] / len(labels_list_after) * 100.0) for i in c]

    assert len(set(sampled_ind)) == len(sampled_ind)
    assert bool(not set(sampled_ind) & set(labeled_inds))
    assert len(new_samples) == acq_size, 'len(new_samples)={}, annotatations_per_it={}'.format(
        len(new_samples),
        acq_size)

    stats = {'length': {'mean': float(length_mean),
                        'std': float(length_std),
                        'min': float(length_min),
                        'max': float(length_max)},
             'class_selected_samples': stats_list,
             'class_samples_after': stats_list_all,
             'class_samples_before': stats_list_previous,
             }

    return sampled_ind, stats


def kmeans_pp(X, k, centers, **kwargs): # initial sample selection
    def closest_center_dist(X, centers):
        # return distance to closest center
        dist = torch.cdist(X, X[centers])
        cd = dist.min(axis=1).values
        return cd
    # kmeans++ algorithm
    if len(centers) == 0:
        # randomly choose first center
        c1 = np.random.choice(X.size(0))
        centers.append(c1)
        k -= 1
    # greedily choose centers
    for i in tqdm(range(k)):
        dist = closest_center_dist(X, centers) ** 2
        prob = (dist / dist.sum()).cpu().detach().numpy()
        ci = np.random.choice(X.size(0), p=prob)
        centers.append(ci)
    return centers

def kmeans(X, k, **kwargs): # choose the closest samples to cluster centers
    # kmeans algorithm
    print("Running Kmeans")
    kmeans = KMeans(n_clusters=k, n_jobs=1).fit(X)
#     kmeans = KMeans(n_clusters=k, n_jobs=-1).fit(X)
    centers = kmeans.cluster_centers_
    # find closest point to centers
    centroids = cdist(centers, X).argmin(axis=1)
    centroids_set = np.unique(centroids)
    m = k - len(centroids_set)
    if m > 0:
        pool = np.delete(np.arange(len(X)), centroids_set)
        p = np.random.choice(len(pool), m)
        centroids = np.concatenate((centroids_set, pool[p]), axis = None)
    return centroids

def kmeans_large_clus(X, k, ru, **kwargs):
    kmeans = KMeans(n_clusters=k*ru, n_jobs=1).fit(X)
#     kmeans = KMeans(n_clusters=k, n_jobs=-1).fit(X)
    centers = kmeans.cluster_centers_
    clus_dic = collections.Counter(kmeans.labels_)
    dic2 = sorted(clus_dic.items(), key=lambda x:x[1], reverse=True) # sorting clusters by their sizes
    clus_ind = [x[0] for x in dic2[:k]]
    centers = centers[clus_ind]

    # find closest point to centers
    centroids = cdist(centers, X).argmin(axis=1)
    centroids_set = np.unique(centroids)
    m = k - len(centroids_set)
    if m > 0:
        pool = np.delete(np.arange(len(X)), centroids_set)
        p = np.random.choice(len(pool), m)
        centroids = np.concatenate((centroids_set, pool[p]), axis = None)
    return centroids

