import json
import logging
import math
import os
import pickle
import sys
import time
import copy

import numpy as np
import torch
from torch.nn.functional import normalize
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from transformers import set_seed, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

sys.path.append("../")
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from proposed_acquisition.new_acquisition import kmeans_pp, tyrogue
from acquisition.cal import contrastive_acquisition
from acquisition.uncertainty import select_alps, calculate_uncertainty
from utilities.preprocessors import output_modes

from utilities.data_loader import get_glue_dataset, get_glue_tensor_dataset
from utilities.trainers import test_transformer_model, train_transformer_model, my_evaluate
from utilities.general_fixedSrand import create_dir, print_stats, create_exp_dirs

from sys_config import acquisition_functions, CACHE_DIR, DATA_DIR, CKPT_DIR

proposal = [
    'tyrogue',
]

logger = logging.getLogger(__name__)


def al_loop(args):
    """
    Main script for the active learning algorithm.
    :param args: contains necessary arguments for model, training, data and AL settings
    Datasets (lists): X_train_original, y_train_original, X_val, y_val
    Indices (lists): X_train_init_inds : inds of first training set (iteration 1)
                     X_train_current_inds : inds of labeled dataset (iteration i)
                     X_train_remaining_inds : inds of unlabeled dataset (iteration i)
                     X_train_original_inds : inds of (full) original training set
    """
    #############
    # Setup
    #############
    # Set the random seed manually for reproducibility.
    set_seed(args.seed)
    ##############################################################
    # Load data
    ##############################################################
    X_test_ood = None
    X_train_original, y_train_original = get_glue_dataset(args,
                                                          args.data_dir,
                                                          args.task_name,
                                                          args.model_type,
                                                          evaluate=False)
    X_val, y_val = get_glue_dataset(args,
                                    args.data_dir,
                                    args.task_name,
                                    args.model_type,
                                    evaluate=True)
    X_test, y_test = get_glue_dataset(args,
                                      args.data_dir,
                                      args.task_name,
                                      args.model_type,
                                      test=True)

    if args.task_name == 'imdb':
        X_test_ood, y_test_ood = get_glue_dataset(args,
                                                  os.path.join(
                                                      DATA_DIR, 'SST-2'),
                                                  'sst-2',
                                                  args.model_type,
                                                  test=True)
    if args.task_name == 'sst-2':
        X_test_ood, y_test_ood = get_glue_dataset(args,
                                                  os.path.join(
                                                      DATA_DIR, 'IMDB'),
                                                  'imdb',
                                                  args.model_type,
                                                  test=True)
    if args.task_name == 'qqp':
        X_test_ood, y_test_ood = get_glue_dataset(args,
                                                  os.path.join(
                                                      DATA_DIR, 'TwitterPPDB'),
                                                  'twitterppdb',
                                                  args.model_type,
                                                  test=True)

    X_train_original_inds = list(np.arange(
        len(X_train_original)))  # [:2500]  # original pool
    X_val_inds = list(np.arange(len(X_val)))
    X_test_inds = list(np.arange(len(X_test)))

    print("data spliting")
    if args.dataset_name in ['dbpedia']:
        # undersample dpool up to 20K + dval up to 2K
        new_X_train_original_inds, X_train_discarded_inds, _, _ = train_test_split(
            X_train_original_inds,
            y_train_original,
            train_size=20000,
            random_state=42,
            stratify=y_train_original)
        new_X_val_inds, X_val_discarded_inds, _, _ = train_test_split(
            X_val_inds,
            y_val,
            train_size=2000,
            random_state=42,
            stratify=y_val)
        X_train_original_inds = new_X_train_original_inds
        X_val_inds = new_X_val_inds
    args.binary = True if len(
        set(np.array(y_train_original)[X_train_original_inds])) == 2 else False
    args.num_classes = len(
        set(np.array(y_train_original)[X_train_original_inds]))

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name
        if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
    )

    ##############################################################
    # Stats
    ##############################################################
    print()
    print_stats(np.array(y_train_original)[X_train_original_inds], 'train')
    print_stats(np.array(y_val)[X_val_inds], 'validation')
    print_stats(np.array(y_test)[X_test_inds], 'test')

    print("\nDataset for annotation: {}\nAcquisition function: {}\n"
          "Budget: {}% of labeled data\n".format(
              args.dataset_name, args.acquisition,
              args.budget / len(X_train_original_inds) * 100))
    print(len(X_train_original_inds))

    ##############################################################
    # Experiment dir
    ##############################################################
    results_per_iteration = {}

    results_dir = create_exp_dirs(args)
    resume_dir = results_dir

    ##############################################################
    # Get BERT representations
    ##############################################################
    bert_representations = None
    if (args.bert_rep or "naive" in args.acquisition
            or args.acquisition.split("_")[-1]
            == 'kmeans') and args.acquisition != "random_naive":
        # if (args.bert_rep or "naive" in args.acquisition or args.acquisition.split("_")[-1] == 'kmeans' or args.acquisition=='rand_clus_unc') and args.acquisition != "random_naive":
        # if args.bert_rep or args.acquisition in proposal or args.acquisition.split("_")[-1] == 'kmeans':
        # if os.path.isfile(os.path.join(args.data_dir, "bert_representations.pkl")):
        #     print('Load bert representations...')
        #     with open(os.path.join(args.data_dir, "bert_representations.pkl"), 'rb') as handle:
        #         bert_representations = pickle.load(handle)
        #         assert bert_representations.shape[0] == len(X_train_original_inds)
        # else:
        args.task_name = args.task_name.lower()
        args.output_mode = output_modes[args.task_name]
        ori_dataset = get_glue_tensor_dataset(X_train_original_inds,
                                              args,
                                              args.task_name,
                                              tokenizer,
                                              train=True)
        bert_config = AutoConfig.from_pretrained(
            # args.config_name if args.config_name else args.model_name_or_path,
            'bert-base-cased',
            num_labels=args.num_classes,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir,
        )
        bert_tokenizer = AutoTokenizer.from_pretrained(
            # args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            'bert-base-cased',
            cache_dir=args.cache_dir,
            use_fast=args.use_fast_tokenizer,
        )
        bert_model = AutoModelForSequenceClassification.from_pretrained(
            # args.model_name_or_path,
            'bert-base-cased',
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=bert_config,
            cache_dir=args.cache_dir,
        )
        bert_model.to(args.device)
        # eval_loss, logits, result
        _, _logits, _results = test_transformer_model(args,
                                                      dataset=ori_dataset,
                                                      model=bert_model,
                                                      return_cls=True)
        bert_representations = _results["bert_cls"]
        assert bert_representations.shape[0] == len(X_train_original_inds)

        # with open(os.path.join(args.data_dir, "bert_representations.pkl"), 'wb') as handle:
        #     pickle.dump(bert_representations, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ##############################################################
    # Get TFIDF representations
    ##############################################################
    tfidf_representations = None
    if args.tfidf:
        if os.path.isfile(
                os.path.join(args.data_dir, "tfidf_representations.pkl")):
            print('Load tfidf representations...')
            with open(os.path.join(args.data_dir, "tfidf_representations.pkl"),
                      'rb') as handle:
                tfidf_representations = pickle.load(handle)
                assert len(tfidf_representations) == len(X_train_original_inds)
        else:
            vectorizer = TfidfVectorizer(
                max_features=15000,
                lowercase=True,
                stop_words=feature_extraction.text.ENGLISH_STOP_WORDS)
            if type(X_train_original[0]) is list or type(
                    X_train_original[0]) is tuple:
                vectors = vectorizer.fit_transform([
                    s[0] + ' ' + s[1]
                    for s in np.array(X_train_original)[X_train_original_inds]
                ])
            else:
                vectors = vectorizer.fit_transform([
                    s
                    for s in np.array(X_train_original)[X_train_original_inds]
                ])
            feature_names = vectorizer.get_feature_names()
            dense = vectors.todense()
            denselist = dense.tolist()
            tfidf_representations = denselist
            # tfidf_representations = torch.tensor(denselist)
            assert len(tfidf_representations) == len(X_train_original_inds)
            with open(os.path.join(args.data_dir, "tfidf_representations.pkl"),
                      'wb') as handle:
                pickle.dump(tfidf_representations,
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

    ##############################################################
    # Start
    ##############################################################

    ids_per_it = {}

    ##############################################################
    # Select first training data
    ##############################################################
    y_strat = np.array(y_train_original)[X_train_original_inds]

    X_train_original_after_sampling_inds = []
    X_train_original_after_sampling = []

    if args.acquisition == 'tyrogue':
        args.init = 'tyrogue'
    if args.acquisition == 'alps':
        args.init = 'alps'
    if args.init == 'oracle_ratio':  # using oracle class size distribution
        X_train_init_inds, X_train_remaining_inds, _, _ = train_test_split(
            X_train_original_inds,
            np.array(y_train_original)[X_train_original_inds],
            # train_size=init_train_percent / 100,
            train_size=args.init_train_data,
            random_state=args.seed,
            stratify=y_strat)
    elif args.init == 'naive':
        print("Initial sampleing by a naive way")
        X_train_init_inds, _ = naive_margin(
            args=args,
            acq_size=args.init_train_data,
            X_original=X_train_original,
            y_original=y_train_original,
            labeled_inds=[],
            dpool_inds=X_train_original_inds,
            discarded_inds=[],
            original_inds=X_train_original_inds,
            results_dpool=_results,
            logits_dpool=_logits,
            bert_representations=bert_representations,
            candidate_scale=args.candidate_scale)
        X_train_remaining_inds = [
            x for x in X_train_original_inds if x not in X_train_init_inds
        ]
    elif args.init == 'random':
        X_train_init_inds, X_train_remaining_inds, _, _ = train_test_split(
            X_train_original_inds,
            np.array(y_train_original)[X_train_original_inds],
            # train_size=init_train_percent / 100,
            train_size=args.init_train_data,
            random_state=args.seed)
        # stratify=y_strat)
    elif args.init == 'alps':
        X_train_init_inds = select_alps(args,
                                        sampled=[],
                                        acquisition_size=args.init_train_data,
                                        original_inds=X_train_original_inds)
        X_train_remaining_inds = [
            x for x in X_train_original_inds if x not in X_train_init_inds
        ]
    elif args.init == 'tyrogue':
        X_train_init_inds, _ = tyrogue(
            args=args,
            acq_size=args.init_train_data,
            X_original=X_train_original,
            y_original=y_train_original,
            labeled_inds=[],
            dpool_inds=X_train_original_inds,
            discarded_inds=[],
            original_inds=X_train_original_inds,
            # results_dpool=_results,
            # logits_dpool=_logits,
            # bert_representations=bert_representations,
            candidate_scale=args.candidate_scale,
            init=True,
        )
        X_train_remaining_inds = [
            x for x in X_train_original_inds if x not in X_train_init_inds
        ]

    else:
        print(args.init)
        raise NotImplementedError

    ####################################################################
    # Create Dpool and Dlabels
    ####################################################################
    # print(X_train_init_inds)
    X_train_init = list(
        np.asarray(X_train_original, dtype='object')[X_train_init_inds])
    y_train_init = list(
        np.asarray(y_train_original, dtype='object')[X_train_init_inds])

    for i in list(set(y_train_init)):
        init_train_dist_class = 100 * np.sum(
            np.array(y_train_init) == i) / len(y_train_init)
        print('init % class {}: {}'.format(i, init_train_dist_class))

    if X_train_original_after_sampling_inds == []:
        assert len(X_train_init_inds) + len(X_train_remaining_inds) == len(
            X_train_original_inds
        ), 'init {}, remaining {}, original {}'.format(
            len(X_train_init_inds), len(X_train_remaining_inds),
            len(X_train_original_inds))
    else:
        assert len(X_train_init_inds) + len(X_train_remaining_inds) == len(
            X_train_original_after_sampling_inds)

    ids_per_it.update({str(0): list(map(int, X_train_init_inds))})
    assert len(ids_per_it[str(0)]) == args.init_train_data

    ####################################################################
    # Annotations & budget
    ####################################################################
    current_annotations = len(X_train_init)  # without validation data
    total_annotations = args.budget
    annotations_per_iteration = args.acquisition_size
    total_iterations = math.ceil(
        total_annotations / annotations_per_iteration) + 1

    print("TOTAL ITERATIONS {}".format(total_iterations))
    X_train_current_inds = X_train_init_inds.copy()

    X_discarded_inds = [
        x for x in X_train_original_inds
        if x not in X_train_remaining_inds and x not in X_train_current_inds
    ]

    it2per = {}  # iterations to data percentage
    val_acc_previous = None
    args.acc_best_iteration = 0
    args.acc_best = 0
    current_iteration = 1

    assert bool(not set(X_train_remaining_inds) & set(X_train_current_inds))
    """
        Indices of X_train_original: X_train_init_inds - inds of first training set (iteration 1)
                                     X_train_current_inds - inds of labeled dataset (iteration i)
                                     X_train_remaining_inds - inds of unlabeled dataset (iteration i)
                                     X_train_original_inds - inds of (full) original training set
                                     X_disgarded_inds - inds from dpool that are disgarded

    """

    #############
    # Start AL!
    #############
    X_history_inds = dict()
    train_results = dict()
    train_results['model'] = None
    while current_iteration < total_iterations + 1:
        for x in X_train_current_inds:
            if x in X_history_inds and args.weighting_sample:
                X_history_inds[x] /= 2
            else:
                X_history_inds[x] = 1

        start = time.time()
        it2per[str(current_iteration)] = round(
            len(X_train_current_inds) / len(X_train_original_inds), 2) * 100

        ##############################################################
        # Train model on training dataset (Dtrain)
        ##############################################################
        print("\n Start Training model of iteration {}!\n".format(
            current_iteration))
        train_results = train_transformer_model(
            args,
            X_train_current_inds,
            X_val_inds,
            X_history_inds,
            model=train_results['model'],
            iteration=current_iteration,
            val_acc_previous=val_acc_previous,
        )
        val_acc_previous = train_results['acc']
        # previous_model = copy.deepcopy(train_results['model'])
        end = time.time()
        training_time = end - start
        print("\nDone Training!\n")

        ##############################################################
        # Test model on test data (D_test)
        ##############################################################
        print("\nStart Testing on test set!\n")
        with torch.no_grad():
            test_dataset = get_glue_tensor_dataset(None,
                                                   args,
                                                   args.task_name,
                                                   tokenizer,
                                                   test=True)
            test_results, test_logits = my_evaluate(test_dataset,
                                                    args,
                                                    train_results['model'],
                                                    prefix="",
                                                    al_test=False,
                                                    mc_samples=None)
            test_results.pop('gold_labels', None)

        ##############################################################
        # Test model on OOD test data (D_ood)
        ##############################################################
        print("\nEvaluating robustness! Start testing on OOD test set!\n")
        # if False:
        if X_test_ood is not None:
            if args.dataset_name == 'sst-2':
                ood_test_dataset = get_glue_tensor_dataset(
                    None,
                    args,
                    'imdb',
                    tokenizer,
                    test=True,
                    data_dir=os.path.join(DATA_DIR, 'IMDB'))
            elif args.dataset_name == 'imdb':
                ood_test_dataset = get_glue_tensor_dataset(
                    None,
                    args,
                    'sst-2',
                    tokenizer,
                    test=True,
                    data_dir=os.path.join(DATA_DIR, 'SST-2'))
            elif args.dataset_name == 'qqp':
                ood_test_dataset = get_glue_tensor_dataset(
                    None,
                    args,
                    'twitterppdb',
                    tokenizer,
                    test=True,
                    data_dir=os.path.join(DATA_DIR, 'TwitterPPDB'))
            else:
                ood_test_dataset = get_glue_tensor_dataset(None,
                                                           args,
                                                           args.task_name,
                                                           tokenizer,
                                                           test=True,
                                                           ood=True)
            ood_test_results, ood_test_logits = my_evaluate(
                ood_test_dataset,
                args,
                train_results['model'],
                prefix="",
                al_test=False,
                mc_samples=None)
            ood_test_results.pop('gold_labels', None)

        ##############################################################
        # Test model on unlabeled data (Dpool)
        ##############################################################
        print("\nEvaluating Dpool!\n")
        start = time.time()
        dpool_loss, logits_dpool, results_dpool = [], [], []
        # if args.acquisition not in ['random', 'alps', 'badge', 'FTbertKM'] or (args.acquisition in ['random_naive', 'random_entropy'] and current_iteration >= args.switch):
        if args.acquisition in [
                'cal',
                'entropy',
        ]:
            with torch.no_grad():
                dpool_loss, logits_dpool, results_dpool = test_transformer_model(
                    args,
                    X_train_remaining_inds,
                    model=train_results['model'],
                    return_mean_embs=args.mean_embs,
                    return_mean_output=args.mean_out,
                    return_cls=args.cls)

                results_dpool.pop('gold_labels', None)
        end = time.time()
        inference_time = end - start

        ##############################################################
        # Select unlabeled samples for annotation
        # -> annotate
        # -> update training dataset & unlabeled dataset
        ##############################################################
        # I moved this in the end!
        # if total_annotations - current_annotations < annotations_per_iteration:
        #     annotations_per_iteration = total_annotations - current_annotations
        #
        # if annotations_per_iteration == 0:
        #     break

        assert len(set(X_train_current_inds)) == len(X_train_current_inds)
        assert len(set(X_train_remaining_inds)) == len(X_train_remaining_inds)

        start = time.time()
        sample_flag = True
        if args.acquisition in ["cal"]:
            if args.tfidf:
                tfidf_dtrain_reprs = torch.tensor(
                    list(
                        np.array(tfidf_representations)[X_train_current_inds]))
                tfidf_dpool_reprs = torch.tensor(
                    list(
                        np.array(tfidf_representations)
                        [X_train_remaining_inds]))
            else:
                tfidf_dtrain_reprs = None
                tfidf_dpool_reprs = None
            sampled_ind, stats = contrastive_acquisition(
                args=args,
                annotations_per_iteration=annotations_per_iteration,
                X_original=X_train_original,
                y_original=y_train_original,
                labeled_inds=X_train_current_inds,
                candidate_inds=X_train_remaining_inds,
                discarded_inds=X_discarded_inds,
                original_inds=X_train_original_inds,
                tokenizer=tokenizer,
                train_results=train_results,
                results_dpool=results_dpool,
                logits_dpool=logits_dpool,
                bert_representations=bert_representations,
                tfidf_dtrain_reprs=tfidf_dtrain_reprs,
                tfidf_dpool_reprs=tfidf_dpool_reprs)
            

        elif args.acquisition == 'tyrogue':
            sampled_ind, stats = tyrogue(
                args=args,
                acq_size=annotations_per_iteration,
                X_original=X_train_original,
                y_original=y_train_original,
                labeled_inds=X_train_current_inds,
                dpool_inds=X_train_remaining_inds,
                discarded_inds=X_discarded_inds,
                original_inds=X_train_original_inds,
                tokenizer=tokenizer,
                # results_dpool=results_dpool,
                # logits_dpool=logits_dpool,
                model=train_results['model'],
                # bert_representations=bert_representations,
                candidate_scale=args.candidate_scale)
        else:
            sampled_ind, stats = calculate_uncertainty(
                args=args,
                method=args.acquisition,
                # method=args.acquisition.split("_")[0],
                logits=logits_dpool,
                annotations_per_it=annotations_per_iteration,
                device=args.device,
                iteration=current_iteration,
                task=args.task_name,
                representations=None,
                candidate_inds=X_train_remaining_inds,
                labeled_inds=X_train_current_inds,
                discarded_inds=X_discarded_inds,
                original_inds=X_train_original_inds,
                model=train_results['model'],
                X_original=X_train_original,
                y_original=y_train_original)
        end = time.time()
        selection_time = end - start

        # Update results dict
        results_per_iteration[str(current_iteration)] = {
            'data_percent': it2per[str(current_iteration)],
            'total_train_samples': len(X_train_current_inds),
            'training_time': training_time,
            'inference_time': inference_time,
            'selection_time': selection_time
        }
        results_per_iteration[str(
            current_iteration)]['val_results'] = copy.deepcopy(train_results)
        results_per_iteration[str(
            current_iteration)]['test_results'] = test_results

        if X_test_ood is not None:
            results_per_iteration[str(
                current_iteration)]['ood_test_results'] = ood_test_results
            results_per_iteration[str(
                current_iteration)]['ood_test_results'].pop('model', None)

        results_per_iteration[str(current_iteration)]['val_results'].pop(
            'model', None)
        results_per_iteration[str(current_iteration)]['test_results'].pop(
            'model', None)
        results_per_iteration[str(current_iteration)].update(stats)

        current_annotations += annotations_per_iteration

        # X_train_current_inds and X_train_remaining_inds are lists of indices of the original dataset
        # sampled_inds is a list of indices OF THE X_train_remaining_inds(!!!!) LIST THAT SHOULD BE REMOVED
        if args.acquisition in [
                'alps', 'badge', 'FTbertKM', 'cal', 'contrastive'
        ] + proposal and sample_flag:
            X_train_current_inds += list(sampled_ind)
        else:
            X_train_current_inds += list(
                np.array(X_train_remaining_inds)[sampled_ind])

        assert len(ids_per_it[str(0)]) == args.init_train_data

        if args.acquisition in [
                'alps', 'badge', 'FTbertKM', 'cal', 'contrastive'
        ] + proposal and sample_flag:
            selected_dataset_ids = sampled_ind
            selected_dataset_ids = list(map(int,
                                            selected_dataset_ids))  # for json
            assert len(ids_per_it[str(0)]) == args.init_train_data
        else:
            selected_dataset_ids = list(
                np.array(X_train_remaining_inds)[sampled_ind])
            selected_dataset_ids = list(map(int,
                                            selected_dataset_ids))  # for json
            assert len(ids_per_it[str(0)]) == args.init_train_data

        ids_per_it.update({str(current_iteration): selected_dataset_ids})

        assert len(ids_per_it[str(0)]) == args.init_train_data
        assert len(
            ids_per_it[str(current_iteration)]) == annotations_per_iteration

        if args.acquisition in [
                'alps', 'badge', 'FTbertKM', 'cal', 'contrastive'
        ] + proposal and sample_flag:
            X_train_remaining_inds = [
                x for x in X_train_original_inds
                if x not in X_train_current_inds and x not in X_discarded_inds
            ]
        else:
            X_train_remaining_inds = list(
                np.delete(X_train_remaining_inds, sampled_ind))

        # Assert no common data in Dlab and Dpool
        assert bool(not set(X_train_current_inds)
                    & set(X_train_remaining_inds))

        # Assert unique (no duplicate) inds in Dlab & Dpool
        assert len(set(X_train_current_inds)) == len(X_train_current_inds)
        assert len(set(X_train_remaining_inds)) == len(X_train_remaining_inds)

        # Assert each list of inds unique
        set(X_train_original_inds).difference(set(X_train_current_inds))
        if args.indicator is None and args.indicator != "small_config":
            assert set(X_train_original_inds).difference(
                set(X_train_current_inds)) == set(X_train_remaining_inds +
                                                  X_discarded_inds)

        results_per_iteration['last_iteration'] = current_iteration
        results_per_iteration['current_annotations'] = current_annotations
        results_per_iteration[
            'annotations_per_iteration'] = annotations_per_iteration
        results_per_iteration['X_val_inds'] = list(map(int, X_val_inds))

        print("\n")
        print("*" * 12)
        print("End of iteration {}:".format(current_iteration))
        if 'loss' in test_results.keys():
            print("Train loss {}, Val loss {}, Test loss {}".format(
                train_results['train_loss'], train_results['loss'],
                test_results['loss']))
        print("Annotated {} samples".format(annotations_per_iteration))
        print("Current labeled (training) data: {} samples".format(
            len(X_train_current_inds)))
        print("Remaining budget: {} (in samples)".format(total_annotations -
                                                         current_annotations))
        print("*" * 12)
        print()

        tmp_res = results_per_iteration[str(current_iteration)]
        current_iteration += 1

        print('Saving json with the results to {}'.format(results_dir))

        with open(os.path.join(results_dir, 'results_of_iteration.json'),
                  'w') as f:
            json.dump(results_per_iteration, f)
        with open(os.path.join(results_dir, 'selected_ids_per_iteration.json'),
                  'w') as f:
            json.dump(ids_per_it, f)

        # Check budget
        if total_annotations - current_annotations < annotations_per_iteration:
            annotations_per_iteration = total_annotations - current_annotations

        if annotations_per_iteration == 0:
            break
    print('The end!....')

    return


def run_main(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device(
            "cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
        args.n_gpu = 0 if args.no_cuda else 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    print('device: {}'.format(args.device))

    # Setup args
    # if args.seed == None:
    #     seed = random.randint(1, 9999)
    #     args.seed = seed
    args.task_name = args.dataset_name.upper()

    args.cache_dir = CACHE_DIR
    args.data_dir = os.path.join(DATA_DIR, args.task_name)

    args.overwrite_cache = bool(True)
    args.evaluate_during_training = True

    # Output dir
    ckpt_dir = os.path.join(
        CKPT_DIR, '{}_{}_{}_{}'.format(args.dataset_name, args.model_type,
                                       args.acquisition, args.seed))
    args.output_dir = os.path.join(
        ckpt_dir, '{}_{}'.format(args.dataset_name, args.model_type))
    if args.model_type == 'allenai/scibert':
        args.output_dir = os.path.join(
            ckpt_dir, '{}_{}'.format(args.dataset_name, 'bert'))

    if args.indicator is not None:
        args.output_dir += '-{}'.format(args.indicator)
    # The following arguments are experiments in the ablation/analysis section of the paper
    if args.reverse: args.output_dir += '-reverse'
    if args.mean_embs: args.output_dir += '-inputs'
    if args.mean_out: args.output_dir += '-outputs'
    if args.cls: args.output_dir += '-cls'
    if args.ce: args.output_dir += '-ce'
    if args.operator != "mean" and args.acquisition == "adv_train":
        args.output_dir += '-{}'.format(args.operator)
    if args.knn_lab: args.output_dir += '-lab'
    if args.bert_score: args.output_dir += '-bs'
    if args.bert_rep: args.output_dir += '-br'
    if args.tfidf: args.output_dir += '-tfidf'

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    args.task_name = args.task_name.lower()

    al_loop(args)
