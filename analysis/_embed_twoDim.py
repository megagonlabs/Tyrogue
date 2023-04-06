import json
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.distributions import Categorical
from torch.utils.data import Subset, SequentialSampler, DataLoader
from torch.nn.functional import normalize
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from sklearn.manifold import TSNE

dim_path = "twoDim_embeddings/"

sys.path.append("../")
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sys_config import CACHE_DIR, DATA_DIR, EXP_DIR, ANALYSIS_DIR
from utilities.preprocessors import processors, output_modes
from utilities.general_fixedSrand import create_dir
from utilities.data_loader import get_glue_dataset, get_glue_tensor_dataset
from utilities.trainers_freezing import train_transformer

logger = logging.getLogger(__name__)


def load_json(filename):
    file = None
    if os.path.isfile(filename):
        with open(filename) as json_file:
            if 'jsonl' in filename:
                json_list = list(json_file)
                file_list = []
                for json_str in json_list:
                    file_list.append(json.loads(json_str))
                    # print(f"result: {result}")
                    # print(isinstance(result, dict))
                file = file_list
            else:
                file = json.load(json_file)
    return file


def compute_entropy(sampled_dataset, model, args):
    """Compute average entropy in label distribution for examples in [sampled]."""
    all_entropy = None
    # data = Subset(dataset, sampled)
    sampler = SequentialSampler(sampled_dataset)
    dataloader = DataLoader(sampled_dataset, sampler=sampler, batch_size=args.per_gpu_eval_batch_size)
    for batch in tqdm(dataloader, desc="Computing entropy"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            logits = outputs[0]
            categorical = Categorical(logits=logits)
            entropy = categorical.entropy()
        if all_entropy is None:
            all_entropy = entropy.detach().cpu().numpy()
        else:
            all_entropy = np.append(all_entropy, entropy.detach().cpu().numpy(), axis=0)
    avg_entropy = all_entropy.mean()
    return avg_entropy

def embed_BERT(X_inds, args, model, tokenizer):
    samples = get_glue_tensor_dataset(X_inds, args, args.task_name, tokenizer, train=True)
    sampler = SequentialSampler(samples)
    dataloader = DataLoader(samples, sampler=sampler, batch_size=args.per_gpu_eval_batch_size)
    cls = []
    for batch in tqdm(dataloader, desc="Computing feature diversity"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            # print(inputs)
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                bert_output, bert_output_cls = model.bert(**inputs)
                cls.append(bert_output_cls)
    cls = torch.cat(cls,0)
    cls = normalize(cls).detach().cpu().numpy()
    return cls

def token_set(data):
    all_tokens = set()
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=32)
    for batch in tqdm(dataloader, desc="Getting tokens"):
        with torch.no_grad():
            tokens = batch[0].unique().tolist()
            all_tokens = all_tokens.union(tokens)
    return all_tokens


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    ##########################################################################
    # Setup args
    ##########################################################################
    parser.add_argument("--local_rank",
                        type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    ##########################################################################
    # Model args
    ##########################################################################
    parser.add_argument("--model_type", default="bert", type=str, help="Pretrained model")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str, help="Pretrained ckpt")
    ##parser.add_argument(
    ##    "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    ##)
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        default=True,
        type=bool,
        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true",
        default=False,
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument("--tapt", default=None, type=str,
                        help="ckpt of tapt model")
    ##########################################################################
    # Training args
    ##########################################################################
    parser.add_argument("--do_train", default=True, type=bool, help="If true do train")
    parser.add_argument("--do_eval", default=True, type=bool, help="If true do evaluation")
    parser.add_argument("--overwrite_output_dir", default=False, type=bool, help="If true do evaluation")
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4096, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("-seed", "--seed", default=42, required=False, type=int, help="seed")
    parser.add_argument("--indicator", required=False,
                        default=None,
                        type=str,
                        help="experiment indicator")
    parser.add_argument("-patience", "--patience", required=False, type=int, help="patience for early stopping (steps)")
    parser.add_argument("-test_uncertainty", "--test_uncertainty", required=False, type=bool, default=True,
                        help=" whether to evaluate uncertainty estimates for [vanilla, mc_3, mc_5, mc_10]")
    ##########################################################################
    # Data args
    ##########################################################################
    parser.add_argument("--dataset_name", default=None, required=True, type=str,
                        help="Dataset [mrpc, ag_news, qnli, sst-2, trec-6]")
    parser.add_argument("--acquisition", default='cal', type=str)
    parser.add_argument("--acquisition_size", default=100, type=int)
    parser.add_argument("--task_name", default=None, type=str, help="Task [MRPC, AG_NEWS, QNLI, SST-2]")
    parser.add_argument("--max_seq_length", default=256, type=int, help="Max sequence length")
    parser.add_argument("--counterfactual", default=None, type=str, help="Max sequence length")
    # parser.add_argument("--ood_name", default=None, type=str, help="name of ood dataset for counterfactual data: [amazon, yelp, semeval]")
    ##########################################################################
    # Data augmentation args
    ##########################################################################
    parser.add_argument("--da", default=None, type=str, help="Apply DA method: [ssmba]")
    parser.add_argument("--da_set", default="lab", type=str, help="if lab augment labeled set else unlabeled")
    parser.add_argument("--num_per_augm", default=1, type=int, help="number of augmentations per example")
    parser.add_argument("--num_augm", default=100, type=int, help="number of examples to augment")
    parser.add_argument("--da_all", default=False, type=bool, help="if True augment the entire dataset")
    parser.add_argument("--uda", default=False, type=bool, help="if true consistency loss, else supervised learning")
    parser.add_argument("--uda_confidence_thresh", default=0.5, type=float, help="confidence threshold")
    parser.add_argument("--uda_softmax_temp", default=0.4, type=float, help="temperature to sharpen predictions")
    parser.add_argument("--uda_coeff", default=1, type=float, help="lambda value (weight) of KL loss")
    ##########################################################################
    # Server args
    ##########################################################################
    parser.add_argument("-g", "--gpu", required=False,
                        default='0', help="gpu on which this experiment runs")
    parser.add_argument("-server", "--server", required=False,
                        default='ford', help="server on which this experiment runs")

    args = parser.parse_args()
    # args.exp_path = 'new_acquisition'

    # Setup
    if args.server is 'ford':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print("\nThis experiment runs on gpu {}...\n".format(args.gpu))
        # VIS['enabled'] = True
        args.n_gpu = 1
        args.device = torch.device('cuda:{}'.format(args.gpu))
    else:
        # Setup CUDA, GPU & distributed training
        if args.local_rank == -1 or args.no_cuda:
            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    if args.task_name is None: args.task_name = args.dataset_name.upper()

    args.cache_dir = CACHE_DIR
    args.data_dir = os.path.join(DATA_DIR, args.task_name)

    args.overwrite_cache = bool(True)
    args.evaluate_during_training = True

    # Output dir
    ##CKPT_DIR = "/nfs/shared/labeler_al/_new_acquisition/full_model_dir"
    CKPT_DIR = "/nfs/shared/labeler_al/low-budget-AL/main/full_model_dir"
    # CKPT_DIR = "./full_model_dir"
    ckpt_dir = os.path.join(CKPT_DIR,
                            '{}_{}_{}_{}'.format(args.dataset_name, args.model_type, args.acquisition, args.seed))
    output_dir = os.path.join(ckpt_dir, '{}_{}'.format(args.dataset_name, args.model_type))
    if args.model_type == 'allenai/scibert':
        args.output_dir = os.path.join(ckpt_dir, '{}_{}'.format(args.dataset_name, 'bert'),
                                       '{}_{}'.format(args.dataset_name, 'bert'))
    args.output_dir = os.path.join(output_dir, 'all_{}'.format(args.seed))

    if args.indicator is not None: args.output_dir += '-{}'.format(args.indicator)
    args.current_output_dir = args.output_dir

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

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    print("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
    )

    print("load dataset")
    # Load (raw) dataset
    X_train, y_train = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type, evaluate=False)
    X_val, y_val = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type, evaluate=True)
    X_test, y_test = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type, test=True)

    X_orig = X_train  # original train set
    y_orig = y_train  # original labels
    X_inds = list(np.arange(len(X_orig)))  # indices to original train set
    X_unlab_inds = []  # indices of ulabeled set to original train set

    args.binary = True if len(set(y_train)) == 2 else False
    args.num_classes = len(set(y_train))

    #######################
    # Datasets
    #######################
    augm_dataset = None

    train_dataset = get_glue_tensor_dataset(X_inds, args, args.task_name, tokenizer, train=True)
    assert len(train_dataset) == len(X_inds)
    # eval_dataset = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, evaluate=True)
    # test_dataset = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, test=True)

    #######################
    # Load fully trained model
    #######################
    # if len(os.listdir(args.output_dir)) != 0:
    print("loading BERT, path: {}".format(args.output_dir))
    logger.warning("loading BERT, path: {}".format(args.output_dir))
    model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
    print("successfully loaded!")
    logger.warning("successfully loaded!")
    model.to(args.device)
    
    embedding_output_path = os.path.join(ANALYSIS_DIR, dim_path)
    
    if not os.path.exists(embedding_output_path):
        os.makedirs(embedding_output_path)

    datasets = [args.dataset_name]
    for dataset in datasets:
        # embed all data points by BERT model with full-supervision
        all_cls = embed_BERT(X_inds, args, model, tokenizer)

        tsne = TSNE(n_components=2, random_state = 0, perplexity = 30, n_iter = 1000)
        X_embedded = tsne.fit_transform(all_cls)
        df = pd.DataFrame(X_embedded, columns = ['col1', 'col2'])
        df.to_csv(os.path.join(embedding_output_path, '{}.csv'.format(dataset)))

    print("saved two dimensional embeddings!")
