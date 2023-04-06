from al_loop_fixed import run_main
import warnings

warnings.simplefilter('ignore')
import argparse
import random
from sys_config import acquisition_functions, CACHE_DIR, DATA_DIR, CKPT_DIR

proposal = [
    "tyrogue"
]

parser = argparse.ArgumentParser()

####### added ########
parser.add_argument("--uncertainty",
                    type=str,
                    default='entropy',
                    choices=['margin_conf', 'entropy'],
                    help="Uncertainty criteria for acquisition")
###############

parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="For distributed training: local_rank")
parser.add_argument("--no_cuda",
                    action="store_true",
                    help="Avoid using CUDA when available")
parser.add_argument(
    "--fp16",
    action="store_true",
    help=
    "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
)
parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help=
    "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    "See details at https://nvidia.github.io/apex/amp.html",
)
##########################################################################
# Model args
##########################################################################
parser.add_argument("--model_type",
                    default="bert",
                    type=str,
                    help="Pretrained model")
parser.add_argument("--model_name_or_path",
                    default="bert-base-cased",
                    type=str,
                    help="Pretrained ckpt")
parser.add_argument(
    "--config_name",
    default="",
    type=str,
    help="Pretrained config name or path if not the same as model_name",
)
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
    help=
    "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.",
)
parser.add_argument(
    "--do_lower_case",
    action="store_true",
    default=False,
    help="Set this flag if you are using an uncased model.",
)
##########################################################################
# Training args
##########################################################################
## for new our method
parser.add_argument("--r", type=int, required=False, default=3)
parser.add_argument("--Srand", type=int, required=False, default=10000)
parser.add_argument("--eval_log", type=bool, required=False, default=False)

##########################################################################
parser.add_argument("--do_train",
                    default=True,
                    type=bool,
                    help="If true do train")
parser.add_argument("--do_eval",
                    default=True,
                    type=bool,
                    help="If true do evaluation")
parser.add_argument("--overwrite_output_dir",
                    default=True,
                    type=bool,
                    help="If true do evaluation")
parser.add_argument("--per_gpu_train_batch_size",
                    default=16,
                    type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size",
                    # default=4096,
                    default=2048,
                    type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help=
    "Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--num_train_epochs",
    default=3,
    type=float,
    help="Total number of training epochs to perform.",
)
parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help=
    "If > 0: set total number of training steps to perform. Override num_train_epochs.",
)
parser.add_argument("--warmup_steps",
                    default=0,
                    type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument("--warmup_thr",
                    default=None,
                    type=int,
                    help="apply min threshold to warmup steps")
parser.add_argument("--logging_steps",
                    type=int,
                    default=50,
                    help="Log every X updates steps.")
parser.add_argument("--save_steps",
                    type=int,
                    default=0,
                    help="Save checkpoint every X updates steps.")
parser.add_argument(
    "--eval_all_checkpoints",
    action="store_true",
    help=
    "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
)
parser.add_argument("--learning_rate",
                    default=2e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay",
                    default=0.0,
                    type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon",
                    default=1e-8,
                    type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm",
                    default=1.0,
                    type=float,
                    help="Max gradient norm.")
parser.add_argument("-seed",
                    "--seed",
                    default=42,
                    required=False,
                    type=int,
                    help="seed")
parser.add_argument("-patience",
                    "--patience",
                    required=False,
                    type=int,
                    default=None,
                    help="patience for early stopping (steps)")
##########################################################################
# Data args
##########################################################################
parser.add_argument("--dataset_name",
                    default=None,
                    required=True,
                    type=str,
                    help="Dataset [mrpc, ag_news, qnli, sst-2]")
parser.add_argument("--max_seq_length",
                    default=256,
                    type=int,
                    help="Max sequence length")
##########################################################################
# AL args
##########################################################################
parser.add_argument("-acquisition",
                    "--acquisition",
                    required=False,
                    default='naive',
                    type=str,
                    choices=acquisition_functions + proposal,
                    help="Choose an acquisition function to be used for AL.")
parser.add_argument("--candidate_scale",
                    required=False,
                    default=50,
                    type=int,
                    help="scale for the size of candidate set")
parser.add_argument("-budget",
                    "--budget",
                    required=False,
                    default=1050,
                    type=int,
                    help="the number of total annotations")
parser.add_argument(
    "-mc_samples",
    "--mc_samples",
    required=False,
    default=None,
    type=int,
    help="number of MC forward passes in calculating uncertainty estimates")
parser.add_argument("--resume",
                    required=False,
                    default=False,
                    type=bool,
                    help="if True resume experiment")
parser.add_argument(
    "--acquisition_size",
    required=False,
    default=50,
    type=int,
    help="acquisition size at each AL iteration; if None we sample 1%")
parser.add_argument("--init_train_data",
                    required=False,
                    default=50,
                    type=int,
                    help="initial training data for AL; if None we sample 1%")
parser.add_argument("--indicator",
                    required=False,
                    default=None,
                    type=str,
                    help="Experiment indicator")
parser.add_argument("--init",
                    required=False,
                    default="random",
                    type=str,
                    help="random, alps, or proposed...")
parser.add_argument("--reverse",
                    default=False,
                    type=bool,
                    help="if True choose opposite data points")
##########################################################################
# Contrastive acquisition args
##########################################################################
parser.add_argument("--mean_embs",
                    default=False,
                    type=bool,
                    help="if True use bert mean embeddings for kNN")
parser.add_argument("--mean_out",
                    default=False,
                    type=bool,
                    help="if True use bert mean outputs for kNN")
parser.add_argument("--cls",
                    default=True,
                    type=bool,
                    help="if True use cls embedding for kNN")
# parser.add_argument("--kl_div", default=True, type=bool, help="if True choose KL divergence for scoring")
parser.add_argument("--ce",
                    default=False,
                    type=bool,
                    help="if True choose cross entropy for scoring")
parser.add_argument("--operator",
                    default="mean",
                    type=str,
                    help="operator to combine scores of neighbours")
parser.add_argument("--num_nei",
                    default=10,
                    type=float,
                    help="number of kNN to find")
parser.add_argument("--conf_mask",
                    default=False,
                    type=bool,
                    help="if True mask neighbours with confidence score")
parser.add_argument("--conf_thresh",
                    default=0.,
                    type=float,
                    help="confidence threshold")
parser.add_argument("--knn_lab",
                    default=False,
                    type=bool,
                    help="if True queries are unlabeled data"
                    "else labeled")
parser.add_argument("--bert_score",
                    default=False,
                    type=bool,
                    help="if True use bertscore similarity")
parser.add_argument("--tfidf",
                    default=False,
                    type=bool,
                    help="if True use tfidf scores")
parser.add_argument("--bert_rep",
                    default=False,
                    type=bool,
                    help="if True use bert embs (pretrained) similarity")

parser.add_argument("--oracle",
                    default=False,
                    type=bool,
                    help="If you do Oracle-random")
##########################################################################
# Server args
##########################################################################
parser.add_argument("-g",
                    "--gpu",
                    required=False,
                    default='0',
                    help="gpu on which this experiment runs")
parser.add_argument("-server",
                    "--server",
                    required=False,
                    default='ford',
                    help="server on which this experiment runs")
# parser.add_argument("--debug", required=False, default=False, help="debug mode")

args = parser.parse_args()

if args.acquisition in ["tyrogue"]:
    args.exp_path = 'tyrogue'
else:
    args.exp_path = 'existing_method'
args.F = 0
args.tapt = None
args.weighting_sample = False

args.sanity = False
args.switch = 5
#############################################################

for seed in [42, 100, 200, 300, 400]:

    args.seed = seed

    run_main(args)
