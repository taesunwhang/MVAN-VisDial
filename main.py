import os
import argparse
import collections
import logging
from datetime import datetime

from config.hparams import *
from single_train import MVAN
from multi_train import MultiMVAN
from single_evaluation import Evaluation
from multi_evaluation import MultiEvaluation

PARAMS_MAP = {
  "mvan" : BASE_PARAMS,
  "mvan_multi" : MVAN_MULTI_PARAMS,
  }

def init_logger(path:str):
  if not os.path.exists(path):
      os.makedirs(path)
  logger = logging.getLogger()
  logger.handlers = []
  logger.setLevel(logging.DEBUG)
  debug_fh = logging.FileHandler(os.path.join(path, "debug.log"))
  debug_fh.setLevel(logging.DEBUG)

  info_fh = logging.FileHandler(os.path.join(path, "info.log"))
  info_fh.setLevel(logging.INFO)

  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)

  info_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
  debug_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s | %(lineno)d:%(funcName)s')

  ch.setFormatter(info_formatter)
  info_fh.setFormatter(info_formatter)
  debug_fh.setFormatter(debug_formatter)

  logger.addHandler(ch)
  logger.addHandler(debug_fh)
  logger.addHandler(info_fh)

  return logger

def train_model(args):
  hparams = PARAMS_MAP[args.model]

  # dataset_version
  if args.version == "0.9":
    hparams.update(OLD_DATASET_PARAMS)

  root_dir = hparams["root_dir"]
  root_dir += '%s-%s_%s' % (hparams["encoder"], hparams["decoder"], args.version)
  hparams["model_name"] = '%s-%s' % (hparams["encoder"], hparams["decoder"])

  hparams.update(root_dir=root_dir)

  if hparams["load_pthpath"] != "":
    hparams.update(random_seed=[int(args.eval_seed)])

  timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
  root_dir = os.path.join(hparams["root_dir"], "%s/" % timestamp)
  logger = init_logger(root_dir)
  logger.info("Hyper-parameters: %s" % str(hparams))
  logger.info(args.description)
  hparams["root_dir"] = root_dir
  hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)

  if hparams.decoder == "disc_gen":
    print("MULTITASK LEARNING")
    model = MultiMVAN(hparams)
    eval_ckpt_path = model.train()
    if hparams.dataset_version == '1.0':
      model = MultiEvaluation(hparams, split="test")
      model.run_evaluate(
        eval_ckpt_path,
        eval_json_path=os.path.join(os.path.dirname(eval_ckpt_path),
                                    "%s_%d_test.json" %
                                    (hparams.encoder + "-" + hparams.decoder, hparams.random_seed[0])))
  else:
    model = MVAN(hparams)
    eval_ckpt_path = model.train()
    if hparams.dataset_version == '1.0':
      model = Evaluation(hparams, split="test")
      model.run_evaluate(
        eval_ckpt_path,
        eval_json_path=os.path.join(os.path.dirname(eval_ckpt_path),
                                    "%s_%d_test.json" %
                                    (hparams.encoder + "-" + hparams.decoder, hparams.random_seed[0])))


def evaluate(args):
  hparams = PARAMS_MAP[args.model]
  hparams["model_name"] = '%s-%s' % (hparams["encoder"], hparams["decoder"])

  # dataset_version
  if args.version == "0.9":
    hparams.update(OLD_DATASET_PARAMS)

  hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)
  print(hparams.decoder)
  if hparams.decoder == "disc_gen":
    print("MULTITASK LEARNING")
    model = MultiEvaluation(hparams, split=args.eval_split)
    model.run_evaluate(
      args.evaluate,
      eval_json_path=os.path.join(os.path.dirname(args.evaluate),
                                  "%s_%s_%s.json" %
                                  (hparams.encoder + "-" + hparams.decoder, args.eval_seed, args.eval_split)),
      eval_seed=args.eval_seed
    )
  else:
    model = Evaluation(hparams, split=args.eval_split)
    model.run_evaluate(
      args.evaluate,
      eval_json_path=os.path.join(os.path.dirname(args.evaluate),
                                  "%s_%s_%s.json" %
                                  (hparams.encoder + "-" + hparams.decoder, args.eval_seed, args.eval_split)),
      eval_seed=args.eval_seed
    )
if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser(description="Mutli-View Attention Networks")
  arg_parser.add_argument("--model", dest="model", type=str, default=None,
                          help="Model Name")
  arg_parser.add_argument("--version", dest="version", type=str, default='1.0',
                          help="Dataset Version")
  arg_parser.add_argument("--evaluate", dest="evaluate", type=str,
                          help="Evaluation Checkpoint", default=False)
  arg_parser.add_argument("--eval_split", dest="eval_split", type=str,
                          help="Evaluation split", default="test")
  arg_parser.add_argument("--description", dest="description", type=str, default='',
                          help="Image Region Proposal Type")
  arg_parser.add_argument("--eval_seed", dest="eval_seed", type=str,
                          help="Evaluation split", default="3143")

  args = arg_parser.parse_args()
  if args.evaluate:
    print("EVALUATION")
    evaluate(args)
  else:
    print("MODEL TRAIN")
    train_model(args)