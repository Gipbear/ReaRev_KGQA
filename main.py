import argparse

from utils import create_logger
import torch
import os
import time
#from Models.ReaRev.rearev import 
from train_model import Trainer_KBQA
from parsing import add_parse_args
from utils import fix_seed

parser = argparse.ArgumentParser()
add_parse_args(parser)

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

if args.experiment_name == None:
    timestamp = str(int(time.time()))
    args.experiment_name = "{}-{}-{}".format(
        args.dataset,
        args.model_name,
        timestamp,
    )

def main():
    fix_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = create_logger(args)
    trainer = Trainer_KBQA(args=vars(args), model_name=args.model_name, logger_=logger)
    if not args.is_eval:
        trainer.train(args.num_epoch)
    else:
        assert args.load_experiment is not None
        if args.load_experiment is not None:
            ckpt_path = os.path.join(args.checkpoint_dir, args.load_experiment)
            print("Loading pre trained model from {}".format(ckpt_path))
        else:
            ckpt_path = None
        trainer.evaluate_single(ckpt_path)


if __name__ == '__main__':
    main()
