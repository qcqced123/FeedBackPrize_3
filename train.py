import argparse, collections
from omegaconf import OmegaConf

from configuration import CFG
from parse_config import ConfigParser
from trainer.train_loop import *
from utils.helper import check_library, all_type_seed
from utils import sync_config


check_library(True)
all_type_seed(CFG, True)


def main(config_path: str, cfg) -> None:
    sync_config(OmegaConf.load(config_path))  # load json config
    # cfg = OmegaConf.structured(CFG)
    # OmegaConf.merge(cfg)  # merge with cli_options
    train_loop(cfg)


if __name__ == '__main__':
    # args = argparse.ArgumentParser(description='PyTorch Template')
    # args.add_argument('-c', '--config', default=None, type=str,
    #                   help='config file path (default: None)')
    # args.add_argument('-r', '--resume', default=None, type=str,
    #                   help='path to latest checkpoint (default: None)')
    # args.add_argument('-d', '--device', default=None, type=str,
    #                   help='indices of GPUs to enable (default: all)')
    #
    # # custom cli options to modify configuration from default values given in json file.
    # CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    # options = [
    #     CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
    #     CustomArgs(['--bs', '--batch_size'], type=int, target='dataset_class;args;batch_size')
    # ]
    # cli_config = ConfigParser.from_args(args, options)
    main('fbp3_config.json', CFG)
