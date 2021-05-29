import json

from easydict import EasyDict as edict


def read_configs(config_file):
    with open(config_file) as configs:
        vgg_configs = json.load(configs)
        vgg_configs = edict(vgg_configs)
        return vgg_configs
