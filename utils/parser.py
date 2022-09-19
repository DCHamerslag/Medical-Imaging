from argparse import ArgumentParser, Namespace
import argparse 
import os
from utils.paths import ROOT
import toml

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--num_epochs',
                        dest='num_epochs', 
                        type=int, 
                        default=20)
    parser.add_argument('--logging', 
                        dest='logging',
                        action='store_true', 
                        default=False)
    parser.add_argument('--device',
                        dest='device', 
                        type=str, 
                        default='cpu')

    args = parser.parse_args()
    toml_config = toml.load(str(ROOT / "config.toml"))
    config = {**toml_config, **vars(args)}
    return Namespace(**config)