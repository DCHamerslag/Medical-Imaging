from argparse import ArgumentParser, Namespace
import argparse 
import os
from typing import Dict
from utils.paths import ROOT, DATA, MODELS
import toml

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--logging', 
                        dest='logging',
                        action='store_true', 
                        default=False)
    parser.add_argument( '--data_dir', action = 'store', 
                        type = str, 
                        help = 'Datadir',
                        default="Default")
    parser.add_argument( '--data_name', action = 'store', 
                        type = str, 
                        help = 'dataset name',
                        default="thing")
    parser.add_argument( '--gamma', action = 'store', 
                        type = float, 
                        help = 'Scheduler gamma',
                        default=0.1)
    parser.add_argument( '--lr', action = 'store', 
                        type = float, 
                        help = 'Learning rate',
                        default=0.001)
    parser.add_argument( '--batch_size', action = 'store', 
                        type = int, 
                        help = 'Batch Size',
                        default=8)


    args = parser.parse_args()
    toml_config = toml.load(str(ROOT / "config.toml"))
    merge = {**toml_config['parameters'], **vars(args)}
    parameters = Namespace(**merge)
    if parameters.data_dir == "Default":
        parameters.data_dir = str(DATA)
        parameters.model_dir = str(MODELS)
    else:
        parameters.model_dir = parameters.data_dir + "/models" # hacky solution for lisa scratch path
    check(parameters, toml_config)

    return parameters

def check(parameters: Namespace, toml_config: Dict) -> None:
    if parameters.model not in toml_config['implemented']['models']:
        print("Enter a valid model")
        quit()
