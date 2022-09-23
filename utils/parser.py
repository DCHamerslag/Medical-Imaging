from argparse import ArgumentParser, Namespace
import argparse 
import os
from typing import Dict
from utils.paths import ROOT
import toml

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--logging', 
                        dest='logging',
                        action='store_true', 
                        default=False)

    args = parser.parse_args()
    toml_config = toml.load(str(ROOT / "config.toml"))
    parameters = {**toml_config['parameters'], **vars(args)}
    parameters = Namespace(**parameters)
    check(parameters, toml_config)
    return parameters

def check(parameters: Namespace, toml_config: Dict) -> None:
    if parameters.model not in toml_config['implemented']['models']:
        print("Enter a valid model")
        quit()
