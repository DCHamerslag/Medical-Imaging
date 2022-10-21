# Airogs-Lite Challenge submission
## Setup
``` bash
conda env create -f env.yml
conda activate airogs
```

## Usage
Use config.toml to select the model and hyperparameters.

The path to the dataset can be specified using command line arguments as shown in utils/parser.py
``` bash
python main.py
```
### Optional Arguments
--logging: to enable WandB logging (requires a Weights and Biases account)


## Authors
Rachel Liao -- Ramon Cremers -- Dico van Leeuwen -- Stijn Hamerslag
