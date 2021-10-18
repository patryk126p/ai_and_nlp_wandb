# AI & NLP Day
## Weights & Biases + MLflow

Repo using Weights & Biases for tracking experiments and MLflow for separating environments for executing steps in pipeline

## Setup

1. Create account at [Weights & Biases](https://wandb.ai/site)
2. Install requirements `pip install -r requirements.txt`
3. Login to Weights & Biases from terminal `wandb login`

## How to use this repo

After making sure setup is finished:
- execute to run full pipeline:
```
cd <REPO_ROOT>
mlflow run .
```
- login to W&B in browser and browse your project (name is set in `config.yaml` under `main.project_name`)

All necessary pipeline configuration options can be found in `config.yaml`.<br>
Options specific to model are stored in `src/2_train_model/model_config.json`
