import yaml
from logging_config import setup_logging
from flow_pipeline import Flow_State


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
params = load_config('params.yaml')

if __name__ == "__main__":
    setup_logging()
    flow = Flow_State(params)
    # flow.training_samples_pipeline()
    # flow.indices_pipeline()
    # flow.preprocessing_pipeline()
    # flow.classification_pipeline()
    flow.tree_height_pipeline()
