import argparse
from pathlib import Path
import yaml


def set_value(config_path, key, value):
    with open(config_path / "config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    config[key] = value
    with open(config_path / "config.yaml", 'w') as f:
        yaml.dump(config, f)


def remove_tokens(config_path):
    with open(config_path / "config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    for x in ["kibana", "huggingface"]:
        config[x]["key"] = "YOUR KEY"

    with open(config_path / "config.yaml", 'w') as f:
        yaml.dump(config, f)


def prepare_conf_for_docker(config_path):
    remove_tokens(config_path)


if __name__ == '__main__':
    # Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", metavar="c", type=str,
                        help='Path to config file')

    args = parser.parse_args()
    prepare_conf_for_docker(Path(args.config_path))