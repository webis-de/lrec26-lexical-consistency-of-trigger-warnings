from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parent / "conf"

with open(CONFIG_PATH / "config.yaml") as f:
    conf_dict = yaml.safe_load(f)

PROJECT_DIR = Path(conf_dict["project_dir"])
TAG_PATH = Path(conf_dict["tag_path"])

OUTPUT_PATH = PROJECT_DIR / "output"
DATA_PATH = OUTPUT_PATH/ "data"
VISUALIZATION_PATH = OUTPUT_PATH / "visualization"
MODEL_PATH = PROJECT_DIR / "models"
RESOURCE_PATH = PROJECT_DIR / "resources"

# Create the path if they do not exist
DATA_PATH.mkdir(exist_ok=True, parents=True)
VISUALIZATION_PATH.mkdir(exist_ok=True, parents=True)

if __name__ == '__main__':
    print(f"{RESOURCE_PATH}|{DATA_PATH}")