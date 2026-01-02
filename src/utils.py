from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML_PATH = PROJECT_ROOT / "data/data.yaml"


def load_class_names():
    with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        # sometimes names can be a dict {0:"Apple", ...}
        names = [names[i] for i in sorted(names.keys())]
    return names


def ensure_exists(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
