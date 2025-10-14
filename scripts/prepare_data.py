from src.utils.io import load_cfg
from src.data.make_ds import run_make_ds
from src.data.build_features import run_build_features

if __name__ == '__main__':
    cfg = load_cfg()
    p1 = run_make_ds(cfg)
    p2 = run_build_features(cfg)
    print(f"Interim: {p1}")
    print(f"Processed: {p2}")
