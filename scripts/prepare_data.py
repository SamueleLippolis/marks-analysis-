from src.utils.io import load_cfg
from src.utils.build_interim_ds import run_build_interim_ds
from src.utils.build_processed_ds import run_build_processed_ds

if __name__ == '__main__':
    cfg = load_cfg()
    p1 = run_make_ds(cfg)
    p2 = run_build_features(cfg)
    print(f"Interim: {p1}")
    print(f"Processed: {p2}")
