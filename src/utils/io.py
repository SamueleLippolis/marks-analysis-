from pathlib import Path
import yaml

# Get a YAML and returns it in a python dictionary shape
def load_cfg(path="config/config.yaml"):
    with open(path) as f: # open the file in path and call it f
        return yaml.safe_load(f) # read the YAML and return it as a python dictionary 

# it akes sure that a folder exists 
def ensure_dir(p: Path): # it takes an input p, which is expected to be a Path
    # parents True: if p = data/2025 and neither "data" exists, it build "data" and "data/2025"
    # exixst_ok = True: if it exists ndo nothng, if not create it 
    p.mkdir(parents=True, exist_ok=True) 
    
