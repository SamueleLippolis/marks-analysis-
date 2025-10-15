# Overview 
I use some ML models to analyse school marks and set a proper github repo. 

# How to use 
Write the following lines on terminal in the folder
- conda activate ENVNAME   (activate your py env)
- export PYTHONPATH=$PWD   (set your python env in the folder)
- python scripts/prepare_data (from raw_data, it builds interim_data and processed_data)
- python scripts/explore_interim (build figures regarding interim_data)
- python scripts/train_meanModel (eval and generate figures about meanModel)
- python scripts/train_lr (train, eval and generate figures regarding linear_regression_model) 

# Folder explanation
- config (config file)
- data (used datasets)
  - raw (original dataset)
  - interim (interpretable cleaned dataset)
  - processed (normalized ready-to-use dataset)
- notebooks (prototype-notebooks (tests) and notebooks (to play))
  - archive (disude notebooks)
- reports (results from data and experiments)
  - data_analysis (results regarding data)
  - experiments (results regarding model applications on data)
    - EXP_NAME1 (figures and metrics of a specific experiment)
    - EXP_NAME2 (same)
    - ... (same)
- scripts (file to run to do all the stuff)
- src (functions storage)
  - models (all model functions)
  - utils (data and generic useful functions)
  - visualize (functions to make figures about data and experiment analysis )


