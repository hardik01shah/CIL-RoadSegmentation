# Road Segmentation CIL

# Installation
```
conda create --name cil python=3.9
conda activate cil
pip install -U segmentation-models-pytorch
pip install -U albumentations
pip install opencv-python
pip install wandb
pip install matplotlib
```

# Directory Structure
```
.
├── configs/                       # Configuration files (e.g., hyperparameters, paths)
├── data/                          # Dataset and related files
│   ├── ethz-cil-road-segmentation-2024.zip   # Original dataset zip file
│   ├── test/                      # Test dataset
│   │   └── images/                # Test images
│   ├── training/                  # Training dataset
│   │   ├── groundtruth/           # Ground truth segmentation masks
│   │   └── images/                # Training images
│   ├── train.txt                  # List of training images
│   └── val.txt                    # List of validation images
├── mask_to_submission.py          # Script to convert masks to submission format
├── notebooks/                     # Jupyter notebooks for exploration and analysis
│   └── data_exploration.ipynb     # Data exploration notebook
├── README.md                      # Project description and instructions
├── scripts/                       # Utility scripts
│   └── gen_img_list.py            # Script to generate image lists
├── src/                           # Source code for training and evaluation
│   ├── dataloader.py              # Data loading and preprocessing
│   ├── eval_engine.py             # Evaluation engine
│   ├── eval.py                    # Evaluation script
│   ├── train_engine.py            # Training engine
│   └── train.py                   # Training script
├── submission_to_mask.py          # Script to convert submission format to masks
└── utils/                         # Utility functions and scripts
    ├── __init__.py                # Init file for utils package
    ├── losses.py                  # Custom loss functions
    ├── metrics.py                 # Custom metrics
    └── visualization.py           # Visualization utilities (e.g., plotting images, segmentation results)
```

### Description

- **configs/**: Contains configuration files for setting hyperparameters and paths.
- **data/**: Contains the dataset files, including the raw zip file, training and test images, and corresponding text files listing the images.
- **mask_to_submission.py**: Converts segmentation masks to the submission format required for evaluation.
- **notebooks/**: Contains Jupyter notebooks for data exploration and analysis.
- **README.md**: Provides an overview of the project, setup instructions, and usage information.
- **scripts/**: Contains utility scripts, such as generating image lists from the dataset.
- **src/**: Contains the main source code for the project, including scripts for data loading, training, and evaluation.
- **submission_to_mask.py**: Converts the submission format back to segmentation masks for analysis.
- **utils/**: Contains utility functions and scripts for losses, metrics, and visualization.