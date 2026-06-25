# Balzer Raphael DCV 2026

## Repository Structure

- `notebooks/`: Contains Jupyter notebooks for training and data validation.
- `results/`: Stores the results of model evaluation.
- `plots/`: Contains generated plots for performance metrics.
- `data_extraction.py`: Script for extracting the datasets from the original source.

## Data Preparation

In theory, the DocLayNet dataset can be accessed via the [HuggingFace Hub](https://huggingface.co/datasets/docling-project/DocLayNet). However, when attempting to download/stream the dataset using the `datasets` library, an error occurs due to HuggingFace no longer supporting custom dataset loading scripts. Therefore, the dataset must be downloaded manually from the official [DocLayNet Github Repository](https://github.com/DS4SD/DocLayNet) and the path to the dataset must be specified in the `data_extractor.py` script. The script will then use the predefined train/validation/test splits to extract the subsets. For reproducability, the extracted datasets are provided in zip format in the following Google Drive folder: [DocLayNet Subsets](https://drive.google.com/drive/folders/1ctvrQwZ049NzQs0W3zVmO3rpAYFHvWNH?usp=sharing).

## Model Training

For model training and evaluation, the `train_evaluate.ipynb` notebook is provided. It is designed to be run in Google Colab, which is why the notebook is set up to download the dataset in zip format from Google Drive. It is designed to load one dataset at a time by switching out the dataset path. If you want to run the notebook locally, you will need to modify the data loading section to point to your local dataset path. For running the notebook in Colab, uploading the dataset zip files to your personal Google Drive and adjusting the paths in the notebook is necessary.

## Data Validation

The `data_validation.ipynb` on the other hand is designed to be run locally after the dataset has been extracted. It checks for the existence of image files corresponding to the annotations in the dataset and ensures the datset splits are correct.
