# VGG-pet-Vision

## About

VGG-pet-Vision is a project that focuses on classifying images of dogs and cats using the VGG-16 convolutional neural network architecture. The goal is to build a machine learning model that accurately distinguishes between images of dogs and cats.

## Project Structure


## Models

The trained machine learning model is saved in the `artifacts/` directory.

Trained Model: `artifacts/model_vgg16.pth`

## Data

The project uses the Kaggle Dogs vs. Cats dataset, which contains images of dogs and cats for training and testing the model.

## Pipelines

1. **Data Preprocessing**:
   - `src/pipelines/data_preprocessing.py`: Script for preprocessing the raw image data, including resizing, normalization, and augmentation.

2. **Model Training**:
   - `src/pipelines/model_training.py`: Script for training the VGG-16 model using the preprocessed data.

3. **Prediction**:
   - `src/pipelines/prediction.py`: Script for making predictions using the trained model on new images.


## Usage

To train the model:
```bash
python src/pipelines/model_training.py
