# AI ERA Assignment5

## Overview

This repository contains a simple MNIST classifier built using PyTorch. The model architecture is based on a convolutional neural network (CNN) with three convolutional layers and two fully connected layers. The model is designed to classify handwritten digits from the MNIST dataset.
This project contains a CI/CD pipeline that trains the model, runs tests, and archives the model artifacts.

## How to run locally
To run this locally, clone the repository and do the following:

1. Set up your environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
2. Train the model:
```
python src/train.py
```
3. Run the tests:
```
pytest tests/
```

