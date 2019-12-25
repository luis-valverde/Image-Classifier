# Image Classifier

## Table Of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Neccessary Files](#neccessary-files)
- [Part 1 - Jupyter notebook](#part-1---jupyter-notebook)
  - [How to Run](#how-to-run)
- [Part 2 - Jupyter notebook](#part-2---command-line-application)
  - [Specifications](#specifications)

## Introduction

This repository contains all my work for the Udacity's Machine Learning Introduction Nanodegree Program.

The goal of this project was to train an image classifier to recognize different species of flowers.
To accomplish this, I loaded a pre-trained model and defined a new untrained feed-forward network as a classifier. The classifier layers were then trained using backpropagation on preprocessed flowers data.
The loss and accuracy on the validation set were tracked to determine the best hyperparameters.

Then, the trained model was verified on test data (images the network has never seen either in training or validation) to give a good estimate of the model's performance on completely new images.
Finally, the trained model was used to predict the class (or classes) of an image using the trained deep learning model.

## Project Overview

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. You'll be using a dataset of 102 flower categories.

This project is broken down into 2 parts:

- **Part 1:** Implement an image classifier with PyTorch using a Jupyter notebook.
- **Part 2:** Convert the code from part 1 into 2 separate scripts: to train a network model and to classify an image based on an existing trained network model.

## Requirements

This project uses the following software and Python libraries:

- [Python](https://www.python.org/downloads/release/python-364/)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [pytorch](https://pytorch.org/)
- [scikit-learn (v0.17)](https://scikit-learn.org/0.17/install.html)
- [Matplotlib](https://matplotlib.org/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html).

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/distribution/) distribution of Python, which already has the above packages and more included.

## Neccessary Files

This repository contains several files needed to solve the project.

1. **image_classifier_project.ipynb:** This is the main file where I performed the work on the project.
2. **flowers directory:** The project dataset.
3. **cat_to_name.json:** This JSON file provides the mapping from category label to category flower name.
4. **train.py**: Python script to train a new network on a dataset and save the model as a checkpoint.
5. **predict.py**: Python script to predict the class for an input image ussing a trained network.
6. **util_functions.py**: Contains utility functions like loading data and preprocessing images.

## Part 1 - Jupyter notebook

### How to Run

In the Terminal or Command Prompt, navigate to the folder on your machine where you've put the project files, and then use the command:

```bash
jupyter notebook image_classifier_project.ipynb
```

 to open up a browser window or tab to work with your notebook.
 Alternatively, you can use the command:

 ```bash
jupyter notebook
```

or

```bash
ipython notebook
```

and navigate to the notebook file (image_classifier_project.ipynb) in the browser window that opens.

## Part 2 - Command Line Application

### Specifications

- Train a new network on a data set with **train.py**
  - Basic Usage : ```python train.py data_directory```
  - Prints out current epoch, training loss, validation loss, and validation accuracy as the netowrk trains
  - Options:
    - Set direcotry to save checkpoints: ```python train.py data_dor --save_dir save_directory```
    - Choose arcitecture (densenet121 or vgg16 available): ```python train.py data_dir --arch "vgg16"```
    - Set hyperparameters: ```python train.py data_dir --learning_rate 0.001 --hidden_layer1 120 --epochs 20```
    - Use GPU for training: ```python train.py data_dir --gpu gpu```
  - Output: A trained network ready with checkpoint saved for doing parsing of flower images and identifying the species.

- Predict flower name from an image with **predict.py** along with the probability of that name. That is you'll pass in a single image /path/to/image and return the flower name and class probability
  - Basic usage: ```python predict.py /path/to/image checkpoint```
  - Options:
    - Return top K most likely classes: ```python predict.py input checkpoint ---top_k 3```
    - Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_To_name.json```
    - Use GPU for inference: ```python predict.py input checkpoint --gpu```
