#!/usr/bin/env python

# Define all the imports
import sys
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image

def create_transforms():
    ''' Create transforms for the training, validation, and testing sets
    '''
    
    # Mean and standard deviation of the images color channels
    mean_values = [0.485, 0.456, 0.406]
    std_values = [0.229, 0.224, 0.225]

    # Tranform for the training data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean_values, std_values)])

    # Tranform for the validation data
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean_values, std_values)])

    # Tranform for the testing data
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean_values, std_values)])
    
    result_transforms = {"train": train_transforms,
                         "valid": valid_transforms,
                         "test":  test_transforms
                        }
    
    return result_transforms
