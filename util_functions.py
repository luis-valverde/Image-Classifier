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

def create_datasets(data_directory, transforms):
    ''' Create data sets for the training, validation and testing sets
    '''
    
    # Set training, validation and testing data directories
    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    test_dir = data_directory + '/test'

    # Load the datasets with ImageFolder

    # Training dataset
    train_data = datasets.ImageFolder(train_dir, transform=transforms['train'])

    # Validation dataset
    valid_data = datasets.ImageFolder(valid_dir, transform=transforms['valid'])

    # Testing dataset
    test_data = datasets.ImageFolder(test_dir, transform=transforms['test'])
    
    result_datasets = {"train": train_data,
                       "valid": valid_data,
                       "test":  test_data
                       }
    
    return result_datasets

def create_dataloaders(datasets):
    ''' Create data loaders for the training, validation and testing sets
    '''

    # Specify the batch size
    batch_size_value = 64

    # Training dataloader
    train_loader = torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size_value, shuffle = True)

    # Validation dataloader
    valid_loader = torch.utils.data.DataLoader(datasets['valid'], batch_size=batch_size_value, shuffle = True)

    # Testing dataloader
    test_loader = torch.utils.data.DataLoader(datasets['test'], batch_size=batch_size_value, shuffle = True)
    
    result_dataloaders = {"train": train_loader,
                          "valid": valid_loader,
                          "test":  test_loader
                         }
    
    return result_dataloaders
    