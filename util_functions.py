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

def choose_model(arch):
    ''' Choose a pre-trained network for a given architecture
    '''
    
    available_models = {'resnet18':[models.resnet18(pretrained=True), models.resnet18(pretrained=True).fc.in_features],
                        'alexnet':[models.alexnet(pretrained=True), models.alexnet(pretrained=True).classifier[1].in_features],
                        'vgg16':[models.vgg16(pretrained=True), models.vgg16(pretrained=True).classifier[0].in_features],
                        'densenet121':[models.densenet121(pretrained=True), models.densenet121(pretrained=True).classifier.in_features]
                       }
    
    
    # Look for the model for the given architecture
    if arch in available_models:
        # Return model and model input_features size
        return available_models[arch][0], available_models[arch][1]
    else:
        sys.exit('There is not a pre-trained model for the specified architecture: {}'.format(arch))

def load_pretrained_model(arch, drop_out, hidden_units, output_units):
    ''' Load a pre-trained network for a given architecture
    '''
    
    print("Loading a pre-trained network")
    
    # Load a pre-trained network
    model, in_features = choose_model(arch=arch)
    
    # Freeze parameters so it doesn't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Build a feed-forward network
    classifier = nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(drop_out),
        nn.Linear(hidden_units, output_units),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier
    
    return model
    
def save_checkpoint(model, args, train_data, save_dir, file_name):
    ''' Save checkpoint of the trained model
    '''
    
    print("Saving checkpoint")
    
    model.class_to_idx = train_data.class_to_idx
    model.cpu()

    checkpoint = {'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'drop_out':args.drop_out,
                  'arch': args.arch,
                  'hidden_units':args.hidden_units, 
                  'output_units':args.output_units}
    
    # Path and name of the checkpoint file
    path = save_dir + file_name + '.pth'
    
    # Save checkpoint
    torch.save(checkpoint, path)
    