#!/usr/bin/env python

# Define all the imports
import argparse
from util_functions import *
    
def parse_args():
    ''' This function parsers the input arguments from command-line
    '''
    
    parser = argparse.ArgumentParser()

    # Set positional arguments
    parser.add_argument('data_directory', type=str, help="Specify data directory")

    # List of possible architectures to use
    archs_list = ['resnet18', 'alexnet', 'vgg16', 'densenet121']
    
    # Set optional arguments 
    parser.add_argument('-s', '--save_dir',      metavar='', type=str,   default='',      help="Set directory to save checkpoints. Example: 'my_directory/'")
    parser.add_argument('-l', '--learning_rate', metavar='', type=float, default=0.01,    help="Learning rate")
    parser.add_argument('-u', '--hidden_units',  metavar='', type=int,   default=512,     help="Model hidden units")
    parser.add_argument('-o', '--output_units',  metavar='', type=int,   default=102,     help="Model output units")
    parser.add_argument('-d', '--drop_out',      metavar='', type=float, default=0.2,     help="Model dropout")
    parser.add_argument('-e', '--epochs',        metavar='', type=int,   default=10,      help="Number of epochs during training")
    parser.add_argument('-g', '--gpu',           action='store_true',                     help="Enable GPU mode")
    
    parser.add_argument('-a', '--arch', metavar='', type=str, default='vgg16', choices=archs_list, 
                        help="Choose architecture. Options: {}".format(archs_list))
    
    # Parse arguments
    args = parser.parse_args()

    return args

def main():
    # Parse command-line arguments
    args = parse_args()

    # Create transforms for the training, validation, and testing sets
    transforms = create_transforms()

    # Create datasets for the training, validation and testing sets
    data_sets = create_datasets(data_directory=args.data_directory, 
                                transforms=transforms)

    # Create data loaders for the training, validation and testing sets
    data_loaders = create_dataloaders(datasets=data_sets)

    # Load a pre-trained network
    pretrained_model = load_pretrained_model(arch=args.arch,
                                             drop_out=args.drop_out,
                                             hidden_units=args.hidden_units,
                                             output_units=args.output_units)
                                                                  
if __name__ == '__main__':
    main()