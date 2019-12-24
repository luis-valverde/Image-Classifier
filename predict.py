#!/usr/bin/env python

# Define all the imports
import argparse
import json
from util_functions import *
    
def parse_args():
    ''' This function parsers the input arguments from command-line
    '''
    
    parser = argparse.ArgumentParser()

    # Set positional arguments
    parser.add_argument('image_path', type=str, help="Specify image to use")
    parser.add_argument('checkpoint', type=str, help="Specify model checkpoint")
    
    # Set optional arguments 
    parser.add_argument('-k', '--top_k',          metavar='', type=int, default=5,                  help="Return top K most likely classes")
    parser.add_argument('-c', '--category_names', metavar='', type=str, default='cat_to_name.json', help="Mapping of categories to real names file")
    parser.add_argument('-g', '--gpu',            action='store_true',                              help="Enable GPU mode")
    
    # Parse arguments
    args = parser.parse_args()

    return args

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Load model from checkpoint
    model = load_checkpoint(args.checkpoint)
    
if __name__ == '__main__':
    main()