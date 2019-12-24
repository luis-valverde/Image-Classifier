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

def predict_topk_classes(image_path, model, top_k, mapping):
    ''' Predict the top_k classes of an image using a trained deep learning model
    '''
    
    # Preprocess the image
    pytorch_np_image = process_image(image_path)
    
    # Changing from numpy to pytorch tensor
    pytorch_tensor = torch.tensor(pytorch_np_image)
    pytorch_tensor = pytorch_tensor.float()
    
    # Used to make size of torch as expected (batch size = 1)
    pytorch_tensor = pytorch_tensor.unsqueeze(0)
    
    # Run model to make predictions
    model.eval()
    output = model.forward(pytorch_tensor)
    predictions = torch.exp(output)
    
    # Identify top predictions and top labels
    top_preds, top_labs = predictions.topk(top_k)

    # Detach top predictions into a numpy list
    top_preds = top_preds.detach().numpy().tolist()[0]
    
    # Change top labels into a list
    top_labs = top_labs.tolist()[0]

    # Open mapping file
    with open(mapping, 'r') as mapping_file:
        # Load category to name file
        category_to_name = json.load(mapping_file)
    
    # Contain label names for top k image predictions
    top_names = [category_to_name[str(name)] for name in top_labs]
    
    return top_preds, top_names
    
def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Load model from checkpoint
    model = load_checkpoint(args.checkpoint)
    
    # Predict the top_k most likely classes for a given image
    top_preds, top_names = predict_topk_classes(image_path=args.image_path, 
                                                model=model, 
                                                top_k=args.top_k,
                                                mapping=args.category_names)
    
    # Print top_k image names with their probabilities
    for index in range(len(top_names)):
        print("Top k:{}, image name is: {}. Probability: {}".format(index+1,top_names[index], top_preds[index]))

if __name__ == '__main__':
    main()