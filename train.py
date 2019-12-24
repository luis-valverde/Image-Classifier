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

def train_model(model, epochs, gpu, learning_rate, data_loaders):
    ''' Train the classifier layers with backpropagation using the pre-trained network to get the features
    '''
    
    print("Training the pre-trained network")
    
    # Use GPU if it's specified in the command-line
    device = torch.device("cuda" if gpu else "cpu")
    
    # Move the model to the specific device
    model.to(device)

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    
    steps = 0
    running_loss = 0
    print_every = 40
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in data_loaders['train']:
            steps += 1

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0

                # switching to evaluation mode so that dropout is turned off
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    for inputs, labels in data_loaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equality = (labels.data == ps.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()

                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Valid Loss: {:.3f}.. ".format(valid_loss/len(data_loaders['valid'])),
                      "Valid Accuracy: {:.3f}%".format(accuracy/len(data_loaders['valid'])*100))

                running_loss = 0

                # Make sure training is back on
                model.train()
                
    return model

def test_model(model, test_loader, gpu):
    ''' Test the trained network and measure the accuracy
    '''

    print("Testing the trained network")
    
    # Use GPU if it's specified in the command-line
    device = torch.device("cuda" if gpu else "cpu")
    
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max (outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = (correct/total) * 100

    print("Accuracy on test images: {:.3f}%. Total of images: {}".format(accuracy, total))
    
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

    # Train the model
    trained_model = train_model(model=pretrained_model, 
                                epochs=args.epochs, 
                                gpu=args.gpu, 
                                learning_rate=args.learning_rate, 
                                data_loaders=data_loaders)

    # Test the trained model
    test_model(model=trained_model, 
               test_loader=data_loaders['test'],
               gpu=args.gpu)

if __name__ == '__main__':
    main()