# Package Imports
import matplotlib
import matplotlib.pyplot as plt
import torch 
from torch import nn, optim
from torch import nn
import torch.nn.functional as F
from torch.utils import data
import torchvision
from torchvision import datasets, transforms, models
import torchvision.models as models
import numpy as np
from PIL import Image
import argparse
import json
from collections import OrderedDict
   
#Training data augmentation
#Data normalization
def data_transformation(args):
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
#Define your transforms for the training & validation
    train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(), 
                                       transforms.ToTensor(), 
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    valid_transforms = transforms.Compose([transforms.Resize(256), 
                                       transforms.CenterCrop(224), 
                                       transforms.ToTensor(), 
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
#Data loading
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

#Data batching
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, shuffle=True)

    return trainloader, validloader, train_dataset.class_to_idx

#Pretrained Network

def train_model(args, trainloader, validloader, class_to_idx):
#Model architecture
    if args.model_arch == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)                         
    elif args.model_arch == 'densenet121':
        model = torchvision.models.densenet121(pretrained=True)     
    for param in model.parameters():
        param.requires_grad = False
 
#Feedforward Classifier/Model hyperparameters
    in_features_of_pretrained_model = model.classifier[0].in_features

    classifier = nn.Sequential(nn.Linear(in_features=in_features_of_pretrained_model, out_features=2048, bias=True),
                               nn.ReLU(inplace=True),
                               nn.Dropout(p=0.2),
                               nn.Linear(in_features=2048, out_features=102, bias=True),
                               nn.LogSoftmax(dim=1)
                              )

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

#Training with GPU
    device = 'cuda'
    model.to(device)
    
#Training the network
    epochs = 3
    print_every = 25

    for e in range(args.epochs):
        steps = 0
        train_loss = 0
        valid_loss = 0
        valid_accuracy = 0
 
        for inputs, labels in trainloader:
            steps += 1
            model.train()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            train_loss = criterion(output, labels)
            train_loss.backward()
            optimizer.step()
            train_loss += train_loss.item()
            
            print("Epoch: {}/{}".format(e+1, epochs, (steps)*100/len(trainloader)))
            
#Validation Loss and Accuracy
#Testing Accuracy
            model.eval()
            with torch.no_grad():
                valid_loss = 0
                valid_accuracy = 0

                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model(inputs)
                    valid_loss = criterion(output, labels)
                    valid_loss += valid_loss.item()
                
                    probs = torch.exp(output)
                    top_prob, top_class = probs.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
#Training validation log        
            avg_train_loss = train_loss/len(trainloader)
            avg_valid_loss = valid_loss/len(validloader)
            valid_accuracy = valid_accuracy/len(validloader)
            print("Epoch: {}/{}".format(e+1, epochs, (steps)*100/len(trainloader)))
            print("Train Loss: {:.2f}".format(avg_train_loss))
            print("Valid Loss: {:.2f}".format(avg_valid_loss))
            print("Accuracy: {:.2f}%".format(valid_accuracy*100)) 

#Saving the model

            model.class_to_idx = class_to_idx
    
            checkpoint = {'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'epochs': args.epochs,
                  'optim_stat_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'model': args.model_arch
                 }

            torch.save(checkpoint,'checkpoint.pth')
            print("Checkpoint saved!")

            
if __name__ == '__main__':
# Creating the Parser & Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', dest='data_dir', help='Directory containing data')
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.003, type=float)
    parser.add_argument('--epochs', dest='epochs', default=3, type=int)
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--model_arch', dest='model_arch', default="vgg16", type=str, choices=['vgg16', 'densenet121'])

    args = parser.parse_args()

#loading and transforming data
    trainloader, validloader, class_to_idx = data_transformation(args)

#training and saving model
    train_model(args, trainloader, validloader, class_to_idx)
              