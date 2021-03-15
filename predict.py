# Package Imports
import matplotlib
import matplotlib.pyplot as plt
import torch 
from torch import nn, optim
import torch.nn.functional as F
from torch.utils import data
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import argparse
import json
from collections import OrderedDict

# Creating the Parser & Arguments
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--img_path', dest='img_path', default='flowers')
    parser.add_argument('--model_path', dest='model_path')
    parser.add_argument('--category_names_json_filepath', dest='category_names_json_filepath',               default='cat_to_name.json')
    parser.add_argument('--top_k', dest='top_k', default=3, type=int)
    parser.add_argument('--gpu', dest='gpu', action='store_true')

    args = parser.parse_args()
    
#Loading checkpoints
#Model architecture

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['model'] == "vgg16":
        model = torchvision.models.vgg16(pretrained=True)

    elif checkpoint['model'] == "densenet":
        model = torchvision.models.densenet121(pretrained=True)
    
#Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.model = checkpoint['model']
    
    return model

#Image Processing
def process_image(img_path):

    pil_image = Image.open(img_path)

    w = 256
    h = 256
    pil_image = pil_image.resize((w, h))

    new_w = 224
    new_h = 224

    left = (w - new_w) / 2
    top = (h - new_h) / 2
    right = (w + new_w) / 2
    bottom = (h + new_h) / 2

    pil_image = pil_image.crop((left, top, right, bottom))
    
    np_image = np.array(pil_image)/255
    np_image = np_image.transpose((2 , 0, 1))
    
    tensor = torch.from_numpy(np_image)
    tensor = tensor.type(torch.FloatTensor)
   
    return tensor

#Predicting classes
def predict(img_path, device, model, topk, cat_to_name):

    image = process_image(img_path)
    image = image.unsqueeze(0)

    image = image.to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.exp(model(image))
        
    idx_to_flower = {v:cat_to_name[k] for k, v in model.class_to_idx.items()}    
    
    top_probs, top_class = probs.topk(topk, dim=1)
    predicted_flowers = [idx_to_flower[i] for i in top_class.tolist()[0]]

    return top_probs.tolist()[0], predicted_flowers

def print_predictions(args):
# load model
    model = load_checkpoint(args.model_path)
    
##Model architecture
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    if args.gpu and not(torch.cuda.is_available()):
        device = 'cpu'
    else:
        device = 'cpu'

    model = model.to(device)

    with open(args.category_names_json_filepath, 'r') as f:
        cat_to_name = json.load(f)

#Predicting Image
    top_probs, top_class = predict(args.img_path, device, model, args.top_k, cat_to_name)

    print("Predictions:")
    for i in range(args.top_k):
          print("No.{: <3} {: <25} -> Probability of {:.2f}%".format(i, top_class[i], top_probs[i]*100))

print_predictions(args)


#Enter: python predict.py --img_path flowers/test/1/image_06743.jpg --model_path checkpoint.pth --category_names cat_to_name.json --top_k 4 --gpu
