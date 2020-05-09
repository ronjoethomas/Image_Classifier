
#This file contains the updated trainied model from image_classifier to provide a viusal proof that it works with basic image classification

#importing the neccessary modules
import numpy as np
import torch
from torch import nn
from torch import optim
#this import allows us to load images of the dataset
from torchvision import datasets,transforms,models
import matplotlib.pyplot as plt
import image_plot_helper


#load the data of cats or dogs (just using a test loader in this case. Can change to anything after)

data_directory = '/home/ron/python_projects/Image_classifier/Image_Classifier/Animal_Data/Animal_data/test'
data_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
data = datasets.ImageFolder(data_directory, transform=data_transforms)
dataloader = torch.utils.data.DataLoader(data, batch_size=64, shuffle = True)

#load a pretrained model. First make sure the classifiers are the same
model = models.densenet121(pretrained=True)


#create the new classifier
model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(512, 2),
                                 nn.LogSoftmax(dim=1))

#Use the trained model (done in image_classifier.py) to update this model with the proper weights
model = torch.load("trained_specific_model.pth")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#main loop that displays random pictures of cats and dogs while classifying them. 
#pressing any button will show a new random classified image
#typing exit will end the loop

while True:
    #can change test loader to any loader that contains the image that we want to see 
    test_iter = iter(dataloader)
    images,labels = test_iter.next()

    images,labels = images.to(device),labels.to(device)

    #get the first image(just to test, can chanage thos)
    image_current=images[0]
    image_current=image_current.view(1,3,224,224)

    with torch.no_grad():
        output = model.forward(image_current)
    ps = torch.exp(output)
        
    #create the plot
    fig, (ax1, ax2) = plt.subplots(figsize=(12,14), ncols=2)
    image_plot_helper.imshow(images[0].cpu(),ax=ax1)
    image_plot_helper.graphShow(ax=ax2,ps=ps.cpu())
    plt.show()

    #once user closes images prompt for user input
    typed_instruction = input("Please press any key (and/or enter) to show a new random image. Or type in 'exit' to stop: ")

    if typed_instruction == 'exit':
        break

