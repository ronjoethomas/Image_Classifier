#This file makes use of deep learning involving Pytorch to creater an image classfier

#importing the neccessary modules

import numpy as np
import torch
from torch import nn
from torch import optim
#this import allows us to load images of the dataset
from torchvision import datasets,transforms,models
import matplotlib.pyplot as plt
import image_plot_helper

#first we must load the data, define the transforms on the data and put it into test and train loaders

#this directory is where the data for testing and training are located

data_directory = '/home/ron/python_projects/Image_classifier/Image_Classifier/Animal_Data/Animal_data'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_directory + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_directory + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle = True)

#load a pretrained model
model = models.densenet121(pretrained=True)

#keep the feature parameters the same but need to update the classfiers
for param in model.parameters():
    param.requires_grad=False  #no need to backdrop through parameters


#create the new classifier
model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(512, 2),
                                 nn.LogSoftmax(dim=1))

#set device to cuda if available or put 'cpu' here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#only need to train the classifier paramters. Also set up the criterion for negative log likelihood
optimizer = optim.Adam(model.classifier.parameters(),lr=0.001)
criterion = nn.NLLLoss()

epochs =1


for e in range(epochs):


    #training loop
    running_loss = 0
    times=0
    #go into training mode (in order to prevent train set memorization by using dropouts)
    model.train()
    for images,labels in trainloader:
        times=times+1
        #move the images and labels to either the cpu and gpu
        images,labels = images.to(device),labels.to(device)

        optimizer.zero_grad()

        #print out the losses and you should see it decreasing as the number of images this iterator goes though increased    
        if times%10 == 0:
            print(f"Training: The steps:... {times}"
                f" The loss:...{running_loss/times:.3f} ")

        log_ps = model.forward(images)
        loss = criterion(log_ps,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    #Testing Loop (measure accuracy)
    model.eval()
    accuracy = 0
    print("please wait while running the test dataset....")
    with torch.no_grad():
        for turns, (images,labels) in enumerate(testloader):
            
            images,labels = images.to(device),labels.to(device)
            log_ps = model.forward(images)

            ps = torch.exp(log_ps)

            top_p, top_class = ps.topk(1,dim=1)
            equals = top_class == labels.view(*top_class.shape)

            #calculate accuracy
            accuracy += torch.mean(equals.type(torch.FloatTensor))
            turns+=turns
        #consider the average accuracy of all test cases
    
    print("Test: The accuracy based on test images is: {:.3f}".format(accuracy/len(testloader)))

        

#-----after testing, change the above part to a load /save situation and load the model onto another file to perform the plotting easier-------

#plotting the probablity

model.eval()
torch.save(model, 'trained_specific_model.pth')

while True:
    #can change test loader to any loader that contains the image that we want to see 
    test_iter = iter(testloader)
    images,labels = test_iter.next()

    images,labels = images.to(device),labels.to(device)

    #get the first image(just to test, can chanage thos)
    image_current=images[0]

    image_current=image_current.view(1,3,224,224)

    with torch.no_grad():
        output = model.forward(image_current)
    
    #create the plot
    ps = torch.exp(output)
    fig, (ax1, ax2) = plt.subplots(figsize=(12,14), ncols=2)
    image_plot_helper.imshow(images[0].cpu(),ax=ax1)
    image_plot_helper.graphShow(ax=ax2,ps=ps.cpu())
    plt.show()

    #once user closes image, prompt for user input
    typed_instruction = input("Please press any key (and/or enter) to show a new random image. Or type in 'exit' to stop: ")

    if typed_instruction == 'exit':
        break
    


