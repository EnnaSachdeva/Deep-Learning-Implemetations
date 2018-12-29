'''This code builds a classifier to classify cats and dogs from Kaggle-
https://www.kaggle.com/c/dogs-vs-cats
using transfer learning. It uses a pretrained model- resnet50, which has been
pre-trained on ImageNet dataset.

The implementation uses GPU during training as well as testing.

'''


import torch
# through nn module, pytorch provides losses such as cross entropy loss
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import helper
from torchvision import datasets, transforms, models  # for densenet model in this code


# Define a transform for the training set
# Introduce randomness in the image so that trained network is invariant to locations, sizes, and orientations of images

train_transform = transforms.Compose([transforms.RandomRotation(30),  # changing images to square
                                transforms.RandomResizedCrop(224),  # crops the center of image with 224 pixels on each side
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


test_transform = transforms.Compose([transforms.Resize(255),  # changing images to square
                                transforms.CenterCrop(224),  # crops the center of image with 224 pixels on each side
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

# get the data from the directory
data_dir = 'Cat_Dog_data'


# The directory 'Cat_Dog_data' contains 2 subfolders 'test' and 'train'
'''This data set consists of colored images of cats and dogs '''

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transform)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transform)


trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle = True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle = True)

'''
images, labels = next(iter(trainloader))  # alternative of for loop

fig, axes = plt.subplots(figsize=(10,4), ncols=4)


# show 4 images
for ii in range(4):
    ax = axes[ii]
    helper.imshow(images[ii], ax=ax, normalize=False)
'''


'''
# using GPU for parallel computations
# Bringing tensors to GPU
model.cuda()
images.cuda()

# Bringing tensors from GPU back to CPU
# model.cpu(), images.cpu()
'''

# Use GPU if its available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# using Transfer learning to build the classifier, using pretrained network called resnet50
# download the pre-trained network parameters and load that into the network
model = models.resnet50(pretrained=True)
#print(model)   # shows the model architecture



# In the pre-trained model, we will keep the features same as in the original model,
# but the classifier/fc (last thing in the model) has been trained on the ImageNet and not on other features.

# freeze the feature parameters so that we dont backprop through them and
# and do not update them

# Turn off gradients for our models
for param in model.parameters():
    param.requires_grad = False

classfier = nn.Sequential(nn.Linear(2048, 512),
                          nn.ReLU(),
                          nn.Dropout(0.2),
                          nn.Linear(512, 2),
                          nn.LogSoftmax(dim=1))


model.fc = classfier

criterion = nn.NLLLoss()


# Only training the classifier parameters as the feature parameters are frozen
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)


# Start the training
epochs = 1
steps = 0
running_loss = 0
print_every = 5

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        steps += 1

        # Move inputs and labels to the GPU
        images, labels = images.to(device), labels.to(device)

        # Clear the gradients as the gradients are accumulated (summing up in each training step) by default in pytorch
        optimizer.zero_grad()

        output = model.forward(images)
        loss = criterion(output, labels)
        #print('Gradients before Backward pass: \n', model[0].weight.grad)
        loss.backward()
        # print('Gradients after Backward pass: \n', model[0].weight.grad)  # weight gradients after the loss function

        # To update weights using gradients calculated above
        optimizer.step()
        # print('Updated weights -', model[0].weight)
        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()

            for images, labels in testloader:
                # Move input and labels to the GPU
                images, labels = images.to(device), labels.to(device)

                logprob = model.forward(images)
                batch_loss = criterion(logprob, labels)

                test_loss += batch_loss.item()


                # Calculate accuracy
                prob = torch.exp(logprob)  # converting it into probabilities from log_prob

                # top_p is the probabilities value and the top_class is the class
                top_p, top_class = prob.topk(1, dim=1)  # topk gives the class with the highest probab

                # top_class is a 2D tensor of shape (64*1) and labels is a 1D tensor of size 64, so making the same shapes
                # equals give 0 or 1 if they match or not
                equals = top_class == labels.view(*top_class.shape)

                # accuracy = sum of 1's/total numbers
                # torch.mean does not work for byte tensor (equals), so changing that to flaot_tensor
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                  "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
                  "Test Accuracy: {:.3f}.. ".format(accuracy / len(testloader))
                  )

            running_loss = 0
            model.train()


