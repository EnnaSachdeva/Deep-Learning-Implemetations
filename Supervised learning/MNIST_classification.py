import torch
# through nn module, pytorch provides losses such as cross entropy loss
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import helper
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ])


# Download and load the training data: Divide the data into training and testing set
# Batchsize: number of images we get from data loader in one iteration and is passed through our network
# shuffle: shuffle the data set every time we start going through the data loader again
trainset = datasets.MNIST('~/.pytorch/MNIST_data', download = True, train= True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST('~/.pytorch/MNIST_data', download = True, train= False, transform=transform)
testloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

'''
# How each image looks like
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
'''


# Defining a Neural Networks
'''
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        self.output = nn.Linear(256, 10)

        def forward(self, x):
            # Pass the input tensor through each of the operations
            # Hidden layer with sigmoid activation
            x = F.sigmoid(self.hidden(x))
            # Output layer with softmax activation
            x = F.softmax(self.output(x, dim=1))
            return x


# Text representation of the model
model = Network()
'''


# Training a Neural Network Model
'''
    Making a forward pass through the network
    Use the network output to calculate the risk
    Perform a backward pass through the network with loss.backward() to calculate gradients
    Take step with optimizer to update the weights
'''
# Building the NN model
'''
 this outputs the scores of each class
 784: input units: fully connected/dense network- one vector of 28*28 images
 256: hidden units
 10: output classes
'''
# This model uses probabilities as input to the loss function

model = nn.Sequential(nn.Linear(784, 128), # linear transformation: matrix multiplication
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss() # negative log likelihood loss

optimizer = optim.SGD(model.parameters(), lr=0.003)

images, labels = next(iter(trainloader))

epochs = 50

train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten Images images pixels to a vector
        images = images.view(images.shape[0], -1)

        # Clear the gradients as the gradients are accumulated (summing up in each training step) by default in pytorch
        optimizer.zero_grad()

        output = model.forward(images)
        loss = criterion(output, labels)
        #print('Gradients before Backward pass: \n', model[0].weight.grad)
        loss.backward()
        #print('Gradients after Backward pass: \n', model[0].weight.grad)  # weight gradients after the loss function

        # To update weights using gradients calculated above
        optimizer.step()
        #print('Updated weights -', model[0].weight)
        running_loss += loss.item()

    else:  # when training is done, carry on with Testing/Validation/Inference

        test_loss = 0
        accuracy = 0

        # Turn off gradients for validation as it saves memory and computation
        with torch.no_grad():
            for images, labels in testloader:
                # Flatten Images images pixels to a vector
                images = images.view(images.shape[0], -1)

                logprob = model(images)
                test_loss += criterion(logprob, labels)
                prob = torch.exp(logprob)  # converting it into probabilities from log_prob

                # top_p is the probabilities value and the top_class is the class
                top_p, top_class = prob.topk(1, dim=1)  # topk gives the class with the highest probab

                # top_class is a 2D tensor of shape (64*1) and labels is a 1D tensor of size 64, so making the same shapes
                # equals give 0 or 1 if they match or not
                equals = top_class == labels.view(*top_class.shape)

                # accuracy = sum of 1's/total numbers
                # torch.mean does not work for byte tensor (equals), so changing that to flaot_tensor
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
              "Test Accuracy: {:.3f}.. ".format(accuracy / len(testloader))
              )







# TESTING: Check predictions of an image with the trained model
img = images[0].view(1, 784)

# Turn off gradients to speed up this part
with torch.no_grad():
    logits = model.forward(img)

# Output of the network are logits, need to take softmax for probabilities
ps = F.softmax(logits, dim=1)
helper.view_classify(img.view(1,28, 28), ps)

# Plot the losses
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
