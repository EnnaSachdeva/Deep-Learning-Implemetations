import torch
from data_generator import train_data_size, test_data_size, filename

#from vae_1 import model1
#from vae_2 import model2

from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
import csv
import random

#device = torch.device("cuda")

from vae_1 import model1_file
from vae_2 import model2_file


data_size = train_data_size + test_data_size

device = torch.device("cuda")
random.seed(100)

input_dim = 2
latent_dim = 1

epochs = 20


from torch.distributions import uniform

distribution = uniform.Uniform(torch.Tensor([0.5]),torch.Tensor([100.0]))


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_dim, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 200)
        self.fc4 = nn.Linear(200, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))


    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 2))
        z = self.reparameterize(mu, logvar)
        return z, self.decode(z), mu, logvar



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 2), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


'''
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(input_dim, 256)
        self.output = nn.Linear(256, 1)

    def forward(self, x):
        # Pass the input tensor through each of the operations
        # Hidden layer with sigmoid activation
        x = F.sigmoid(self.hidden(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)
        return x


model = Network()
'''


N, D_in, H, D_out = 64, 2, 100, 1

model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        )

if __name__ == '__main__':

    # Loading trained models

    model1 = VAE()
    model1.load_state_dict(torch.load(model1_file))
    model1 = model1.to(device)
    model1.eval()

    model2 = VAE()
    model2.load_state_dict(torch.load(model2_file))
    model2 = model2.to(device)
    model2.eval()

    #print(input1, input2, input3, tolearn)

    #input_dim = 28 * 28
    #batch_size = 32
    #transform = transforms.Compose([transforms.ToTensor()])
    #mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)
    #dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True, num_workers=2)

    #print('Number of samples: ', len(mnist))

    input_dim = 2*1

    # Load data from file

    inputs = []

    for line in open(filename, "r"):
        values = [float(s) for s in line.split()]
        # Normalize Data
        normalizing_factor = 150
        inputs.append([values[0] / normalizing_factor, values[1] / normalizing_factor,
                       values[2] / normalizing_factor, values[3] / normalizing_factor,
                       values[4] / normalizing_factor])

    train_data = inputs[0:train_data_size]

    test_data = inputs[train_data_size:data_size]


# dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True, num_workers=2)
# print('Number of samples: ', len(mnist))
#encoder = Encoder(input_dim, 100, 100)  # dim_in, hidden_layers, dim_out
#decoder = Decoder(latent_dim, 100, input_dim)
#encoder = encoder.cuda()
#decoder = decoder.cuda()
#criterion = nn.MSELoss()
#optimizer = optim.Adam(vae.parameters(), lr=0.0001)
l = None
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print("Training for 1st operation....")
for epoch in range(epochs):
    model.train()
    train_loss = 0
    # shuffle the data
    #print("TRAINING--------------", inputs1[0:train_data_size])
    np.random.shuffle(train_data)
    for i in range(train_data_size):
        #a = distribution.sample(torch.Size([1]))
        #a1 = input1[i]
        #out1 = input2[i]
        # Normalize data points
        #normalize_a1 = a1
        #normalize_out1 = out1
        #c = torch.cat((a, out1), 0)
        #inputs = c
        #inputs1 = torch.tensor([normalize_a1, normalize_out1])
        a = [train_data[i][0], train_data[i][1]]
        input = a
        input = torch.tensor(input)
        input = input.to(device)
        latent_rep1, _, _, _ = model1.forward(input)

        b = [train_data[i][2], train_data[i][3]]
        input = b
        input = torch.tensor(input)
        input = input.to(device)
        latent_rep2, _, _, _ = model2.forward(input)
        latent_rep1 = latent_rep1.cpu().detach().numpy()
        latent_rep2 = latent_rep2.cpu().detach().numpy()
        normalize = 20
        input = torch.tensor([latent_rep1/normalize, latent_rep2/normalize])
        input = input.resize_(1, latent_dim * 2)
        input = input.to(device)

        optimizer.zero_grad()
        output = model.forward(input)


        c = train_data[i][4]
        label = c
        label = label/normalize
        label = torch.tensor(label)
        label = label.to(device)
        #latent_space, dec = vae(input1)
        loss = criterion(output, label)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    #print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / train_data_size))
    #if epoch % epochs == 0:
    #    torch.save(model1, 'oper1')
        #print("Mean", vae.z_mean.data, "Sigma", vae.z_sigma.data)
    # print("Our model: \n\n", vae, '\n')
    # print("The state dict keys: \n\n", vae.state_dict().keys())
    else:# when training is done, carry on with Testing/Validation/Inference
        # shuffle the data
        #print("TESTING--------------", inputs1[train_data_size: test_data_size - 1])
        # print("TRAINING--------------", inputs1[0:train_data_size])
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        model.eval()
        test_loss = 0
        accuracy = 0
        # Turn off gradients for validation as it saves memory and computation
        with torch.no_grad():
            for i in range(test_data_size):
                # Flatten Images images pixels to a vector
                a = [test_data[i][0], test_data[i][1]]
                input = a
                input = torch.tensor(input)
                #a = input[0]
                input = input.to(device)
                latent_rep1, _, _, _ = model1.forward(input)

                b = [test_data[i][2], test_data[i][3]]
                input = b
                input = torch.tensor(input)
                input = input.to(device)
                latent_rep2, _, _, _ = model2.forward(input)
                latent_rep1 = latent_rep1.cpu().detach().numpy()
                latent_rep2 = latent_rep2.cpu().detach().numpy()
                input = torch.tensor([latent_rep1/normalize, latent_rep2/normalize])
                input = input.resize_(1, latent_dim * 2)
                input = input.to(device)


                output = model.forward(input)

                c = test_data[i][4]
                label = c
                label = label / normalize
                label = torch.tensor(label)
                #label = label.double()
                label = label.to(device)
                # latent_space, dec = vae(input1)
                loss = criterion(output, label)


                test_loss += loss.item()

        print("INPUT: ",  [a[0]*normalizing_factor, b[0]*normalizing_factor], "Expected: ", (a[0]+b[0])*normalizing_factor, "OUTPUT: ", output*normalizing_factor*normalize)
        print("====> Epoch: {}/{}.. ".format(epoch + 1, epochs),
              "Training Loss: {:.3f}.. ".format(train_loss / train_data_size),
              "Test Loss: {:.3f}.. ".format(test_loss / test_data_size),
              #"Test Accuracy: {:.3f}.. ".format(accuracy / test_data_size)
              )
