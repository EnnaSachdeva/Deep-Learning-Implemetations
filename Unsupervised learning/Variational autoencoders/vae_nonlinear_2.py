import torch
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

device = torch.device("cuda")
random.seed(100)

filename = "data.txt"

input_dim = 2
latent_dim = 1

epochs = 5
train_data_size = 10000
test_data_size = 500
data_size = train_data_size + test_data_size

from torch.distributions import uniform

distribution = uniform.Uniform(torch.Tensor([0.5]),torch.Tensor([100.0]))



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_dim, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, input_dim)

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



# Load data from file
for line in open(filename, "r"):
    values = [float(s) for s in line.split()]
    inputs1.append(torch.tensor([values[0], values[1]))
    inputs2.append(torch.tensor([values[2], values[3]))
    tolearn.append(torch.tensor(values[4])


# Normalize data
normalize_a11 = a11 / 50
normalize_a12 = a12 / 50

normalize_a21 = a21 / 50
normalize_a22 = a22 / 50

normalize_out = out / 50


if __name__ == '__main__':


    #print(input1, input2, input3, tolearn)

    #input_dim = 28 * 28
    #batch_size = 32
    #transform = transforms.Compose([transforms.ToTensor()])
    #mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)
    #dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True, num_workers=2)

    #print('Number of samples: ', len(mnist))

    input_dim = 2*1
    # dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True, num_workers=2)

    # print('Number of samples: ', len(mnist))

    #encoder = Encoder(input_dim, 100, 100)  # dim_in, hidden_layers, dim_out
    #decoder = Decoder(latent_dim, 100, input_dim)

    #encoder = encoder.cuda()
    #decoder = decoder.cuda()

    #criterion = nn.MSELoss()
    #optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    l = None
    inputs = []

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Training for 1st operation....")
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        # create data_set

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



            input1 = inputs1[i]
            input1 = input1.to(device)

            optimizer.zero_grad()

            latent_rep, recon_batch, mu, logvar = model(input1)
            #latent_space, dec = vae(input1)
            loss = loss_function(recon_batch, input1, mu, logvar)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()


        #print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / train_data_size))

        if epoch % epochs == 0:
            model1 = model
            torch.save(model1, 'oper1')
            #print("Mean", vae.z_mean.data, "Sigma", vae.z_sigma.data)

        # print("Our model: \n\n", vae, '\n')
        # print("The state dict keys: \n\n", vae.state_dict().keys())


        else:# when training is done, carry on with Testing/Validation/Inference
            model.eval()
            test_loss = 0
            accuracy = 0

            # Turn off gradients for validation as it saves memory and computation
            with torch.no_grad():
                for i in range(test_data_size):
                    # Flatten Images images pixels to a vector
                    input1 = inputs1[i+train_data_size]
                    input1 = input1.to(device)
                    latent_rep, recon_batch, mu, logvar = model(input1)

                    test_loss += loss_function(recon_batch, input1, mu, logvar).item()

            print("INPUT: ", input1, "OUTPUT: ", recon_batch, "LATENT REP: ", latent_rep)
            print("====> Epoch: {}/{}.. ".format(epoch + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss / train_data_size),
                  "Test Loss: {:.3f}.. ".format(test_loss / test_data_size),
                  #"Test Accuracy: {:.3f}.. ".format(accuracy / test_data_size)
                  )
