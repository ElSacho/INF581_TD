import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import sys

import numpy as np
import matplotlib.pyplot as plt

info = {
        # TODO replace the following with your own
        'Email' : 'sacha.braun@polytechnique.edu',
        'Alias' : 'Sacho', # optional
}


# Récupération de l'argument entré par l'utilisateur
if len(sys.argv) != 2:
    print("Usage: python3 main.py <number>")
    sys.exit(1)

try:
    number = int(sys.argv[1])
    if number < 0 or number > 9:
        raise ValueError()
except ValueError:
    print("Le nombre doit être compris entre 0 et 9")
    sys.exit(1)

batch_size = 128

data_dir = 'data'
# MNIST dataset
dataset = torchvision.datasets.MNIST(root=data_dir,
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor()),
    batch_size=10, shuffle=False)

    # Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a directory if not exists
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Hyper-parameters
image_size = 784
h_dim = 400
z_dim = 20
num_epochs = 15
learning_rate = 1e-3
n_classes = 10
beta = 10.

def label_hot_encoding(labels,nb_digits = n_classes):
    return F.one_hot(torch.tensor(np.array(labels)), num_classes=nb_digits)

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    
def plot_reconstruction(model, n=24):
    x,_ = next(iter(data_loader))
    x = x[:n,:,:,:].to(device)
    try:
        out, _, _, log_p = model(x.view(-1, image_size)) 
    except:
        out, _, _ = model(x.view(-1, image_size)) 
    x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
    out_grid = torchvision.utils.make_grid(x_concat).cpu().data
    show(out_grid)

def plot_generation(model, n=24):
    with torch.no_grad():
        z = torch.randn(n, z_dim).to(device)
        out = model.decode(z).view(-1, 1, 28, 28)

    out_grid = torchvision.utils.make_grid(out).cpu()
    show(out_grid)

def plot_conditional_generation(model, n=8, z_dim=2, fix_number=None):
    with torch.no_grad():
        matrix = np.zeros((n,n_classes))
        matrix[:,0] = 1

        if fix_number is None:
            final = matrix[:]
            for i in range(1,n_classes):
                final = np.vstack((final,np.roll(matrix,i)))
            z = torch.randn(8*n_classes, z_dim).to(device)
            y_onehot = torch.tensor(final).type(torch.FloatTensor).to(device)
            concat_input = torch.cat([z, y_onehot], 1)
            out = model.decode(z,y_onehot).view(-1, 1, 28, 28)
        else:
            z = torch.randn(n, z_dim).to(device)
            y_onehot = torch.tensor(np.roll(matrix, fix_number)).type(torch.FloatTensor).to(device)
            out = model.decode(z,y_onehot).view(-1, 1, 28, 28)

    out_grid = torchvision.utils.make_grid(out).cpu()
    return out_grid

class CVAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20, n_classes = 10):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim + n_classes, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, l_onehot):
        merge = torch.cat((z, l_onehot), dim=1)
        h = F.relu(self.fc4(merge))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x, l_onehot):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z, l_onehot)
        return x_reconst, mu, log_var


if os.path.exists("model_cvae.pth"):
    model_cvae = CVAE().to(device)
    model_cvae.load_state_dict(torch.load("model_cvae.pth"))
else:
    model_cvae = CVAE().to(device)
    optimizer_cvae = torch.optim.Adam(model_cvae.parameters(), lr=learning_rate)

    # Start training
    for epoch in range(num_epochs):
        for i, (x, labels) in enumerate(data_loader):
            # Forward pass
            x = x.to(device).view(-1, image_size)
            x_reconst, mu, log_var = model_cvae(x, label_hot_encoding(labels))
            
            # Compute reconstruction loss and kl divergence
            reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
            kl_div =  - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Backprop and optimize
            loss = reconst_loss + beta*kl_div
            optimizer_cvae.zero_grad()
            loss.backward()
            optimizer_cvae.step()
        
            if (i+1) % 10 == 0:
                print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                    .format(epoch+1, num_epochs, i+1, len(data_loader), reconst_loss.item()/batch_size, kl_div.item()/batch_size))
                
    torch.save(model_cvae.state_dict(), "model_cvae.pth")


img = plot_conditional_generation(model_cvae, z_dim=z_dim, fix_number=number)
# Enregistrement de l'image au format PDF
filename = "generation_number_{}.pdf".format(number)
filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
plt.figure()
plt.imshow(img.permute(1, 2, 0))
plt.axis('off')
plt.savefig(filepath, bbox_inches='tight', pad_inches=0)

print("L'image a été enregistrée sous le nom", filename)