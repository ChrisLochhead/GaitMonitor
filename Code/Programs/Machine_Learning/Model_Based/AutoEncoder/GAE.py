import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import torch
from torch.utils.data import random_split
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import Programs.Data_Processing.Model_Based.Utilities as Utilities

class VariationalEncoder(nn.Module):
    def __init__(self, dim_in, dim_h, latent_dims, heads=[8,1]):  
        super(VariationalEncoder, self).__init__()

        dim_half = int(dim_h/2)
        self.dim_half = dim_half
        self.dim_in = dim_in
        self.gcn1 = GCNConv(dim_in, dim_half)


        self.linear1 = nn.Linear(dim_half, latent_dims)
        self.linear3 = nn.Linear(latent_dims, latent_dims)
        #self.linear4 = nn.Linear(latent_dims, latent_dims)

        # hack to get sampling on the GPU
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0
        #########################################################

    def forward(self, x, edge_index, batch):
        x = x.to("cuda")
        edge_index = edge_index.to("cuda")
        batch = batch.to("cuda")
        #print("x going in: ", x.shape, self.dim_in, self.dim_half)
        #print("gcn", self.gcn1)
        g1 = F.leaky_relu(self.gcn1(x, edge_index))
        #print("1: ", g1.shape)
        l1 = F.leaky_relu(self.linear1(g1))
       # print("2: ", l1.shape)
        l3 = self.linear3(l1)
       # print("3: ", l3.shape)

        #mu =  self.linear4(l3)
        #sigma = torch.exp(self.linear4(l3))
        #z = mu + sigma*self.N.sample(mu.shape)
        #self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return l3 #z      



class Decoder(nn.Module):
    
    def __init__(self, latent_dims, out, dim_in, dim_h, heads=[8,1]):
        super().__init__()

        dim_half = int(dim_h/2)
        self.linear1 = nn.Linear(latent_dims, latent_dims)
        self.linear3 = nn.Linear(latent_dims, dim_half)
        self.gcn2 = GCNConv(dim_half, dim_in)

        
    def forward(self, x, edge_index, batch):
        x = x.to("cuda")
        edge_index = edge_index.to("cuda")
        batch = batch.to("cuda")

        #print("output ", x.shape)
        #This is 64,9
        l1 = F.leaky_relu(self.linear1(x))
        #print("after l3: ", l1.shape)

        l3 = F.leaky_relu(self.linear3(l1))
        #print("after l1", l3.shape)

        g2 = F.sigmoid(self.gcn2(l3, edge_index))
        #print("after g1: ", g2.shape)

        return g2

    


class VariationalAutoencoder(nn.Module):
    def __init__(self, dim_in, dim_h, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        #Default heads
        self.encoder = VariationalEncoder(dim_in, dim_h, latent_dims)
        self.decoder = Decoder(latent_dims, latent_dims, dim_in, dim_h)

    def forward(self, x, edge_index, batch):
        x = x.to(device)
        z = self.encoder(x, edge_index, batch)
        return self.decoder(z, edge_index, batch)
    


def tensor_to_csv(data, labels, skeleton_size = 17):
    final_data = []

    for i, batch in enumerate(data): # each element of data is the output results of a batch...
        #Append each node individually to the row
        skeleton_iter = 0
        skeleton_count = 0
        final_row = []
        for j, arr in enumerate(batch):
            if skeleton_iter == 0:
                #print("skeleton size: ", skeleton_size, len(arr))
                final_row = [0,0, labels[i][skeleton_count].item()]

            if skeleton_iter < skeleton_size:
                final_row.append(arr.tolist())
                skeleton_iter += 1
            else:
                final_row.append(arr.tolist())
                skeleton_iter = 0
                skeleton_count += 1
                final_data.append(final_row)
    
    #print("should have 1960 examples: ", skeleton_count, skeleton_iter, len(data), batch_sizes)
    
    return final_data
        


### Training function
def train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for i, x in enumerate(dataloader): 
        # Move tensor to the proper device
        #print("x data: ", x)
        x = x.to(device)
        x_hat = vae(x.x, x.edge_index, x.batch)
        # Evaluate loss
        loss = ((x.x - x_hat)**2).sum()# + vae.encoder.kl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        #if i % 100 == 0:
        #    print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss+=loss.item()

    return train_loss / len(dataloader.dataset)
        
### Testing function
def test_epoch(vae, device, dataloader, joint_output, skeleton_size):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    encoded_dataset = []
    labels = []
    with torch.no_grad(): # No need to track the gradients
        for x in dataloader:
            # Move tensor to the proper device
            x = x.to(device)
            # Encode data
            encoded_data = vae.encoder(x.x, x.edge_index, x.batch)
            encoded_dataset.append(encoded_data)
            labels.append(x.y)
            #print("x.y: ", x.y)
            #print("tensor: ", encoded_data)
            # Decode data
            x_hat = vae(x.x, x.edge_index, x.batch)
            loss = ((x.x - x_hat)**2).sum()# + vae.encoder.kl
            val_loss += loss.item()

    #Save lower dimensional embedding as it's own dataset
    Utilities.save_dataset(tensor_to_csv(encoded_dataset, labels, skeleton_size=skeleton_size), joint_output)
    return val_loss / len(dataloader.dataset)

def plot_ae_outputs(encoder,decoder,n=10, test_dataset=None):
    plt.figure(figsize=(16,4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()  
