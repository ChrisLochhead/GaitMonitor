import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch_geometric.loader import DataLoader as GeoLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch_geometric.nn import GCNConv, global_add_pool, GINConv, GATv2Conv

from sklearn.manifold import TSNE
import plotly.express as px
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import Programs.Data_Processing.Model_Based.Utilities as Utilities
import Programs.Machine_Learning.Model_Based.GCN.Graph_Nets as GN

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


# define a cross validation function
def cross_valid(MY_model, test_dataset, criterion=None,optimizer=None,dataset=None,k_fold=5, batch = 64):
    
    train_score = []
    val_score = []
    test_score = []

    total_size = len(dataset)
    fraction = 1/k_fold
    seg = int(total_size * fraction)
    # tr:train,val:valid; r:right,l:left;  eg: trrr: right index of right side train subset 
    # index: [trll,trlr],[vall,valr],[trrl,trrr]
    for i in range(k_fold):
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size
        # msg
#         print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)" 
#               % (trll,trlr,trrl,trrr,vall,valr))
        
        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))
        
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall,valr))

        train_set = dataset[train_indices]
        val_set = dataset[val_indices]
        test_set = test_dataset
        #train_set = torch.utils.data.dataset.Subset(dataset,train_indices)
        #val_set = torch.utils.data.dataset.Subset(dataset,val_indices)
        #print("types here: ", type(train_set), type(val_set))
        
#         print(len(train_set),len(val_set))
#         print()
        model = GN.GAT(dim_in = dataset.num_node_features, dim_h=128, dim_out=3)
        model = model.to("cuda")
            
        train_loader = GeoLoader(train_set, batch_size=batch,
                                          shuffle=True)
        val_loader = GeoLoader(val_set, batch_size=batch,
                                          shuffle=True)
        test_loader = GeoLoader(test_set, batch_size=batch,
                                          shuffle=True)
    
        
        model, embeddings, losses, accuracies, outputs, vals, tests = GN.train(model, train_loader, val_loader, test_loader)
        train_score.append(accuracies[1])
        #val_acc = valid(res_model,criterion,optimizer,val_loader)
        val_score.append(vals[-1])
        test_score.append(tests[-1])
    
    return train_score,val_score, test_score
        
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



def show_image(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

'''def run_VAE(train_loader, val_loader, test_dataset):

    ### Set the random seed for reproducible results
    d = 9
    vae = VariationalAutoencoder(dim_in=3, dim_h=16, latent_dims=d)
    lr = 1e-3 
    optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)
    print(f'Selected device: {device}')
    vae.to(device)
    vae.eval()
    num_epochs = 50

    for epoch in range(num_epochs):
        train_loss = train_epoch(vae,device,train_loader,optim)
        val_loss = test_epoch(vae,device,val_loader)
        print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))

    with torch.no_grad():

        # sample latent vectors from the normal distribution
        latent = torch.randn(128, d, device=device)

        # reconstruct images from the latent vectors
        img_recon = vae.decoder(latent)
        img_recon = img_recon.cpu()

        fig, ax = plt.subplots(figsize=(20, 8.5))
        show_image(torchvision.utils.make_grid(img_recon.data[:100],10,5))
        plt.show()


    encoded_samples = []
    for sample in tqdm(test_dataset):
        img = sample[0].unsqueeze(0).to(device)
        label = sample[1]
        # Encode image
        vae.eval()
        with torch.no_grad():
            encoded_img  = vae.encoder(img)
        # Append to list
        encoded_img = encoded_img.flatten().cpu().numpy()
        encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
        encoded_sample['label'] = label
        encoded_samples.append(encoded_sample)
        
    encoded_samples = pd.DataFrame(encoded_samples)
    encoded_samples


    px.scatter(encoded_samples, x='Enc. Variable 0', y='Enc. Variable 1', color=encoded_samples.label.astype(str), opacity=0.7)

    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(encoded_samples.drop(['label'],axis=1))

    fig = px.scatter(tsne_results, x=0, y=1, color=encoded_samples.label.astype(str),labels={'0': 'tsne-2d-one', '1': 'tsne-2d-two'})
    fig.show()
'''