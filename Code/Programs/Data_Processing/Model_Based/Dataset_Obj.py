import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import os
from tqdm import tqdm
from Programs.Data_Processing.Model_Based.Utilities import joint_connections

class JointDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        super(JointDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0], header=None)
        print("data being read from: ", self.raw_paths[0], type(self.raw_paths[0]))
        print(self.raw_paths[0])
        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]
        

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0], header=None)#.reset_index()
        self.data = convert_to_literals(self.data)
        coo_matrix = get_COO_matrix()

        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule
            data = data_to_graph(row, coo_matrix)

            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_test_{index}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_{index}.pt'))
            

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))        
        return data

def get_COO_matrix():
  res = [[],[]]
  for connection in joint_connections:
    #Once for each of the 3 coords in each connection
    for i in range(0, 3):
        res[0] += [connection[0], connection[1]]
        res[1] += [connection[1], connection[0]]
  return res

import ast 
import copy 

def convert_to_literals(data):
    for i,  (index, row) in enumerate(data.iterrows()):
        for col_index, col in enumerate(row):
            if col_index >= 3:
                tmp = ast.literal_eval(row[col_index])
                data.iat[i, col_index] = copy.deepcopy(tmp)
            else:
                data.iat[i, col_index] = int(data.iat[i, col_index])

    return data
            
#Input here would be each row
def data_to_graph(row, coo_matrix):
    refined_row = row.iloc[3:]
    node_f= refined_row

    #This is standard Data that has edge shit
    row_as_array = np.array(node_f.values.tolist())

    #Turn into one-hot vector
    y = int(row.iloc[2])

    data = Data(x=torch.tensor(row_as_array, dtype=torch.float),
                y=torch.tensor([y], dtype=torch.long),
                edge_index=torch.tensor(coo_matrix, dtype=torch.long),
                #This isn't needed
                #edge_attr=torch.tensor(edge_attr,dtype=torch.float)
                )
    return data