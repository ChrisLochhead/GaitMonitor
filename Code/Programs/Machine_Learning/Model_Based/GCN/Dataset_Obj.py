import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import os
from tqdm import tqdm
from torchvision import transforms
import Programs.Data_Processing.Model_Based.Render as Render
import Programs.Data_Processing.Model_Based.HCF as HCF
import Programs.Data_Processing.Model_Based.Dataset_Creator as Creator
import copy

class JointDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None, joint_connections = Render.joint_connections_m_hip,
                  cycles = False, meta = 5, person = None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.joint_connections = joint_connections
        self.cycles = cycles
        self.num_cycles = 0
        self.num_nodes_per_graph = 0
        self.meta = meta
        self.cycle_indices = []
        self.person = person
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
        
        print("\ndata being read from: ", self.raw_paths[0])
        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]
        

    def download(self):
        pass
               

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0], header=None)
        self.data = convert_to_literals(self.data)

        if self.person != None:
            self.data = self.data[self.data.iloc[:, 5] == self.person]

            instances = []
            for index, row in self.data.iterrows():
                if row[0] not in instances:
                    instances.append(row[0])
            
            print("instances: ", len(instances), instances)

        coo_matrix = get_COO_matrix(self.joint_connections)

        if self.cycles:
            #Full cycles per instance
            #self.data_cycles = HCF.split_by_instance(self.data.to_numpy())
            #Several cycles per instance
            print("data length: ", len(self.data))
            self.data_cycles = HCF.get_gait_cycles(self.data.to_numpy(), None)
            #self.data_cycles = HCF.alternate_get_gait_cycles(self.data.to_numpy(), None)
            self.data_cycles = HCF.sample_gait_cycles(self.data_cycles)
            self.data_cycles = HCF.normalize_gait_cycle_lengths(self.data_cycles)
            self.data_cycles = Creator.interpolate_gait_cycle(self.data_cycles, None)

            print("here's the cycles: ", len(self.data_cycles))

            self.num_nodes_per_graph = len(self.data.columns) - self.meta - 1

            t = 0
            self.max_cycle = 0
            for i, d in enumerate(self.data_cycles):
                t += len(d)

                if len(d) > self.max_cycle:
                    self.max_cycle = len(d)
            

            self.data = pd.DataFrame(self.data_cycles)
            self.num_cycles = len(self.data_cycles)
            for index, row in enumerate(tqdm(self.data_cycles, total=len(self.data_cycles[0]))):
                
                self.cycle_indices.append(len(self.data_cycles[index]))
                    
                #Start from scratch with coo matrix every cycle as cycles will be different lengths
                coo_matrix = get_COO_matrix(self.joint_connections)
                mod_coo_matrix = self.modify_COO_matrix(len(row), self.joint_connections, coo_matrix)
                # Featurize molecule
                data = data_to_graph(row, mod_coo_matrix)
                if self.test:
                    torch.save(data, 
                        os.path.join(self.processed_dir, 
                                    f'data_test_{index}.pt'))
                else:
                    torch.save(data, 
                        os.path.join(self.processed_dir, 
                                    f'data_{index}.pt'))
                
                t_data = torch.load(os.path.join(self.processed_dir, 
                        f'data_{index}.pt')) 
            
                 
        else:

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
            

    def len(self):
        return self.data.shape[0]

    def get(self, idx):

        if self.cycles:
            frame_count = 0
            for i, c in enumerate(self.cycle_indices):
                if idx <= frame_count:
                    true_indice = i
                    break
                else:
                    frame_count += c

    
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))    

        return data

    def modify_COO_matrix(self, gait_cycle_length, connections, current_matrix):
        #Get number of joints per graph:
        #17 connections is mid-hip appended full graph
        current_matrix = copy.deepcopy(current_matrix)
        max_value = 0
        #Quickly get number of nodes in graph as this can vary depending on joints
        for c in connections:
            if c[0] > max_value:
                max_value = c[0]
            if c[1] > max_value:
                max_value = c[1]

        graph_size = max_value

        for i in range(gait_cycle_length):
            #Don't append connections to the last frame as there's nothing after it
            if i != gait_cycle_length - 1: # -1 to ignore the last one
                for node_index in range(graph_size):
                    #Add connections pointing to the next graph
                    for j in range(0, 3):
                        current_matrix[0] += [node_index + (((i + 1) * graph_size) + 1), node_index]
                        current_matrix[1] += [node_index , node_index + (((i + 1) * graph_size) + 1)]


            #After connections between graphs are added, add a new matrix onto the bottom of another graph
            if i != 0: # Ignore the first one: the original coo matrix already has it's first self-connections
                for connection in connections:
                    for i in range(0, 3):
                        current_matrix[0] += [connection[0] + (((i + 1) * graph_size) + 1), connection[1] + (((i + 1) * graph_size) + 1)]
                        current_matrix[1] += [connection[1] + (((i + 1) * graph_size) + 1), connection[0] + (((i + 1) * graph_size) + 1)]

                
        return current_matrix

def get_COO_matrix(connections):
  res = [[],[]]
  for connection in connections:
    #Once for each of the 3 coords in each connection
    for i in range(0, 3):
        res[0] += [connection[0], connection[1]]
        res[1] += [connection[1], connection[0]]
  return res
        
import ast 
import copy 

def convert_to_literals(data, meta = 5):
    for i,  (index, row) in enumerate(data.iterrows()):
        for col_index, col in enumerate(row):
            if col_index >= meta + 1 and type(row[col_index]) != np.float64:
                tmp = ast.literal_eval(row[col_index])
                data.iat[i, col_index] = copy.deepcopy(tmp)
            else:
                data.iat[i, col_index] = data.iat[i, col_index]

    return data
            
#Input here would be each row
def data_to_graph(row, coo_matrix, meta = 5):
    
    #The single instance per row case (No gait cycles, row is a single frame)
    if isinstance(row[0], np.ndarray) == False and isinstance(row[0], list) == False:
        refined_row = row.iloc[meta + 1:]
        node_f= refined_row

        #This is standard Data that has edges
        row_as_array = np.array(node_f.values.tolist())
        #Turn into one-hot vector
        y = int(row.iloc[2])

        data = Data(x=torch.tensor(row_as_array, dtype=torch.float),
                    y=torch.tensor(y, dtype=torch.long),
                    edge_index=torch.tensor(coo_matrix, dtype=torch.long),
                    #This isn't needed
                    #edge_attr=torch.tensor(edge_attr,dtype=torch.float)
                    )
        
        return data
    #The multi-instance case (with gait cycles, row is a full cycle of frames or varying length)
    else:
        gait_cycle = []
        y_arr = []
        for cycle_part in row:
            refined_row = cycle_part[meta + 1 :]   
            row_as_array = np.array(refined_row)  

            y = int(cycle_part[2])  
            y_arr.append(y)
            if len(gait_cycle) <= 0:
                gait_cycle = row_as_array
            else:   
                #print("len gait: ", len(gait_cycle), len(gait_cycle[0]), len(row_as_array), len(row_as_array[0]))
                #print("gait 1: ", gait_cycle, type(gait_cycle), type(gait_cycle[0]), type(gait_cycle[0][0]))
                #print("gait 2: ", row_as_array, type(row_as_array), type(row_as_array[0]), type(row_as_array[0][0]))

                gait_cycle = np.concatenate((gait_cycle, row_as_array), axis=0)
                #stop = 5/0 

        #Verify gait cycle is calculated correctly:
        #Pass entire cycle as single graph
        data = Data(x=torch.tensor(list(gait_cycle), dtype=torch.float),
            y=torch.tensor([y], dtype=torch.long),
            edge_index=torch.tensor(coo_matrix, dtype=torch.long),
            #This isn't needed
            #edge_attr=torch.tensor(edge_attr,dtype=torch.float)
            )
        
        return data







































class HCFDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None,
                  cycles = False, cycle_preset = None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.cycles = cycles
        self.num_cycles = 0
        self.cycle_indices = []
        self.cycle_preset = cycle_preset
        self.max_cycle = 1
        self.num_nodes_per_graph = 1
        super(HCFDataset, self).__init__(root, transform, pre_transform)
        
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
        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]
        

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0], header=None)
        self.data = convert_to_literals(self.data)

        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule
            data = data_to_tensor(row)

            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                f'data_test_{index}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                f'data_{index}.pt'))

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        if self.cycles:
            frame_count = 0
            for i, c in enumerate(self.cycle_indices):
                if idx <= frame_count:
                    true_indice = i
                    break
                else:
                    frame_count += c

    
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))    

        return data
    
    
    def modify_COO_matrix(self, gait_cycle_length, connections, current_matrix):
        #Get number of joints per graph:
        #17 connections is mid-hip appended full graph
        current_matrix = copy.deepcopy(current_matrix)
        max_value = 0
        #Quickly get number of nodes in graph as this can vary depending on joints
        for c in connections:
            if c[0] > max_value:
                max_value = c[0]
            if c[1] > max_value:
                max_value = c[1]

        graph_size = max_value

        for i in range(gait_cycle_length):
            #Don't append connections to the last frame as there's nothing after it
            if i != gait_cycle_length - 1: # -1 to ignore the last one
                for node_index in range(graph_size):
                    #Add connections pointing to the next graph
                    for j in range(0, 3):
                        current_matrix[0] += [node_index + (((i + 1) * graph_size) + 1), node_index]
                        current_matrix[1] += [node_index , node_index + (((i + 1) * graph_size) + 1)]


            #After connections between graphs are added, add a new matrix onto the bottom of another graph
            if i != 0: # Ignore the first one: the original coo matrix already has it's first self-connections
                for connection in connections:
                    for i in range(0, 3):
                        current_matrix[0] += [connection[0] + (((i + 1) * graph_size) + 1), connection[1] + (((i + 1) * graph_size) + 1)]
                        current_matrix[1] += [connection[1] + (((i + 1) * graph_size) + 1), connection[0] + (((i + 1) * graph_size) + 1)]

                
        return current_matrix
    
#Input here would be each row
def data_to_tensor(row):
    #The single instance per row case (No gait cycles, row is a single frame)
    if isinstance(row[0], np.ndarray) == False and isinstance(row[0], list) == False:
        refined_row = row.iloc[3:]
        node_f= refined_row

        #This is standard Data that has edge shit
        row_as_array = np.array(node_f.values.tolist())

        #Turn into one-hot vector
        y = int(row.iloc[2])

        data = Data(x=torch.tensor([row_as_array], dtype=torch.float),
                    y=torch.tensor([y], dtype=torch.long))
                
        return data
    #The multi-instance case (with gait cycles, row is a full cycle of frames or varying length)
    else:
        gait_cycle = []
        y_arr = []
        for cycle_part in row:
            refined_row = cycle_part[3:]    
            row_as_array = np.array(refined_row)     
            y = int(cycle_part[2])  

            y_arr.append(y)
            if len(gait_cycle) <= 0:
                gait_cycle = row_as_array
            else:   
                gait_cycle = np.concatenate((gait_cycle, row_as_array), axis=0)

        
        #Verify gait cycle is calculated correctly:
        #Pass entire cycle as single graph
        data = Data(x=torch.tensor(list(gait_cycle), dtype=torch.float),
            y=torch.tensor([y], dtype=torch.long))
        
        return data