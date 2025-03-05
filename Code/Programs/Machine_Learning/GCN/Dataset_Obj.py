'''
This contains the class containing all datasets used in this research
'''
#imports
import numpy as np
import torch
torch.manual_seed(42)
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import os
from tqdm import tqdm
from torchvision import transforms
import copy
import ast 

#dependencies
from Programs.Data_Processing.Render import joint_connections_m_hip
from Programs.Machine_Learning.GCN.Utilities import get_gait_segments
import Programs.Data_Processing.HCF as HCF


class JointDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None, joint_connections = joint_connections_m_hip,
                  cycles = True, meta = 5, preset_cycle = None, interpolate = True, ensemble = False, class_loc = 2, num_classes = 9):
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
        self.ensemble = ensemble
        self.preset_cycle = preset_cycle
        self.interpolate = interpolate
        self.class_loc = class_loc
        self.n_c = num_classes
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
    

    def set_gait_cycles(self, data):
        '''
        Sets an unsegmented set of joints to the same shape as an already segmented set of gait cycles

        Arguments
        ---------
        data: List(List)
            unsegmented joint data       
        Returns
        -------
        List(List())
            The original data in the shape of the preset gait cycles.
        '''
        new_cycles = []
        data_iter = 0
        total_size_of_preset = 0
        for c in self.preset_cycle:
            total_size_of_preset += len(c)
        for i, cycle in enumerate(self.preset_cycle):
            new_cycle = []
            for j, frame in enumerate(cycle):
                new_cycle.append(data[data_iter])
                data_iter += 1
            new_cycles.append(new_cycle)
        return new_cycles

    def process(self):
        '''
        Overloaded process function, this function processes dataset objects passed as the dataset to be contained into Torch format and saves them
        as .pt files

        Returns
        -------
        None
        '''
        #read in the raw data
        self.data = pd.read_csv(self.raw_paths[0], header=None)
        self.data = convert_to_literals(self.data)

        #Get the joint adjacency matrix
        coo_matrix = get_COO_matrix(self.joint_connections)

        #Extract the gait segment for the data points
        if self.preset_cycle == None:
            self.base_cycles = get_gait_segments(self.data.to_numpy())
            counter = 0
            for cycle in self.base_cycles:
                for frame in cycle:
                    counter +=1
        else:
        #Alternatively, use a preset if provided
            self.base_cycles = self.set_gait_cycles(self.data.to_numpy())

        self.data_cycles = HCF.sample_gait_cycles(copy.deepcopy(self.base_cycles), self.n_c, self.class_loc)
        self.num_nodes_per_graph = len(self.data.columns) - self.meta - 1

        #Find the largest cycle
        t = 0
        self.max_cycle = 0
        for i, d in enumerate(self.data_cycles):
            t += len(d)
            if len(d) > self.max_cycle:
                self.max_cycle = len(d)
        
        #Save the data points after converting them to graphs
        self.data = pd.DataFrame(self.data_cycles)
        self.num_cycles = len(self.data_cycles)
        for index, row in enumerate(tqdm(self.data_cycles)):
            self.cycle_indices.append(len(self.data_cycles[index]))
            #Start from scratch with coo matrix every cycle as cycles will be different lengths
            coo_matrix = get_COO_matrix(self.joint_connections)
            mod_coo_matrix = self.modify_COO_matrix(len(row), self.joint_connections, coo_matrix)
            #Convert the data into pytorch format 
            data = data_to_graph(row, mod_coo_matrix, class_loc = self.class_loc)
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

    def modify_COO_matrix(self, gait_cycle_length, connections, current_matrix):
        '''
        This continually adds new sub-graphs to the COO matrix to thread together all the frames
        of a gait cycle into one graph

        Arguments
        ---------
        gait_cycle_length: int
            the length of the current cycle to be added     
        connections: List(Tuple())
            The indices of the joint connections
        current_matrix:
            The current main graph to add the new subgraph to
        Returns
        -------
        List(List())
            The updated COO matrix
        '''
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
def data_to_graph(row, coo_matrix, meta = 5, class_loc = 2):
    '''
    Transforms graph objects into Pytorch objects

    Arguments
    ---------
    row: List()
        the frames added to the graph
    coo_matrix: List(Tuple())
        coo_maxtrix of the graph
    meta: int (optional, default = 5)
        The amount of metadata to be expected

    Returns
    -------
    Pytorch.Data
        The graph as a Pytorch Data object

    '''
    gait_cycle = []
    y_arr = []
    per_arr = []
    for cycle_part in row:
        refined_row = cycle_part[meta + 1 :]   
        #refined_row = refined_row[-7:]
        row_as_array = np.array(refined_row)  
        y = int(cycle_part[class_loc])  
        per_arr = int(cycle_part[5])
        #Remove certain parts of the body, just leave the legs 

        y_arr.append(y)
        #print("y val: ", y)
        if len(gait_cycle) <= 0:
            gait_cycle = row_as_array
        else:   
            gait_cycle = np.concatenate((gait_cycle, row_as_array), axis=0)
    #print("gait cycle shape: ", len(gait_cycle), len(gait_cycle[0]))
    #done = 5/0
    #Pass entire cycle as single graph
    data = Data(x=torch.tensor(list(gait_cycle), dtype=torch.float),
        y=torch.tensor([y], dtype=torch.long),
        edge_index=torch.tensor(coo_matrix, dtype=torch.long),
        person = per_arr
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
        '''
        Overloaded process function, this function processes dataset objects passed as the dataset to be contained into Torch format and saves them
        as .pt files

        Returns
        -------
        None
        '''
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
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))    
        return data
    
#Input here would be each row
def data_to_tensor(row):
    '''
    Transforms HCF data rows into Pytorch objects

    Arguments
    ---------
    row: List()
        the frames added to the datapoint

    Returns
    -------
    Pytorch.Data
        The data as a Pytorch Data object

    '''
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