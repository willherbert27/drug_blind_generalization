import torch
import pandas as pd
import numpy as np

class DrugResponseDataset(torch.utils.data.Dataset):
    def __init__(self, response_file, omics_file, smiles_mat, cancerTypes=None, minMax=False, customSmiles=False):
        with open(response_file, 'r') as f:
            lines = f.readlines()
            self.response_list = [x.strip().split('\t') for x in lines]
            response_lens = set([len(x) for x in self.response_list])
            if len(response_lens) > 1:
                print(response_lens)
                for x in self.response_list: 
                    if len(x) < 3: 
                        print(x)
                raise ValueError('Responses should all be of length 3 - Cell line, drug, IC50')
            
            if minMax:
                self.response_vals = [float(x[2]) for x in self.response_list]
                self.max_response_val = max(self.response_vals)
                self.min_response_val = min(self.response_vals)
                self.difference = self.max_response_val - self.min_response_val
                self.response_list = [[x[0], x[1], (float(x[2])-self.min_response_val)/self.difference] for x in self.response_list]
            else:
                self.response_list = [[x[0], x[1], float(x[2])] for x in self.response_list]
            # print(self.response_list)
            
        self.customSmiles = customSmiles
        self.omics = pd.read_csv(omics_file, header = 0, index_col=0)
        
        cell_lines_in_omics = list(self.omics.index)
        self.response_list = [x for x in self.response_list if x[0] in cell_lines_in_omics]
        if cancerTypes is not None:
            self.response_list = [x for x in self.response_list if x[0] in cancerTypes.keys()]
        print(f'There are {len(self.response_list)} samples in the dataset')
        self.smiles = pd.read_csv(smiles_mat, sep='\t', header=[0], index_col=[0])
        self.standardized_smiles_names = self.smiles
        self.standardized_smiles_names.index = self.standardized_smiles_names.index.str.strip('-')
        self.standardized_smiles_names.index = self.standardized_smiles_names.index.str.strip(' ')

    def __len__(self):
        return len(self.response_list)
    
    def __getitem__(self, idx):
        curr_combo = self.response_list[idx]
        cell_line, drug, response = curr_combo
        

        cell_profile = torch.Tensor(list(self.omics.loc[cell_line]))
        curr_smiles = torch.Tensor(list(self.smiles.loc[drug]))
        if self.customSmiles:
            if drug in list(self.smiles.index):
                curr_smiles = torch.Tensor(list(self.smiles.loc[drug]))
            elif drug in list(self.standardized_smiles_names.index):
                curr_smiles = torch.Tensor(list(self.standardized_smiles_names.loc[drug]))
            elif drug.lower() in list(self.smiles.index):
                curr_smiles = torch.Tensor(list(self.smiles.loc[drug.lower()]))
            elif drug.lower() in list(self.standardized_smiles_names.index):
                curr_smiles = torch.Tensor(list(self.standardized_smiles_names.loc[drug.lower()]))
            else:
                raise KeyError('Drug not present in current smiles dataset: {}'.format(drug))

        return cell_profile, curr_smiles, response, cell_line, drug
    
class DrugResponseDatasetFast(torch.utils.data.Dataset):
    def __init__(self, response_file, omics_file, smiles_mat, device, cancerTypes=None, minMax=False, customSmiles=False):
        with open(response_file, 'r') as f:
            lines = f.readlines()
            self.response_list = [x.strip().split('\t') for x in lines]
            response_lens = set([len(x) for x in self.response_list])
            if len(response_lens) > 1:
                print(response_lens)
                for x in self.response_list: 
                    if len(x) < 3: 
                        print(x)
                raise ValueError('Responses should all be of length 3 - Cell line, drug, IC50')
            
            if minMax:
                self.response_vals = [float(x[2]) for x in self.response_list]
                self.max_response_val = max(self.response_vals)
                self.min_response_val = min(self.response_vals)
                self.difference = self.max_response_val - self.min_response_val
                self.response_list = [[x[0], x[1], (float(x[2])-self.min_response_val)/self.difference] for x in self.response_list]
            else:
                self.response_list = [[x[0], x[1], float(x[2])] for x in self.response_list]
            # print(self.response_list)
            
        self.customSmiles = customSmiles
        self.omics = pd.read_csv(omics_file, header = 0, index_col=0)
        
        cell_lines_in_omics = list(self.omics.index)
        self.response_list = [x for x in self.response_list if x[0] in cell_lines_in_omics]
        if cancerTypes is not None:
            self.response_list = [x for x in self.response_list if x[0] in cancerTypes.keys()]
        print(f'There are {len(self.response_list)} samples in the dataset')
        self.smiles = pd.read_csv(smiles_mat, sep='\t', header=[0], index_col=[0])
        self.standardized_smiles_names = self.smiles
        self.standardized_smiles_names.index = self.standardized_smiles_names.index.str.strip('-')
        self.standardized_smiles_names.index = self.standardized_smiles_names.index.str.strip(' ')

        self.cell_line_dim = len(self.omics.columns)
        self.drug_dim = len(self.smiles.columns)
        self.cell_line_plus_drug = [(torch.Tensor(self.omics.loc[cell_line].to_numpy()).to(device), torch.Tensor(self.smiles.loc[drug].to_numpy()).to(device)) for cell_line,drug,_ in self.response_list]
        self.cell_drug_names = [(cell_line, drug) for cell_line,drug,_ in self.response_list]
        self.responses = [torch.Tensor(np.array([response])).to(device) for _,_,response in self.response_list]


    def __len__(self):
        return len(self.response_list)
    
    def __getitem__(self, idx):
        
        cell_profile, curr_smiles = self.cell_line_plus_drug[idx]
        response = self.responses[idx]
        cell_line,drug = self.cell_drug_names[idx]

        return cell_profile, curr_smiles, response, cell_line, drug