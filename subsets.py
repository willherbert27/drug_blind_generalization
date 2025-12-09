import torch
import pandas as pd
from dataloader import *
from model import *
import sys
import os
import pickle
from train import train
import random
import argparse 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default=None, help='Prefix for experiment output files')
    parser.add_argument('--response_file', type=str, default=None, help='Cancer drug response dataset')
    parser.add_argument('--cancer_type', type=str, default='cancerType_dict.pkl', help='Pickle file containing dictionary of cancer types for included cell lines')
    parser.add_argument('--omics_file', type=str, default='input_data/depmap_expression_pt_filtered.txt')
    parser.add_argument('--drugs', type=str, default=None, help='File containing Morgan fingerprints for drugs')
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_cell_lines', type=int, default=500, help='Number of cell lines to test in subset experiments')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()

    with open(args.cancer_type, 'rb') as f:
        gdsc_cancer_dict = pickle.load(f)

    with open(args.omics_file, 'r') as f:
        lines = f.readlines()
        num_feats = len(lines[0].split(',')) - 1

    generator1 = torch.Generator().manual_seed(42)
    batch = args.batch_size
    learning_rate = args.lr
    save_path = args.save_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    N = args.num_cell_lines # number of each cell line to have for each drug
    seed = random.getrandbits(32) 
    random.seed(seed)

    # Create subset of drug response dataset that is N cell lines per drug
    # numCellLineSubsets is created from original gdsc data s.t. there is a minimum number of cell lines each drug has
    df = pd.read_csv(args.response_file, sep='\t', header=None) 
    unique_drugs = set(df.iloc[:,1])
    all_df_train = []
    all_df_val = []
    for drug in unique_drugs:
        curr_df = df[df.iloc[:,1]==drug]
        sample_idxs = random.sample(list(range(0,curr_df.shape[0])), N)
        unsamples_idxs = [i for i in range(0,curr_df.shape[0]) if i not in sample_idxs]
        all_df_train.append(curr_df.iloc[sample_idxs,:])
        all_df_val.append(curr_df.iloc[unsamples_idxs,:])
    final_df_train = pd.concat(all_df_train)
    final_df_val = pd.concat(all_df_val)
    final_df_val = final_df_val.iloc[random.sample(list(range(0,final_df_val.shape[0])), int(df.shape[0]*0.20)),:] # randomly sample validation dataset that is 20% of size of original, keep things consistent across N
    final_df_train.to_csv(f'gdsc_temp_{N}subset_train_{seed}.txt', sep='\t', index=None, header=None)
    final_df_val.to_csv(f'gdsc_temp_{N}subset_val_{seed}.txt', sep='\t', index=None, header=None)

    train_dataset = DrugResponseDatasetFast(f'gdsc_temp_{N}subset_train_{seed}.txt', 
                                        args.omics_file, 
                                        args.drugs, device, gdsc_cancer_dict)
    val_dataset = DrugResponseDatasetFast(f'gdsc_temp_{N}subset_val_{seed}.txt', 
                                        args.omics_file, 
                                        args.drugs, device, gdsc_cancer_dict)
    val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [0.5, 0.5], generator=generator1) # split val in half to 10/10 and given the same seed, we can retrieve the same val/test split by using a generator
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch)

    model = DrugResponseModel_Mini(num_feats)
    loss_fn = torch.nn.MSELoss()
   
    print('========================')
    print(f'Training on GDSC {save_path}')
    print(f'{batch} batch size, {learning_rate} learning rate')
    print('========================')

    train(save_path, model, args.num_epoch, train_loader, device, loss_fn, cancerType_dict=gdsc_cancer_dict, learning_rate=args.lr, val_loader=val_loader, test_loader=test_loader, save_models=True, save_model_every=10)

    # Remove temporary subset files, this can generate lots of files. Remove to save if you'd like.
    os.remove(f'gdsc_temp_{N}subset_train_{seed}.txt')
    os.remove(f'gdsc_temp_{N}subset_val_{seed}.txt')