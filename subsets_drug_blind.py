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
    parser.add_argument('--cancer_type', type=str, default='violet_data/violet_data_backup/gcsi_cancerType_dict.pkl', help='Pickle file containing dictionary of cancer types for included cell lines')
    parser.add_argument('--omics_file', type=str, default='violet_data/clean_omics_data/depmap_expression_pt_filtered.txt')
    parser.add_argument('--drugs', type=str, default=None, help='File containing Morgan fingerprints for drugs')
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_cell_lines', type=int, default=500, help='Number of cell lines to test in subset experiments')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--track_training_composition', type=bool, default=False, help='Enable to keep training composition files for prediction')
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
    
    N = args.num_cell_lines
    seed = random.getrandbits(32) 
    random.seed(seed)

    # response file is the subset file we created during preprocessing
    df = pd.read_csv(args.response_file, sep='\t', header=None)

    # we need to presplit into training test and validation for each replicate due to drug-blind condition
    unique_drugs = list(set(df.iloc[:,1]))
    sample_idxs = random.sample(list(range(0,len(unique_drugs))), int(0.80*len(unique_drugs))) # 80% size training set
    sampled_drugs = [unique_drugs[i] for i in sample_idxs]
    val_drugs_presplit = [drug for drug in unique_drugs if drug not in sampled_drugs] # 20% validation set, before splitting into test
    val_idx = random.sample(list(range(0,len(val_drugs_presplit))), int(0.5*len(val_drugs_presplit))) 
    val_drugs = [val_drugs_presplit[i] for i in val_idx]
    test_drugs = [drug for drug in val_drugs_presplit if drug not in val_drugs]
    sampled_df = df.loc[df.iloc[:,1].isin(sampled_drugs),:]
    val_df = df.loc[df.iloc[:,1].isin(val_drugs),:]
    test_df = df.loc[df.iloc[:,1].isin(test_drugs)]

    sampled_df.to_csv(f'gdsc_drugBlind_train_exp_seed{seed}.txt', sep='\t', index=False, header=False)
    val_df.to_csv(f'gdsc_drugBlind_val_exp_seed{seed}.txt', sep='\t', index=False, header=False)
    test_df.to_csv(f'gdsc_drugBlind_test_exp_seed{seed}.txt', sep='\t', index=False, header=False)

    # Select N cell lines for each unique drug in each of train/val/test
    curr_df = pd.read_csv(f'gdsc_drugBlind_train_exp_seed{seed}.txt', sep='\t', header=None)
    curr_unique_drugs = set(curr_df.iloc[:,1])
    all_df_train = [] 
    for drug in curr_unique_drugs:
        temp_df = curr_df[curr_df.iloc[:,1]==drug]
        sample_idxs = random.sample(list(range(0,temp_df.shape[0])), N)
        all_df_train.append(temp_df.iloc[sample_idxs,:])

    final_df_train = pd.concat(all_df_train)

    curr_df = pd.read_csv(f'gdsc_drugBlind_val_exp_seed{seed}.txt', sep='\t', header=None)
    curr_unique_drugs = set(curr_df.iloc[:,1])
    all_df_val = []  
    for drug in curr_unique_drugs:
        temp_df = curr_df[curr_df.iloc[:,1]==drug]
        sample_idxs = random.sample(list(range(0,temp_df.shape[0])), N)
        all_df_val.append(temp_df.iloc[sample_idxs,:])

    final_df_val = pd.concat(all_df_val)

    curr_df = pd.read_csv(f'gdsc_drugBlind_test_exp_seed{seed}.txt', sep='\t', header=None)
    curr_unique_drugs = set(curr_df.iloc[:,1])
    all_df_test = [] 
    for drug in curr_unique_drugs:
        temp_df = curr_df[curr_df.iloc[:,1]==drug]
        sample_idxs = random.sample(list(range(0,temp_df.shape[0])), N)
        all_df_test.append(temp_df.iloc[sample_idxs,:])

    final_df_test = pd.concat(all_df_test)

    final_df_train.to_csv(f'gdsc_temp_{N}subset_drugBlind_train.txt', sep='\t', index=None, header=None)
    final_df_val.to_csv(f'gdsc_temp_{N}subset_drugBlind_val.txt', sep='\t', index=None, header=None)
    final_df_test.to_csv(f'gdsc_temp_{N}subset_drugBlind_test.txt', sep='\t', index=None, header=None)
                         
    generator1 = torch.Generator().manual_seed(42)

    train_dataset = DrugResponseDatasetFast(f'gdsc_temp_{N}subset_drugBlind_train.txt', 
                                        args.omics_file, 
                                        args.drugs, device, gdsc_cancer_dict)
    val_dataset = DrugResponseDatasetFast(f'gdsc_temp_{N}subset_drugBlind_val.txt', 
                                        args.omics_file, 
                                        args.drugs, device, gdsc_cancer_dict)
    test_dataset = DrugResponseDatasetFast(f'gdsc_temp_{N}subset_drugBlind_test.txt', 
                                        args.omics_file, 
                                        args.drugs, device, gdsc_cancer_dict)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch)

    num_epoch = args.num_epochs
    model = DrugResponseModel_Mini(num_feats)
    loss_fn = torch.nn.MSELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('========================')
    print(f'Training on GDSC {save_path}')
    print(f'{batch} batch size, {learning_rate} learning rate')
    print('========================')

    train(save_path, model, num_epoch, train_loader, device, loss_fn, cancerType_dict=gdsc_cancer_dict, learning_rate=args.lr, val_loader=val_loader, test_loader=test_loader, save_models=True)

    # Remove temporary split files. Enable track training composition argument to keep them
    if not args.track_training_composition:
        os.remove(f'gdsc_drugBlind_train_exp_seed{seed}.txt')
        os.remove(f'gdsc_drugBlind_val_exp_seed{seed}.txt')
        os.remove(f'gdsc_drugBlind_test_exp_seed{seed}.txt')
    
    os.remove(f'gdsc_temp_{N}subset_drugBlind_train.txt')
    os.remove(f'gdsc_temp_{N}subset_drugBlind_val.txt')
    os.remove(f'gdsc_temp_{N}subset_drugBlind_test.txt')

