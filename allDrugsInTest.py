import torch
import pandas as pd
from dataloader import *
from model import *
import sys
import os
from tqdm.auto import tqdm
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
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--experiment_directory', type=str, default='results', help='Specify a directory to output experimental results. This experiment generates a lot of files when run repeatedly.')
    args = parser.parse_args()
    
    base_save_path = args.save_path
    seed = random.getrandbits(32)
    random.seed(seed)

    with open(args.cancer_type, 'rb') as f:
        gdsc_cancer_dict = pickle.load(f)

    with open(args.omics_file, 'r') as f:
        lines = f.readlines()
        num_feats = len(lines[0].split(',')) - 1

    '''
    Create dataset where:
        (1) All drugs in dataset are present in the test set
        (2) Drugs in training set are half the size of the set of unique drugs (randomly chosen)
    '''
    
    curr_df = pd.read_csv(args.response_file, sep='\t', header=None)
    all_df_val = []
    all_df_test = []
    remaining = []
    val_size = int(curr_df.shape[0]*0.2) # how big we want our val split to be for 0.8/0.1/0.1 split
    unique_drugs = list(set(curr_df.iloc[:,1]))
    N = len(unique_drugs)//2 # 
    num_samples_per_drug = int(val_size/len(unique_drugs)) # spread number of samples out evenly across all drugs

    '''
    For every unique drug:
        (1) Select the subset of data corresponding to that drug
        (2) Sample the amount of cell lines that corresponds to creating val+test set that is combined 20% the size of original dataset
        (3) From (2), sample half that list of cell lines for the validation set. The rest go to the test set.
        (4) The remaining indices are held for our training set. Even if not drug-blind, the model will never see the same cell/drug pairing
    '''
    for drug in unique_drugs:
        temp_df = curr_df[curr_df.iloc[:,1]==drug]
        sample_idxs = random.sample(list(range(0,temp_df.shape[0])), num_samples_per_drug)
        val_idxs = random.sample(sample_idxs, num_samples_per_drug//2)
        test_idxs = [i for i in sample_idxs if i not in val_idxs]
        remaining_idxs = [i for i in range(0,temp_df.shape[0]) if i not in sample_idxs]
        all_df_val.append(temp_df.iloc[val_idxs,:])
        all_df_test.append(temp_df.iloc[test_idxs,:])
        remaining.append(temp_df.iloc[remaining_idxs, :])

    final_df_val = pd.concat(all_df_val)
    final_df_test = pd.concat(all_df_test)
    remaining_df = pd.concat(remaining)

    print("Test shape :", final_df_test.shape, "Val shape :", final_df_val.shape, "Train_shape :", remaining_df.shape)
    
    # select half of the unique drugs from remaining data to be the training set
    unique_drugs = list(set(remaining_df.iloc[:,1]))
    sample_idxs = random.sample(list(range(0,len(unique_drugs))), N)
    sampled_drugs = [unique_drugs[i] for i in sample_idxs]
    train_df = curr_df.loc[curr_df.iloc[:,1].isin(sampled_drugs),:]

    if not os.path.isdir(args.experiment_directory):
        os.mkdir(args.experiment_directory)

    train_df.to_csv(f'{args.experiment_directory}/gdsc_{N}uniqueDrug_allDrugsInTest_train_exp_seed{seed}.txt', sep='\t', index=False, header=False)
    final_df_val.to_csv(f'{args.experiment_directory}/gdsc_{N}uniqueDrug_allDrugsInTest_val_exp_seed{seed}.txt', sep='\t', index=False, header=False)
    final_df_test.to_csv(f'{args.experiment_directory}/gdsc_{N}uniqueDrug_allDrugsInTest_test_exp_seed{seed}.txt', sep='\t', index=False, header=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = DrugResponseDatasetFast(f'{args.experiment_directory}/gdsc_{N}uniqueDrug_allDrugsInTest_train_exp_seed{seed}.txt', 
                                        args.omics_file, 
                                        args.drugs, device, gdsc_cancer_dict)
    val_dataset = DrugResponseDatasetFast(f'{args.experiment_directory}/gdsc_{N}uniqueDrug_allDrugsInTest_val_exp_seed{seed}.txt', 
                                        args.omics_file, 
                                        args.drugs, device, gdsc_cancer_dict)
    test_dataset = DrugResponseDatasetFast(f'{args.experiment_directory}/gdsc_{N}uniqueDrug_allDrugsInTest_test_exp_seed{seed}.txt', 
                                        args.omics_file, 
                                        args.drugs, device, gdsc_cancer_dict)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    model = DrugResponseModel_Mini(num_feats)
    loss_fn = torch.nn.MSELoss()
   
    save_path = f'{base_save_path}_{N}_{seed}'
    print('========================')
    print(f'Training on GDSC {save_path}')
    print(f'{args.batch_size} batch size, {args.lr} learning rate')
    print('========================')
    
    train(save_path, model, args.num_epoch, train_loader, device, loss_fn, cancerType_dict=gdsc_cancer_dict, learning_rate=args.lr, val_loader=val_loader, test_loader=test_loader, save_models=False)