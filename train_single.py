import torch
import pandas as pd
from dataloader import *
from model import *
import sys
import os
from tqdm.auto import tqdm
import pickle
from train import train
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default=None, help='Prefix for experiment output files')
    parser.add_argument('--response_file', type=str, default=None, help='Cancer drug response dataset')
    parser.add_argument('--cancer_type', type=str, default='violet_data/violet_data_backup/gcsi_cancerType_dict.pkl', help='Pickle file containing dictionary of cancer types for included cell lines')
    parser.add_argument('--omics_file', type=str, default='violet_data/clean_omics_data/depmap_expression_pt_filtered.txt')
    parser.add_argument('--drugs', type=str, default=None, help='File containing Morgan fingerprints for drugs')
    parser.add_argument('--val_file', type=str, help='Premade validation set for experiments requiring it')
    parser.add_argument('--test_file', type=str, help='Premade test split file for experiments requiring it')
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()
    
    # Can reuse gcsi dict with GDSC bc I have them both saved as ACH- type cell lines
    with open(args.cancer_type, 'rb') as f:
        cancer_dict = pickle.load(f)

    with open(args.omics_file, 'r') as f:
            lines = f.readlines()
            num_feats = len(lines[0].split(',')) - 1

    generator1 = torch.Generator().manual_seed(42)
    save_path = args.save_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.val_file is None and args.test_file is None:
        response_dataset = DrugResponseDatasetFast(args.response_file, 
                                            args.omics_file, 
                                            args.drugs, device, cancerTypes=cancer_dict)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(response_dataset, [0.8, 0.1, 0.1], generator=generator1)  
        loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    elif (args.val_file is not None and args.test_file is None) or (args.val_file is None and args.test_file is not None):
        raise RuntimeError('If providing premade validation and test sets, both most be provided. You have only provided one of them.')
    else:
        print('*****************************')
        print('Using custom validation and test set instead of random splits. Ensure they are non-overlapping!')
        print('*****************************')
        response_dataset = DrugResponseDatasetFast(args.response_file, 
                                            args.omics_file, 
                                            args.drugs, device, cancerTypes=cancer_dict)
        loader = torch.utils.data.DataLoader(response_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataset = DrugResponseDatasetFast(args.val_file, 
                                            args.omics_file, 
                                            args.drugs, device, cancerTypes=cancer_dict)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
        test_dataset = DrugResponseDatasetFast(args.test_file, 
                                            args.omics_file, 
                                            args.drugs, device, cancerTypes=cancer_dict)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    
    model = DrugResponseModel_Mini(num_feats)
    loss_fn = torch.nn.MSELoss()
    print('Using device: ', device)
    print('========================')
    print(f'Training on GCSI {save_path}')
    print(f'{args.response_file} train/val, {args.batch_size} batch size, {args.learning_rate} learning rate')
    print('========================')
    train(save_path, model, args.num_epoch, loader, device, loss_fn, cancerType_dict=cancer_dict, learning_rate=args.lr, val_loader=val_loader, test_loader=test_loader, save_models=args.save_model)