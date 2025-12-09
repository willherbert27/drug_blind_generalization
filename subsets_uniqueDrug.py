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
    parser.add_argument('--cancer_type', type=str, default='cancerType_dict.pkl', help='Pickle file containing dictionary of cancer types for included cell lines')
    parser.add_argument('--omics_file', type=str, default='input_files/depmap_expression_pt_filtered.txt')
    parser.add_argument('--drugs', type=str, default=None, help='File containing Morgan fingerprints for drugs')
    parser.add_argument('--train_file', type=str, default=None, help='Cancer drug response dataset')
    parser.add_argument('--val_file', type=str, help='Premade validation set for experiments requiring it')
    parser.add_argument('--test_file', type=str, help='Premade test split file for experiments requiring it')
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_drugs', type=int, help='Number of drugs to include in training set')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()
    
    if args.num_drugs is None:
        raise RuntimeError('You must provide the number of unique drugs to include in the training set for this experiment.')
    if args.val_file is None or args.test_file is None:
        raise RuntimeError('You must provide a constant validation and test set for this experiment.')
    
    seed = random.getrandbits(32) 
    random.seed(seed)
    N = args.num_drugs
   
    with open(args.cancer_type, 'rb') as f:
        gdsc_cancer_dict = pickle.load(f)

    with open(args.omics_file, 'r') as f:
        lines = f.readlines()
        num_feats = len(lines[0].split(',')) - 1

    # Create training dataset by selecting N unique drugs from training split
    train_df = pd.read_csv(args.train_file, sep='\t', header=None)
    unique_drugs = list(set(train_df.iloc[:,1]))
    print(len(unique_drugs))
    
    drugs_for_train = random.sample(unique_drugs, N)
    final_df_train = train_df.loc[train_df.iloc[:,1].isin(drugs_for_train),:]
    if not os.path.isdir('uniqueDrug_drugBlind_constantTest'):
        os.mkdir('uniqueDrug_drugBlind_constantTest')
    final_df_train.to_csv(f'uniqueDrug_drugBlind_constantTest/gdsc_uniqueDrug_train_{seed}.txt', sep='\t', index=None, header=None)

    generator1 = torch.Generator().manual_seed(42)
    batch = args.batch_size
    learning_rate = args.lr
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = DrugResponseDatasetFast(f'uniqueDrug_drugBlind_constantTest/gdsc_uniqueDrug_train_{seed}.txt', 
                                        args.omics_file, 
                                        args.drugs, device, gdsc_cancer_dict)
    val_dataset = DrugResponseDatasetFast(args.val_file, 
                                        args.omics_file, 
                                        args.drugs, device, gdsc_cancer_dict)
    test_dataset = DrugResponseDatasetFast(args.test_file, 
                                        args.omics_file, 
                                        args.drugs, device, gdsc_cancer_dict)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch)

    num_epoch = args.num_epochs
    model = DrugResponseModel_Mini(num_feats)
    loss_fn = torch.nn.MSELoss()
   
    save_path = f'{args.save_path}_{N}_{seed}'
    print('========================')
    print(f'Training on GDSC {save_path}')
    print(f'{batch} batch size, {learning_rate} learning rate')
    print('========================')
    
    train(save_path, model, num_epoch, train_loader, device, loss_fn, cancerType_dict=gdsc_cancer_dict, learning_rate=args.lr, val_loader=val_loader, test_loader=test_loader, save_models=False)