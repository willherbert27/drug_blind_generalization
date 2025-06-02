import torch
import pandas as pd
import numpy as np
from dataloader import *
from model import *
import sys
import copy
import os
from tqdm.auto import tqdm
import pickle
from sklearn.model_selection import StratifiedKFold

def train(save_path, model, num_epoch, train_loader, device, loss_fn, learning_rate=0.0001, cancerType_dict=None, scheduler=None,val_loader=None,test_loader=None,save_models=True, save_model_every=None, save_results_every=20):
    
    # Results file cleanup if rerunning - I prefer to custom save and plot results than use TensorBoard
    if not os.path.isdir('results'):
        os.mkdir('results')
    if not os.path.isdir('results/cancer_type_corr'):
        os.mkdir('results/cancer_type_corr')
    if os.path.exists(f'results/{save_path}.txt'.format(save_path)):
        os.remove(f'results/{save_path}.txt'.format(save_path))
    
    model.to(device)
    running_loss = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_vloss = np.inf

    for epoch in range(num_epoch):
        better = False
        model.train()
        running_loss = 0
        for i,data in enumerate(tqdm(train_loader, leave=True)):
            
            data = [elem for i,elem in enumerate(data)]  
            cell, drug, response, lineName, drugName = data

            model.train()
            optimizer.zero_grad()

            pred_response = model(cell, drug)
            
            loss = loss_fn(pred_response.float(), response.float())
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            # print and record loss statistics every desired step
            if i % save_results_every == (save_results_every-1):
                # print('True', response)
                # print('Predicted', pred_response)
                loss_line = 'epoch {} batch {} average loss: {}'.format(epoch+1, i+1, running_loss/save_results_every)
                with open('./results/{}.txt'.format(save_path), 'a') as f:
                    f.write('{}\t{}\n'.format(i+1, running_loss/save_results_every))
                tqdm.write(loss_line)
                running_loss = 0

        if val_loader is not None:
            model.eval()
            running_vloss = 0
            
            with torch.no_grad():
                all_responses = []
                all_pred_responses = []
                all_cellLines = []
                all_drugs = []
                for j,data in enumerate(tqdm((val_loader), leave=True)):
            
                    data = [elem for i,elem in enumerate(data)]  
                    cell, drug, response, lineName, drugName = data
                    pred_response = model(cell, drug)
                    vloss = loss_fn(pred_response.float(), response.float())
                    running_vloss += vloss.item()
                
                avg_vloss = running_vloss / (j+1)

                # Save test performance at best iteration of model
                if avg_vloss < best_vloss:
                    epochSinceLastImprovement = 0
                    better = True
                    print('Saving test performance...')
                    for k,data in enumerate(tqdm((test_loader), leave=True)):
            
                        data = [elem for i,elem in enumerate(data)]  
                        cell, drug, response, lineName, drugName = data
                        pred_response = model(cell, drug)
                        all_responses.extend(response.float())
                        all_pred_responses.extend(pred_response.float())
                        all_cellLines.extend(lineName)
                        all_drugs.extend(drugName)

                    best_vloss = avg_vloss
                    if os.path.exists('./results/cancer_type_corr/{}.txt'.format(save_path)):
                        os.remove('./results/cancer_type_corr/{}.txt'.format(save_path))
                    with open('./results/cancer_type_corr/{}.txt'.format(save_path), 'a') as f:
                        f.write(f'epoch: {epoch}\n')
                        for i, entry in enumerate(all_pred_responses):
                            if cancerType_dict is not None:
                                f.write('{}\t{}\t{}\t{}\t{}\n'.format(all_cellLines[i], all_drugs[i], cancerType_dict[all_cellLines[i]], entry, all_responses[i]))
                            else:
                                f.write('{}\t{}\t{}\t{}\n'.format(all_cellLines[i], all_drugs[i], entry, all_responses[i]))
                # else:
                #     epochSinceLastImprovement += 1
                #     if epochSinceLastImprovement >= 10:
                #         earlyStopping = True
                #         with open('./results/{}.txt'.format(save_path), 'a') as f:
                #             f.write('STOPPING EARLY ON EPOCH {}\n'.format(epoch))
                
                if save_models:
                    if epoch == 0:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss
                        }, './models/{}.model'.format(save_path))
                    elif better:
                        best_vloss = avg_vloss
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss
                        }, './models/{}.model'.format(save_path))

                    if save_model_every is not None:
                        if (epoch+1) % save_model_every == 0:
                            torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss
                        }, './models/{}_epoch{}.model'.format(save_path,epoch+1))
                print('epoch {} valid LOSS  {}'.format(epoch, avg_vloss))
                with open('./results/{}.txt'.format(save_path), 'a') as f:
                    f.write('epoch {} valid LOSS  {}\n'.format(epoch, avg_vloss))

    if scheduler is not None:
        scheduler.step()
        print(scheduler.get_last_lr())

    del model

# Function for finetuning model trained on one dataset on another dataset
def finetune(save_path, ckpt, num_epoch, train_loader, val_loader, device, loss_fn, cancerType_dict, scheduler=None):
    
    # load model ckpt
    if not os.path.isdir('results/finetuning'):
        os.mkdir('results/finetuning')
    if os.path.exists('results/finetuning/{}.txt'.format(save_path)):
        os.remove('results/finetuning/{}.txt'.format(save_path))

    model_path = '{}'.format(ckpt)
    print('Loading model from {}'.format(model_path))
    checkpoint = torch.load('{}'.format(ckpt))
    num_feats = checkpoint['model_state_dict']['full_cell_embed.0.weight'].shape[1]
    model = DrugResponseModel_Mini(num_feats)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_vloss = np.inf
    # Freeze weights of first layer of drug input and cell input
    # model.cell_embed1.weight.requires_grad = False
    # model.cell_embed1.bias.requires_grad = False
    # model.drug_embed1.bias.requires_grad = False
    # model.drug_embed1.weight.requires_grad = False

    for epoch in range(num_epoch):
        model.train()
        running_loss = 0
        for i,data in enumerate(tqdm(train_loader, leave=True)):
            
            data = [elem.to(device) if i < 3 else elem for i,elem in enumerate(data)]   
            cell, drug, response, lineName, drugName = data

            response = response.unsqueeze(1)

            model.train()
            optimizer.zero_grad()

            pred_response = model(cell, drug)
            
            loss = loss_fn(pred_response.float(), response.float())
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            # record loss statistics every 10 steps
            if i % 20 == 19:
                # print('True', response)
                # print('Predicted', pred_response)
                loss_line = 'epoch {} batch {} average loss: {}'.format(epoch+1, i+1, running_loss/20)
                with open('./results/fine-tuning/{}.txt'.format(save_path), 'a') as f:
                    f.write('{}\t{}\n'.format(i+1, running_loss/20))
                tqdm.write(loss_line)
                running_loss = 0

        model.eval()
        running_vloss = 0
        
        
        with torch.no_grad():
            all_responses = []
            all_pred_responses = []
            all_cellLines = []
            all_drugs = []
            for j,data in enumerate(tqdm((val_loader), leave=True)):
            
                data = [elem.to(device) if i < 3 else elem for i,elem in enumerate(data)]  
                cell, drug, response, lineName, drugName = data
                response = response.unsqueeze(1)
                pred_response = model(cell, drug)
                all_responses.extend(response.float())
                all_pred_responses.extend(pred_response.float())
                all_cellLines.extend(lineName)
                all_drugs.extend(drugName)
                vloss = loss_fn(pred_response.float(), response.float())
                running_vloss += vloss.item()
                
            
            avg_vloss = running_vloss / (j+1)
            
            for j,data in enumerate(tqdm((val_loader), leave=True)):
                if avg_vloss < best_vloss:
                    best_vloss = avg_vloss
                    if os.path.exists('./results/fine-tuning/cancer_type_corr/{}_finetuned.txt'.format(save_path)):
                        os.remove('./results/fine-tuning/cancer_type_corr/{}_finetuned.txt'.format(save_path))
                    with open('./results/fine-tuning/cancer_type_corr/{}_finetuned.txt'.format(save_path), 'a') as f:
                        for i, entry in enumerate(all_pred_responses):
                            f.write('{}\t{}\t{}\t{}\t{}\n'.format(all_cellLines[i], all_drugs[i], cancerType_dict[all_cellLines[i]], entry, all_responses[i]))

            # if epoch == 0:
            #     best_vloss = avg_vloss
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'loss': loss
            #     }, './models/{}_finetuned_gCSI.model'.format(ckpt))
            # elif avg_vloss < best_vloss:
            #     best_vloss = avg_vloss
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'loss': loss
            #     }, './models/{}_finetuned_gCSI.model'.format(ckpt))
            print('epoch {} valid LOSS  {}'.format(epoch, avg_vloss))
            with open('./results/fine-tuning/{}.txt'.format(save_path), 'a') as f:
                f.write('epoch {} valid LOSS  {}\n'.format(epoch, avg_vloss))
        if scheduler is not None:
            scheduler.step()
            print(scheduler.get_last_lr())     
    
    del model
    torch.cuda.empty_cache()

if __name__ == '__main__':
    
    print('Training function is run from another file, likely train_single.py')