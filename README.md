# Monotherapy cancer drug-blind response prediction is limited to intraclass generalization

This repository includes code for the paper "Monotherapy cancer drug-blind response prediction is limited to intraclass generalization."

## Data preprocessing

Code for data preprocessing is provided in preprocessing_final.ipynb. This notebook generates all data files required for downstream model training.

## Model training

Most model training will be done using the script train_single.py. Example usage:

```
python train_single.py
      --save_path gdsc_ec50_base
      --response_file gdsc_ec50.txt
```

Dataset diversity experiments are performed using the subsets.py file. This has similar input to train_single but also includes a parameter for number of cell lines per drug. 
subsets_drug_blind.py performs the drug blind dataset diversity experiment.
subsets_uniqueDrug.py subsets training set by number of unique drugs present rather than number of unique cell lines and is a legacy experiment. 

When performing mechanism specific replicates, it is useful to generate the train/validation/test split beforehand. train_single.py has an input option for predefined splits. 

The drug-to-drug information sharing experiment is performed by repeatedly running the allDrugsInTest.py file. This is best done by either scheduling a cron job or looping over in a bash script. This experiment generates a unique 32-bit seed and saves it as part of the experiment file, so there is no need to provide a unique seed for replicates. Analysis of these experiments and heatmap plotting is provided in the file analysis.ipynb.

Model architecture is contained in model.py.
Dataloading methods are in dataloader.py. 
Train loop is contained in train.py. 
