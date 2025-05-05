import torch
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

from params import datapath, num_workers
from data import read_config, get_datasets
# from main import setup_pretraining
import models
from training import Trainer
from proc_4_decodability import feature_decodability, features

config_path = 'experiments/commonvoice_sequential.cfg'
langs = ['en','fr']
inits = ['none','en19','en49','fr19','fr49']

# Get args
parser = argparse.ArgumentParser()
parser.add_argument('--step', type=str, default='accuracy')
parser.add_argument('--run', type=int, default=1)
args = parser.parse_args()

nums_samples = np.hstack([
    np.logspace(np.log10(500),np.log10(100000),40).astype(int),
    np.ones(20).astype(int)*100000
])

config = read_config(config_path)
np.random.seed(8) # for subselecting validation set

# Generate datasets for different languages
train_datasets, valid_datasets = {}, {}
for lang in ['en','fr','fren']:
    # Take subsample of dev data
    df_manifest = pd.read_csv(os.path.join(
        datapath,f'commonvoice_speakers_{lang}_dev.txt'
    ),header=None)
    indices = np.random.permutation(df_manifest.index)[:7000] 
    all_files = df_manifest.loc[indices,0].values.tolist()
    small_manifest = os.path.join(datapath,f'commonvoice_speakers_{lang}_dev_small.txt')
    with open(small_manifest,'w') as f:
        for file in all_files:
            f.write(file+'\n')
    # Generate datasets
    train_dataset, valid_dataset, test_dataset = get_datasets(config, datapath,
        f'commonvoice_speakers_{lang}_train.txt',
        small_manifest,
        f'commonvoice_speakers_{lang}_test.txt', num_workers)
    train_datasets[lang] = train_dataset
    valid_datasets[lang] = valid_dataset

tqdmout = open('/dev/null', 'w')
df_accuracy = pd.DataFrame(columns = [
    'lang_train', 'lang', 'level', 'epoch', 'init', 
    'num_samples', 'loss', 'accuracy', 'time'
])
dfs_decode = []
index = 0

for init in inits:

    if init=='none':
        lang_combis = ['en','fr','fren']
    else:
        lang_combis = ['fren']

    for lang in lang_combis:

        # Initialize model
        model = getattr(models, config.type)(config=config)
        # Train the model
        trainer = Trainer(model=model, config=config)

        if init!='none':
            lang_init = init[:2]
            epoch_init = init[2:]
            state_dict_path_init = f'sequential/run_{args.run}/state_dict-init_none-lang_{lang_init}-iter_{epoch_init}.pth'
            trainer.model.load_state_dict(torch.load(state_dict_path_init,map_location=torch.device('cpu')))

        print(f'======== Init {init} - Language {lang}')
        num_samples_cumsum = 0
        for epoch, num_samples in enumerate(tqdm(nums_samples)):
            state_dict_path = f'sequential/run_{args.run}/state_dict-init_{init}-lang_{lang}-iter_{epoch}.pth'
            num_samples_cumsum += num_samples
            if args.step=='train':
                train_datasets[lang].loader.sampler.num_samples = num_samples
                train_losses = trainer.train(train_datasets[lang])
                torch.save(trainer.model.state_dict(), state_dict_path)
            elif args.step=='decodability':
                df_decode = feature_decodability(state_dict_path, langs, features, num_permute=1,tqdmout=tqdmout)
                df_decode['epoch'] = epoch
                df_decode['lang_train'] = lang
                df_decode['init'] = init
                df_decode['num_samples'] = num_samples_cumsum
                dfs_decode.append(df_decode)
                pd.concat(dfs_decode).reset_index(drop=True).to_csv(
                    f'sequential/decodability_run_{args.run}.csv'
                )
            elif args.step=='accuracy':
                trainer.model.load_state_dict(torch.load(state_dict_path))
                for lang_valid in langs:
                    valid_losses, valid_accuracies = trainer.test(valid_datasets[lang_valid],tqdmout=tqdmout)
                    # Save results in dataframe
                    for loss, accuracy, level in zip(valid_losses, valid_accuracies, ['phonemes','words']):
                        df_accuracy.loc[index,'lang_train'] = lang
                        df_accuracy.loc[index,'lang'] = lang_valid
                        df_accuracy.loc[index,'level'] = level
                        df_accuracy.loc[index,'epoch'] = epoch
                        df_accuracy.loc[index,'init'] = init
                        df_accuracy.loc[index,'num_samples'] = num_samples_cumsum
                        df_accuracy.loc[index,'loss'] = loss
                        df_accuracy.loc[index,'accuracy'] = accuracy
                        index += 1
                df_accuracy.to_csv(f'sequential/accuracies_run_{args.run}.csv')