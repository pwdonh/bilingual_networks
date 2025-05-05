from data import read_config, CollateItems
from main import setup_pretraining
import torch
import numpy as np
import argparse
import pandas as pd
from params import datapath, num_workers, is_cpu

langs = ['de','en','fr']

# Get args
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str,
                    help='path to config file with hyperparameters, etc.')
args = parser.parse_args()

basepath = f'{args.config_path.replace(".cfg","")}/pretraining'

langs_test, accuracies = [], []

# Load full test dataset and network checkpoint
config = read_config(args.config_path)
trainer, _, _, test_dataset = setup_pretraining(config, datapath, num_workers, is_cpu)
trainer.load_checkpoint() 

# Split test dataset by language
test_datasets = {
    'de': torch.utils.data.Subset(test_dataset, np.where(test_dataset.lang_ind==0)[0]),
    'en': torch.utils.data.Subset(test_dataset, np.where(test_dataset.lang_ind==1)[0]),
    'fr': torch.utils.data.Subset(test_dataset, np.where(test_dataset.lang_ind==2)[0])
}

# Evaluate performance by language
for lang_test in langs:

    test_dataset = test_datasets[lang_test]
    if len(test_dataset)>0:
        test_dataset.loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=5, pin_memory=True,
            num_workers=10, shuffle=False, collate_fn=CollateItems()
        )
        test_losses, test_accuracy = trainer.test(test_datasets[lang_test], set='test')
        # break
        langs_test.append(lang_test)
        accuracies.append(test_accuracy)
        print(accuracies)

# Save as dataframe
df = pd.DataFrame(dict(
    lang_test = langs_test
))
df['phone_acc'] = np.vstack(accuracies)[:,0]
df['word_acc'] = np.vstack(accuracies)[:,1]
df.to_csv(
    f'{basepath}/accuracy_per_language.csv'
)