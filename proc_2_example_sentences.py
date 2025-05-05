import torch
import pandas as pd
import joblib
import numpy as np

from params import (
    num_workers, is_cpu
)
from data import read_config, CollateItems
from main import setup_pretraining
import soundfile as sf

datapath = './data/'

sfreq = 100
num_smooth = 3

# Test monolingual model with English sentence
# & bilingual model with English + French sentence
for lang_code, lang_use in zip(['en','fren','fren'],['en','en','fr']):

    config_path = f'experiments/commonvoice_{lang_code}_phonemes_words.cfg'
    basepath = f'{config_path.replace(".cfg","")}/pretraining'

    # Setup network and dataset, load checkpoint
    config = read_config(config_path)
    config.pretraining_manifest_train = f'sentences_{lang_use}.txt'
    config.pretraining_manifest_dev = f'sentences_{lang_use}.txt'
    config.pretraining_manifest_test = f'sentences_{lang_use}.txt' # Sentences generated with Google TTS
    trainer, train_dataset, valid_dataset, test_dataset = setup_pretraining(
        config, datapath, num_workers, is_cpu
    )
    words = np.concatenate(test_dataset.Sy_word+[['']])
    phonemes = np.concatenate(test_dataset.Sy_phoneme)
    trainer.load_checkpoint()
    trainer.model.eval()

    # Load PCA
    pca = joblib.load(f'{basepath}/pca_phonemes.pickle')

    # Load sentence as input to the network
    collate_fn = CollateItems()
    batch = collate_fn([test_dataset[0]])
    inputs, targets, lengths = trainer.generate_inputs_and_targets(batch, lang_input=0)

    # Run through the network
    outputs = trainer.model(inputs, lengths)
    
    # Project hidden activity into PCA space and smooth time-series
    y = pca.transform(outputs[4].data.cpu()[0])
    L = y.shape[0]/sfreq
    t = np.arange(0,L,1/sfreq)
    for ii in range(y.shape[1]):
        y[:,ii] = np.convolve(
            y[:,ii], np.ones(num_smooth)/num_smooth, mode='same'
        )

    # Save PCA traces in dataframe
    df = pd.DataFrame(dict(
        time = t,
        word = targets[1][0].cpu(),
        phoneme = targets[0][0].cpu(),
    ))
    df[['pca_0','pca_1','pca_2']] = y
    df.to_csv(f'{basepath}/example_traces_sentence_{lang_use}.csv')

    # Compute phoneme & word probabilities, save
    p_phoneme = torch.softmax(outputs[0][0],1).data.cpu()
    p_word = torch.softmax(outputs[1][0],1).data.cpu()
    joblib.dump(dict(
        p_word = p_word,
        p_phoneme = p_phoneme,
        spec = inputs[0].cpu().numpy()[0].T
    ), f'{basepath}/example_probabilities_sentence_{lang_use}.pickle')
