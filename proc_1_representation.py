import torch
import pandas as pd
from sklearn.decomposition import PCA
import joblib
import numpy as np
import os

from params import langs, nanfiller

def phoneme_representations(state_dict_path, langs, layer='phone_linear', pca=None, pca_o=None, normalize=True):

    state_dict = torch.load(state_dict_path,map_location=torch.device('cpu'))

    # Load phoneme dataframes, concatenate and add columns
    df = pd.read_csv('data/phonemes.csv',index_col=0)
    df['select'] = ~df['ipa'].isna()
    lang_index = ~np.isin(df['lang'],langs)
    df.loc[lang_index,'select'] = False
    df['obstruents'] = df['select'] & ((df.manner=='plosive')|(df.manner=='fricative')|(df.manner=='affricate'))

    h_phone = state_dict[f'{layer}.weight']
    if normalize:
        h_phone = h_phone / torch.norm(h_phone,dim=1)[:,None]
    h_phone = h_phone.cpu().numpy()[:-1]

    # PCA of all valid phonemes
    if pca is None:
        pca = PCA(n_components=3).fit(h_phone[df['select']])
    df[['pca_0','pca_1','pca_2']] = pca.transform(h_phone)

    # PCA of all obstruents
    if pca_o is None:
        pca_o = PCA(n_components=3).fit(h_phone[df['obstruents']])
    df[['pca_o_0','pca_o_1','pca_o_2']] = pca_o.transform(h_phone)

    df = df.fillna(nanfiller)

    return df, pca, pca_o

if __name__=='__main__':

    # Main models for Figure 2
    for lang_code in ['en','fren','defren']:

        print(lang_code)

        # Load state dict of trained model
        basepath = f'experiments/commonvoice_{lang_code}_phonemes_words/pretraining'
        state_dict_path = os.path.join(basepath,'model_state.pth')
        # Compute PCA of phoneme readout vectors
        df, pca, pca_o = phoneme_representations(state_dict_path, langs[lang_code], layer='phone_linear')
        # Save fitted PCA for use in other scripts
        joblib.dump(pca, f'{basepath}/pca_phonemes.pickle')
        joblib.dump(pca_o, f'{basepath}/pca_obstruents.pickle')
        # Save 3d phoneme representation dataframe
        df.to_csv(f'{basepath}/phonemes.csv')

    # Alternative models for Figure S4
    for mtype in ['homologes_words','words']:

        print(mtype)

        basepath = f'experiments/commonvoice_fren_{mtype}/training'
        basepath_out = basepath.replace('training','pretraining')
        state_dict_path = os.path.join(basepath,'model_state.pth')
        df, pca, pca_o = phoneme_representations(state_dict_path, ['fr','en'], layer='rnn_phone_linear')
        joblib.dump(pca, f'{basepath_out}/pca_phonemes.pickle')
        joblib.dump(pca_o, f'{basepath_out}/pca_obstruents.pickle')
        df.to_csv(f'{basepath_out}/phonemes.csv')