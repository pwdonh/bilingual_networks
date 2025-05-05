import torch
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut
import numpy as np
import os
from tqdm import tqdm
import sys

features = ['sonority','manner','voicing','place']

def feature_decodability(state_dict_path, langs_use, features, layer='phone_linear', num_permute=50,tqdmout=sys.stderr):

    state_dict = torch.load(state_dict_path,map_location=torch.device('cpu'))
    h_phone = state_dict[f'{layer}.weight'].numpy()[:-1]

    df = pd.read_csv('data/phonemes.csv',index_col=0)
    enough_counts = df['count']>2000

    df_results = pd.DataFrame(columns=['lang','feature','accuracy'])
    ii = 0

    for permute in tqdm(range(num_permute),file=tqdmout):

        for feature in features:

            has_feature = np.logical_and(~df[feature].isna(),enough_counts)

            # == Feature decodability within languages
            for lang_use in langs_use:
                indices = np.logical_and(has_feature,df['lang']==lang_use).values
                label_indices = np.where(indices)[0]
                if permute>0:
                    label_indices = np.random.permutation(label_indices)
                model = LogisticRegression()
                scores = cross_val_score(
                    model, h_phone[indices], 
                    df[feature].iloc[label_indices],
                    cv=LeaveOneOut()
                )
                assert(~np.any(np.isnan(scores)))
                df_results.loc[ii,'lang'] = lang_use
                df_results.loc[ii,'feature'] = feature
                df_results.loc[ii,'accuracy'] = np.mean(scores)
                df_results.loc[ii,'permute'] = permute
                ii += 1

            if len(langs_use)>1: # Only if it's a bilingual model

                # == Transfer between languages
                indices, label_indices = {}, {}
                # get indices for both languages
                for lang in langs_use:
                    indices[lang] = np.logical_and(has_feature,df['lang']==lang).values
                    label_indices[lang] = np.where(indices[lang])[0]
                    if permute>0:
                        label_indices[lang] = np.random.permutation(label_indices[lang])
                # alternate both languages for training and testing
                scores = []
                for lang_source, lang_target in zip(langs_use,np.flipud(langs_use)):
                    model = LogisticRegression()
                    model.fit(
                        h_phone[indices[lang_source]], 
                        df[feature].iloc[label_indices[lang_source]],
                    )
                    score = model.score(
                        h_phone[indices[lang_target]], 
                        df[feature].iloc[label_indices[lang_target]],
                    )
                    assert(~np.isnan(score))
                    scores.append(score)

                df_results.loc[ii,'lang'] = ''.join(langs_use)
                df_results.loc[ii,'feature'] = feature
                df_results.loc[ii,'accuracy'] = np.mean(scores)
                df_results.loc[ii,'permute'] = permute
                ii += 1

        if len(langs_use)>1: # Only if it's a bilingual model
            # Language decodability
            indices = np.logical_and(np.isin(df['lang'],langs_use),enough_counts)
            label_indices = np.where(indices)[0]
            if permute>0:
                label_indices = np.random.permutation(label_indices)
            model = LogisticRegression()
            scores = cross_val_score(
                model, h_phone[indices], 
                df['lang'].iloc[label_indices],
                cv=LeaveOneOut()
            )
            assert(~np.any(np.isnan(scores)))
            df_results.loc[ii,'lang'] = ''.join(langs_use)
            df_results.loc[ii,'feature'] = 'lang'
            df_results.loc[ii,'accuracy'] = np.mean(scores)
            df_results.loc[ii,'permute'] = permute
            ii += 1

    # == Compute z-scores
    if num_permute>2:
        for lang in df_results['lang'].unique():
            df_lang = df_results.query(f'lang=="{lang}"')
            for feature in df_lang['feature'].unique():
                df_feat = df_lang.query(f'feature=="{feature}"')
                vals_permute = df_feat.query('permute>0')['accuracy'].values
                z = (df_feat.query('permute==0')['accuracy'].values[0]-vals_permute.mean())/vals_permute.std()
                df_results.loc[df_feat.query('permute==0').index,'z'] = z

    return df_results

if __name__=='__main__':

    # Monolingual English model
    config_path = 'experiments/commonvoice_en_phonemes_words.cfg'
    state_dict_path = os.path.join(config_path.split('.')[0],'pretraining','model_state.pth')
    df_results = feature_decodability(state_dict_path, ['en'], features)

    df_results.to_csv(
        os.path.join(config_path.split('.')[0],'pretraining','decodability.csv')
    )

    # Bilingual French-English model
    config_path = 'experiments/commonvoice_fren_phonemes_words.cfg'
    state_dict_path = os.path.join(config_path.split('.')[0],'pretraining','model_state.pth')
    df_results = feature_decodability(state_dict_path, ['fr','en'], features)

    df_results.to_csv(
        os.path.join(config_path.split('.')[0],'pretraining','decodability.csv')
    )

    # Alternative models
    config_paths = [
        'experiments/commonvoice_fren_homologes_words.cfg',
        'experiments/commonvoice_fren_words.cfg'
    ]
    for config_path in config_paths:
        state_dict_path = os.path.join(config_path.split('.')[0],'training','model_state.pth')
        df_results = feature_decodability(state_dict_path, ['fr','en'], features, layer='rnn_phone_linear', num_permute=50)
        df_results.to_csv(
            os.path.join(config_path.split('.')[0],'training','decodability.csv')
        )