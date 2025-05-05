import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
import seaborn as sns

from params import palette
from figure_2_representation import (
    normalize_columns, select_axis_sign,
    figure_adjustments, figure_adjustments_rep,
    figure_adjustments_trace
)

colors = {
    'en': 'Reds_r',
    'fr': 'Blues_r'
}

for config_label, langs_use in zip(['en_phonemes_words','fren_phonemes_words'],[['en'],['en','fr']]):

    config_path = f'experiments/commonvoice_{config_label}.cfg'
    cols = ['pca_0','pca_1','pca_2']

    df_w = pd.read_csv('words.csv',index_col=0)
    df_p = pd.read_csv('phonemes.csv',index_col=0)

    df = pd.read_csv(f'{config_path[:-4]}/pretraining/phonemes.csv',index_col=0)
    dfs = []
    for lang_use in langs_use:
        df_trace = pd.read_csv(f'{config_path[:-4]}/pretraining/example_traces_sentence_{lang_use}.csv',index_col=0)
        df_trace['lang_use'] = lang_use
        if lang_use=='fr':
            df_trace.loc[df_trace['phoneme']==126,'word'] = df_w.query('word=="j"').index.item()
            df_trace.loc[df_trace['phoneme']==107,'word'] = df_w.query('word=="ai"').index.item()
        dfs.append(df_trace)
    df_trace = pd.concat(dfs).reset_index(drop=True)
    lang_code = config_label.split('_')[0].strip('01234')

    df = df.query('select').copy()
    df, df_trace = normalize_columns(df, cols, df_trace)
    df, signs = select_axis_sign(df, 'sonority', lang_code, cols)
    df_trace[cols] *= signs
    df_trace[cols] *= .9
    # df['gray'] = nanfiller

    fig = px.scatter_3d(
        df, x=cols[0], y=cols[1], z=cols[2],
        color = 'lang', text='ipa',
        color_discrete_map=palette
    )
    figure_adjustments_rep(fig)
    figure_adjustments(fig,[-1.1,1.1])
    img_path = config_path.replace('experiments/',f'imgs/representation_lang_').replace('.cfg','.png')
    pio.write_image(fig, img_path, format='png', scale=3)

    traces = []
    for lang_use in langs_use:

        start = df_trace.query(f'lang_use=="{lang_use}"').index[0]#.query(f'phoneme=="{phons[0]}"').index[0]
        end = df_trace.query(f'lang_use=="{lang_use}"').query('phoneme!=-1').index[-1]#.query(f'phoneme=="{phons[1]}"').index[-1]
        fig = px.scatter_3d(
            df_trace.loc[start:end], 
            x=cols[0], y=cols[1], z=cols[2],
            color = 'time'
        )
        figure_adjustments_trace(fig)
        fig.data[0]['line'] = dict(
            width=12, color=df_trace.loc[:end,'time'],
            colorscale=colors[lang_use],
            cmin=df_trace.loc[start,'time'],
            cmax=df_trace.loc[end,'time']
        )
        traces.append(fig.data[0])
        
    fig = go.Figure(data=traces)
    figure_adjustments(fig,[-1.1,1.1])
    fig.layout['scene']['yaxis']['range'] = [-1.55,1.4]
    fig.layout['scene']['zaxis']['range'] = [-1.4,1.4]
    img_path = config_path.replace('experiments/',f'imgs/representation_trace_').replace('.cfg','.png')
    pio.write_image(fig, img_path, format='png', scale=3)

    import torch
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import json

    words, word_indices = torch.unique_consecutive(torch.Tensor(
        df_trace['word'].iloc[start:end].astype(int).values),return_inverse=True
    )
    if words[0]==-1:
        words = words[1:]
        word_indices -= 1
    cols = sns.color_palette(colors[lang_use],len(word_indices))
    cols_words = [mpl.colors.to_hex(cols[int(np.where(word_indices==word)[0].mean())]) for word in range(len(words))]
    x_words = [np.where(word_indices==word)[0][0] for word in range(len(words))]
    phones, phone_indices = torch.unique_consecutive(torch.Tensor(
        df_trace['phoneme'].iloc[start:end].astype(int).values),return_inverse=True
    )
    if phones[0]==-1:
        phones = phones[1:]
        phone_indices -= 1
    cols = sns.color_palette(colors[lang_use],len(phone_indices))
    cols_phones = [mpl.colors.to_hex(cols[int(np.where(phone_indices==phone)[0].mean())]) for phone in range(len(phones))]
    x_phones = [np.where(phone_indices==phone)[0][0] for phone in range(len(phones))]

    cmus = df_p.loc[phones.numpy(),'cmu'].values
    palette = {p+'_'+lang_use:c for p,c in zip(np.flipud(cmus),np.flipud(cols_phones))}
    json.dump(palette,open(f'imgs/example_{lang_use}.json','w'))

    plt.close(1)
    fig = plt.figure(1)
    axs = fig.subplots(1,2)
    for ax, xs, cols, items, df_i in zip(axs, [x_words,x_phones], [cols_words,cols_phones], [words,phones], [df_w['word'],df_p['ipa']]):
        for x, col, word in zip(xs,cols,items):
            ax.text(x/(end-start)*10,.5,df_i.loc[word.item()],color=col,weight='bold')
        ax.set_xlim([-.2,10.2])
        ax.set_ylim([-.2,1.2])
        ax.set_aspect(1)
        ax.axis(False)
    fig.set_size_inches([9,5])
    fig.savefig('imgs/example_words_and_phones.pdf')

    pal_en = json.load(open(f'imgs/example_en.json'))
    if 'fr' in langs_use:
        pal_fr = json.load(open(f'imgs/example_fr.json'))
        palette = {**pal_en, **pal_fr}
    else:
        palette = {**pal_en}
    cols = ['pca_0','pca_1','pca_2']

    df['cmu_lang'] = df['cmu']+'_'+df['lang']
    fig_rep = px.scatter_3d(
        df, x=cols[0], y=cols[1], z=cols[2],
        color = 'cmu_lang', text='ipa',
    )
    figure_adjustments_rep(fig_rep)
    figure_adjustments(fig_rep,[-1.1,1.1])
    for data in fig_rep.data:
        if data['legendgroup'] in palette:
            data['textfont']['color'] = palette[data['legendgroup']]
        else:
            data['textfont']['color'] = '#777777'
    img_path = config_path.replace('experiments/',f'imgs/representation_lang_').replace('.cfg','.png')
    pio.write_image(fig_rep, img_path, format='png', scale=3)