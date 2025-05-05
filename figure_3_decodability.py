# %%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn.objects as so
from params import palette
from figure_functions import adjust_axis

plt.style.use('../training/mpl_stylesheet')

# %%

inits = {
    'l1_english': ['none','en19','en49'],
    'l1_french': ['none','fr19','fr49']
}
languages = {
    'en': 'English',
    'fr': 'French',
    'enfr': 'Transfer'
}
sequences = [
    'Simultaneous', 'Early sequential', 'Late sequential'
]

dfs = []
for run in [1,2,3,4,5,6]:
    csvpath = f'sequential/decodability_run_{run}.csv'
    for simulation in inits:
        df = pd.read_csv(csvpath, index_col=0)
        df = df.query('lang_train=="fren"')
        df = df.query('feature!="lang"')
        df = df.query(f'init in {inits[simulation]}')
        df = df.groupby(['lang','init','num_samples'],as_index=False).agg({
            'epoch': 'first', 'accuracy': 'mean'
        })
        df['simulation'] = simulation
        df['sequence'] = 'Simultaneous'
        df.loc[df['init']==inits[simulation][1],'sequence'] = 'Early sequential'
        df.loc[df['init']==inits[simulation][2],'sequence'] = 'Late sequential'
        df['language_order'] = 'L2'
        df.loc[df['lang']==inits[simulation][2][:2],'language_order'] = 'L1'
        df.loc[df['lang']=='enfr','language_order'] = 'Transfer'
        df['language'] = 'Transfer'
        df.loc[df['lang']=='en','language'] = 'English'
        df.loc[df['lang']=='fr','language'] = 'French'
        df['run'] = run
        dfs.append(df)
df = pd.concat(dfs).reset_index(drop=True)
df['sequence_language'] = df['sequence']+' - '+df['language']

# %%

figure_palette = {
    'Simultaneous - English': palette['en-3'],
    'Early sequential - English': palette['en-2'],
    'Late sequential - English': palette['en-1'],
    'Simultaneous - French': palette['fr-3'],
    'Early sequential - French': palette['fr-2'],
    'Late sequential - French': palette['fr-1'],
    'Simultaneous - Transfer': '#cccccc',
    'Early sequential - Transfer': '#888888',
    'Late sequential - Transfer': '#333333',
}

# %%

for simulation in inits.keys():
    plt.close(1)
    fig = plt.figure(1)
    axs = fig.subplots(1,3)
    for jj, (ax, language_order) in enumerate(zip(axs, ['L1','L2','Transfer'])):
        df_plot = df.query(f'simulation=="{simulation}"')
        df_plot = df_plot.query(f'language_order=="{language_order}"')
        language = languages[df_plot['lang'].unique().item()]
        order = [f'{sequence} - {language}' for sequence in sequences]
        sns.lineplot(
            x = 'num_samples', y='accuracy', hue='sequence_language',
            data = df_plot, n_boot=100, palette=figure_palette,
            ax=ax, hue_order=order, legend=False, alpha=.5
        )
        ax.set_ylim([0,1])
        ax.set_xscale('log')
        ax.set_xlim([500,3000000])
        ax.set_xticks([1000,10000,100000,1000000])
        ax.set_title(language_order)
        ax.set_xlabel('Number of L2 clips')
        if jj!=0:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Feature decodability')
    adjust_axis(axs[1],.015,0,1,1)
    adjust_axis(axs[2],.065,0,1,1)
    for ax in axs:
        adjust_axis(ax,0,.1,1.05,.9)
    sns.despine(fig=fig)
    fig.set_size_inches([3.42,1.6])
    # fig.tight_layout()
    fig.savefig(f'imgs/figure_3_representation_{simulation}.png',dpi=300)
    fig.savefig(f'imgs/figure_3_representation_{simulation}.pdf')

# %%

from proc_1_representation import phoneme_representations
from figure_2_representation import select_axis_sign

iters = [0,3,13,29,51]

for simulation in inits.keys():

    dfs = []
    for init in inits[simulation]:
        state_dict_path = f'sequential/run_1/state_dict-init_{init}-lang_fren-iter_59.pth'
        df_rep, pca, pca_o = phoneme_representations(
            state_dict_path,langs=['en','fr'],normalize=False
        )
        for iter in iters:
            state_dict_path = f'sequential/run_1/state_dict-init_{init}-lang_fren-iter_{iter}.pth'
            df_rep, pca, pca_o = phoneme_representations(
                state_dict_path,langs=['en','fr'],pca_o=pca_o,normalize=False
            )
            df_rep, _ = select_axis_sign(df_rep, 'voicing', 'fren', ['pca_o_0','pca_o_1','pca_o_2'])
            df_rep = df_rep.query('select').query('manner=="plosive"')
            df_rep['init'] = init
            df_rep['epoch'] = iter
            dfs.append(df_rep)
    df_rep = pd.concat(dfs).reset_index(drop=True)
    df_rep['lang_voicing'] = df_rep['lang']+'_'+df_rep['voicing']
    df_rep['pca_o_1'] *= -1

    figure_palette = {
        'fr_voiced': '#0359c8',
        'fr_voiceless': palette['fr-3'],
        'en_voiced': '#aa060a',
        'en_voiceless': palette['en-3']
    }

    plt.close(1)
    fig = plt.figure(1)
    axs = fig.subplots(3,5)
    for ax_row, init in zip(axs, inits[simulation]):
        for ax, iter in zip(ax_row, iters):
            p = (
                so.Plot(
                    df_rep.query(f'init=="{init}"').query(f'epoch=={iter}'), 
                    x='pca_o_2', y='pca_o_1', text="ipa", color='lang_voicing'
                )
                .add(so.Text(fontsize=9))
                .scale(color=figure_palette)
                .on(ax)
            )
            p.plot()
            ax.set_ylim([-1,1])
            ax.set_xlim([-1,1])
            ax.set_ylim([-3.,3.])
            ax.set_xlim([-2.5,2.5])
            ax.set_aspect(1)
            ax.axis(False)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.set_size_inches([5,3])
    fig.tight_layout()
    fig.savefig(f'imgs/figure_3_illustrations_{simulation}.png',dpi=300)
    fig.savefig(f'imgs/figure_3_illustrations_{simulation}.pdf')