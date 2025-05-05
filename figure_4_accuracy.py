# %%

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import seaborn.objects as so
from seaborn import axes_style
import os
from params import palette
from figure_functions import adjust_axis

plt.style.use('../training/mpl_stylesheet')

# %%

inits = {
    'l1_english': ['none','en19','en49'],
    'l1_french': ['none','fr19','fr49']
}
langs_train = {
    'l1_english': ['fren','fr'],
    'l1_french': ['fren','en']
}
languages = {
    'en': 'English',
    'fr': 'French',
    'enfr': 'Transfer'
}
sequences = [
    'Monolingual', 'Simultaneous', 'Early sequential', 'Late sequential'
]

dfs = []
for run in [1,2,3,4,5,6]:
    csvpath = f'sequential/accuracies_run_{run}.csv'
    # figure rows
    for simulation in inits:
        df = pd.read_csv(csvpath, index_col=0)
        df = df.query('level=="phonemes"')
        df = df.query(f'init in {inits[simulation]}')
        df = df.query(f'lang_train in {langs_train[simulation]}')
        df['simulation'] = simulation
        # figure columns
        df['sequence'] = 'Monolingual'
        df.loc[df['lang_train']=='fren','sequence'] = 'Simultaneous'
        df.loc[df['init']==inits[simulation][1],'sequence'] = 'Early sequential'
        df.loc[df['init']==inits[simulation][2],'sequence'] = 'Late sequential'
        # color highlighting
        df['language_order'] = 'L2'
        df.loc[df['lang']==inits[simulation][2][:2],'language_order'] = 'L1'
        # show only the relevant monolingual model in each row
        df = df.query('~((sequence=="Monolingual")&(language_order=="L1"))')
        # x-axis: overall number of training samples
        df['num_samples_total'] = df['num_samples']
        df.loc[df['init']==inits[simulation][1],'num_samples_total'] += df.query('epoch==19')['num_samples'].values[0]
        df.loc[df['init']==inits[simulation][2],'num_samples_total'] += df.query('epoch==49')['num_samples'].values[0]
        # x-axis: training samples per language
        df['num_samples_lang'] = df['num_samples']
        df.loc[df['lang_train']=='fren','num_samples_lang'] /= 2
        df['num_samples_lang'].astype(int)
        # Collect runs
        df['run'] = run
        df['is_initials'] = False
        dfs.append(df)
df = pd.concat(dfs).reset_index(drop=True)

# Add initial epochs
for l1 in ['en','fr']:
    for epoch_init, sequence in zip([19,49],sequences[2:]):
        df_init = df.query(f'(init=="none")&(lang_train=="{l1}")&(epoch<={epoch_init})').copy()
        df_init['sequence'] = sequence
        df_init['simulation'] = 'l1_'+{'en':'english','fr':'french'}[l1]
        df_init['is_initials'] = True
        dfs.append(df_init)
df = pd.concat(dfs).reset_index(drop=True)

df['colors'] = df['simulation']+' - '+df['sequence']+' - '+df['lang']

# %%

palette = {
    'en-0': '#570305',
    'en-1': '#aa060a',
    'en-2': '#fc2a2a',
    'en-3': '#f89fa6',
    'fr-0': '#022c64',
    'fr-1': '#0359c8',
    'fr-2': '#52a0ff',
    'fr-3': '#aed2fe',
}

gray = '#a2a2a2'

figure_palette = {
    'l1_english - Monolingual - fr': palette['fr-3'],
    'l1_english - Simultaneous - en': gray,
    'l1_english - Simultaneous - fr': palette['fr-2'],
    'l1_english - Early sequential - en': gray,
    'l1_english - Early sequential - fr': palette['fr-1'],
    'l1_english - Late sequential - en': gray,
    'l1_english - Late sequential - fr': palette['fr-0'],
    'l1_french - Monolingual - en': palette['en-3'],
    'l1_french - Simultaneous - en': palette['en-2'],
    'l1_french - Simultaneous - fr': gray,
    'l1_french - Early sequential - en': palette['en-1'],
    'l1_french - Early sequential - fr': gray,
    'l1_french - Late sequential - en': palette['en-0'],
    'l1_french - Late sequential - fr': gray
}

# %%

plt.close(1)
fig = plt.figure(1)
axs = fig.subplots(2,5)
for ax_row, simulation in zip(axs, ['l1_french','l1_english']):
    for ax, sequence in zip(ax_row, sequences):
        df_plot = df.query(f'simulation=="{simulation}"')
        df_plot = df_plot.query(f'sequence=="{sequence}"')
        order = df_plot['colors'].unique()
        if figure_palette[order[0]]!=gray:
            order = np.flipud(order)
        sns.lineplot(
            x = 'num_samples_total', y='accuracy', hue='colors',
            data = df_plot, n_boot=100, palette=figure_palette, #estimator=None, units='run',
            ax=ax, legend=False, alpha=.5, hue_order=order, errorbar='ci'
        )
    df_plot = df.query(f'simulation=="{simulation}"')
    df_plot = df_plot.query('language_order=="L2"').query('~is_initials')
    sns.lineplot(
        x = 'num_samples_lang', y='accuracy', hue='colors',
        data = df_plot, n_boot=100, palette=figure_palette, #estimator=None, units='run',
        ax=ax_row[-1], legend=False, alpha=.5, errorbar='ci'
    )
for ax in axs.flatten():
    ax.set_xscale('log')
    ax.set_xlim([500,3000000])
    ax.set_ylim([0,.8])
    ax.set_yticks([0,.2,.4,.6,.8])
    ax.set_yticklabels([])
    ax.set_xticks(10**np.arange(3,7))
    ax.set_xlabel('')
    ax.set_ylabel('')
for ax, sequence in zip(axs[0][:-1],sequences):
    ax.set_title(sequence)
    ax.set_xticklabels([])
for ax in axs[1][:-1]:
    ax.set_xlabel('Number of clips\n(total)')
axs[0][-1].set_title('Comparison')
axs[1][-1].set_xlabel('Number of clips\n(per language)')
for ax_col in [axs[:,0],axs[:,-1]]:
    for ax in ax_col:
        ax.set_yticklabels([0,.2,.4,.6,.8])
        ax.set_ylabel('Phoneme accuracy')
for x,ax_row in enumerate(axs):
    for y,ax in enumerate(ax_row):
        adjust_axis(ax,[-0.06, -0.045, -.03, -.015, 0.05][y],.05,1.15,.95)
sns.despine(fig=fig)
fig.set_size_inches([7,3.5])
fig.savefig(f'imgs/figure_4_accuracy.png',dpi=300)
fig.savefig(f'imgs/figure_4_accuracy.pdf')
