import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn.objects as so
from seaborn import axes_style
import os

from params import palette
palette['fren'] = palette['transfer']
palette['langdecode'] = palette['transfer']

plt.style.use('../training/mpl_stylesheet')

mtypes = [
    'homologes_words', 'words'
]
labels = {
    'homologes_words': 'Homologes & Words',
    'words': 'Words only',
}
dfs = []
for mtype in mtypes:
    df = pd.read_csv(
        os.path.join(f'experiments/commonvoice_fren_{mtype}','pretraining','decodability.csv'),
        index_col=0
    )
    df['type'] = mtype
    dfs.append(df)
df = pd.concat(dfs)

df.loc[df['feature']=="lang",'lang'] = 'langdecode'

plt.close(1)
fig = plt.figure(1)
axs = fig.subplots(3,len(mtypes))
for ax_col, mtype in zip(axs.T, mtypes):
    for ax, lang in zip(ax_col, ['fr','en','fren']):
        if lang=='langdecode':
            features = ['lang']
        else:
            features = ['sonority','manner','voicing','placerough']
        if mtype=='mel':
            query_strings = ['permute==0']
        else:
            query_strings = ['permute==0','permute>0']
        for query_string, dotsize in zip(query_strings,[5,1]):
            p = (
                so.Plot(
                    df.query(query_string).query(f'type=="{mtype}"').query(f'lang=="{lang}"'), y='accuracy', 
                    x='feature', color='lang'
                )
                .add(so.Dots(pointsize=dotsize),so.Jitter())
                .scale(
                    x=so.Nominal(order=features),
                    color=so.Nominal(palette)
                )
                .theme(axes_style("ticks"))
                .on(ax)
            )
            p.plot()
        ax.set_ylim([-.05,1.05])
        ax.set_xlabel('')
        ax.set_ylabel('')
        if lang=='fr':
            ax.set_title(labels[mtype])
            ax.set_xticklabels([])
        elif lang=='fren':
            ax.set_xticklabels(['Sonority','Manner','Voicing','Place','Language'],rotation=45,ha='right')
        else:
            ax.set_xticklabels([])
            ax.set_ylabel('Feature decodability')
        if mtype==mtypes[-1]:
            ax.set_yticklabels([])
            ax.set_ylabel('')
sns.despine(fig=fig)
fig.set_size_inches([2.6,3])
fig.savefig('imgs/figure_s4_alternative_models.png',dpi=300)
fig.savefig('imgs/figure_s4_alternative_models.pdf')