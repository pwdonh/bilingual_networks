import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn.objects as so
from seaborn import axes_style
import os

from params import palette
from figure_functions import adjust_axis
palette['fren'] = palette['transfer']
palette['langdecode'] = palette['transfer']

plt.style.use('../training/mpl_stylesheet')

df = pd.read_csv(
    os.path.join(f'experiments/commonvoice_fren_phonemes_words','pretraining','decodability.csv'),
    index_col=0
)

plt.close(1)
fig = plt.figure(1)
axs = fig.subplots(1,3)
for ii, (ax, lang) in enumerate(zip(axs, ['fr','en','fren'])):
    if lang=='langdecode':
        features = ['lang']
    else:
        features = ['sonority','manner','voicing','place']
    query_strings = ['permute==0','permute>0']
    for query_string, dotsize in zip(query_strings,[5,1]):
        p = (
            so.Plot(
                df.query(query_string).query(f'lang=="{lang}"'), y='accuracy', 
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
        ax.set_ylabel('Feature decodability')
    else:
        ax.set_yticklabels([])
    ax.set_xticklabels(['Sonority','Manner','Voicing','Place'])
    adjust_axis(ax,ii*.025,0,1.1,1)
sns.despine(fig=fig)
fig.set_size_inches([4.5,2])
fig.savefig('imgs/figure_s2_decodability.png',dpi=300)
fig.savefig('imgs/figure_s2_decodability.pdf')