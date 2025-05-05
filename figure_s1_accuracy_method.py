# %%

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

plt.style.use('../training/mpl_stylesheet')

# %%

config_path = 'experiments/commonvoice_en_phonemes_words.cfg'
basepath = f'{config_path.replace(".cfg","")}/pretraining'

level = 'word'

df = pd.read_csv(
    os.path.join(basepath, f'example_traces_sentence_en.csv'), index_col=0
)
f = joblib.load(os.path.join(basepath, f'example_probabilities_sentence_en.pickle'))
p_word = f['p_word']
p_phoneme = f['p_phoneme']
spec = f['spec']
df_w = pd.read_csv('data/words.csv',index_col=0)
df_p = pd.read_csv('data/phonemes.csv',index_col=0)
df_p.loc[df_p['ipa'].isna(),'ipa'] = df_p.loc[df_p['ipa'].isna(),'cmu']

word_indices = torch.unique_consecutive(torch.LongTensor(df['word'].values))
word_indices = word_indices[word_indices!=-1].numpy()
words = df_w.loc[word_indices,'word'].values.tolist()

predicted = p_word.argmax(1)
df['predicted'] = predicted.numpy()
df['correct'] = (df['predicted']==df['word']).astype(int)
df.loc[df['word']==-1,'correct'] = 2
accuracy = df.loc[df['word']!=-1,'correct'].mean()

# %%

cols = sns.color_palette()[:len(words)]
cols_rest = list(sns.color_palette('pastel'))
cols_rest = cols_rest+cols_rest+cols_rest+cols_rest

frames = np.arange(0,spec.shape[0]-25)

uniques, inverse = torch.unique_consecutive(predicted,return_inverse=True)
inverse = inverse[frames]
uniques = uniques[:max(inverse[frames])+2]
uniq, inv = torch.unique(uniques,return_inverse=True)
words_palette = words+np.setdiff1d(df_w.loc[uniq.numpy(),'word'].values,words).tolist()
palette = {word: col for word, col in zip(words_palette,cols+cols_rest)}
words_pred = df_w.loc[uniques.numpy(),'word'].values

uniques_c, inverse_c = torch.unique_consecutive(torch.Tensor(df['correct']),return_inverse=True)

# %%

t = df.loc[frames,'time']
f = np.arange(spec.shape[1])

plt.close(1)
fig = plt.figure(1)
ax_spec, ax_pred, ax_annot, ax_correct = fig.subplots(4,1,sharex=True,height_ratios=[5,1,1,1])
ax = ax_spec.twinx()
ax_spec.pcolormesh(t,f,spec[frames].T,cmap='rocket', rasterized=True)
for i_word, word in enumerate(words_pred):
    w_index = df_w.query(f'word=="{word}"').query('lang=="en"').index.item()
    ax.plot(t, p_word[frames,w_index],'w',linewidth=3)
    color = palette[word]
    ax.plot(t, p_word[frames,w_index],linewidth=2,color=color)
    frames_pred = np.where(inverse==i_word)[0]
    frame = np.round(np.mean(frames_pred)).astype(int)
    rect = plt.Rectangle(
        [t[frames_pred[0]],-.35],width=len(frames_pred)*.01,
        height=.7,facecolor=color,edgecolor='k'
        )
    ax_pred.add_patch(rect)
    ax_pred.text(t[frame],[.7,-.7][i_word%2],word,color=color,ha='center',va='center')
ax.set_ylabel('Word probability')
ax.yaxis.tick_left()
ax.yaxis.set_label_position("left")
ax_spec.xaxis.tick_top()
ax_spec.set_yticks([])
ax_pred.set_ylim([-.8,.8])
ax_pred.axis(False)
for i_word, word_index in enumerate(word_indices):
    word = df_w.loc[word_index,'word']
    frames_annot = df.query(f'word=={word_index}').index.values
    frame = np.round(np.mean(frames_annot)).astype(int)
    color = palette[word]
    print(t[frames_annot[0]])
    rect = plt.Rectangle(
        [t[frames_annot[0]],-.35],width=len(frames_annot)*.01,
        height=.7,facecolor=color,edgecolor='k'
    )
    ax_annot.add_patch(rect)
    ax_annot.text(t[frame],.7,word,color=color,ha='center',va='center')
ax_annot.set_ylim([-.8,.8])
ax_annot.axis(False)
frames_cs = []
for i_correct, correct in enumerate(uniques_c):
    frames_c = np.where(inverse_c==i_correct)[0]
    frames_cs.append(int(np.round(frames_c.mean())))
    color = ['w','k','w'][int(correct)]
    hatch = ["","",r"///////"][int(correct)]
    rect = plt.Rectangle(
        [t[frames_c[0]],-.35],width=len(frames_c)*.01,
        height=.7,facecolor=color,edgecolor='k', hatch=hatch
    )
    ax_correct.add_patch(rect)
ax_correct.text(t[frames_cs[1]],.7,'incorrect',ha='center',va='center')
ax_correct.text(t[frames_cs[4]],.7,'correct',ha='center',va='center')
ax_correct.text(t[frames[-1]],.7,'not a word',ha='center',va='center')
ax_correct.text(t[frames[-1]],-.8,'Accuracy: {:0.2f}'.format(accuracy),ha='right',va='top')
ax_correct.set_ylim([-.8,.8])
ax_correct.axis(False)
ax_spec.set_xticks(np.arange(0,2,.2))
ax.set_ylim([0,1])
ax.set_xlim([t[frames[0]],t[frames[-1]]])
fig.set_size_inches([6.2,3.2])
fig.savefig('imgs/accuracy_method_words.png',dpi=300)
fig.savefig('imgs/accuracy_method_words.pdf')

# %%

frames = np.arange(0,spec.shape[0]-25)
# frames = np.arange(50)

phoneme_indices, annot_inverse = torch.unique_consecutive(torch.LongTensor(df['phoneme'].values),return_inverse=True)
phonemes = df_p.loc[phoneme_indices[phoneme_indices!=-1].numpy(),'cmu'].values.tolist()

cols = list(sns.color_palette())
cols = (cols+cols)[:len(np.unique(phonemes))]
cols_rest = list(sns.color_palette('pastel'))
cols_rest = cols_rest+cols_rest+cols_rest+cols_rest
unique_phonemes = np.array(phonemes)[np.sort(np.unique(phonemes,return_index=True)[1])]

predicted = p_phoneme.argmax(1)
df['predicted'] = predicted.numpy()
df['correct'] = (df['predicted']==df['phoneme']).astype(int)
df.loc[df['phoneme']==-1,'correct'] = 2
accuracy = df.loc[df['phoneme']!=-1,'correct'].mean()

uniques, inverse = torch.unique_consecutive(predicted,return_inverse=True)
inverse = inverse[frames]
uniques = uniques[min(inverse):]
uniq, inv = torch.unique(uniques,return_inverse=True)
phonemes_palette = unique_phonemes.tolist()+np.setdiff1d(df_p.loc[uniq.numpy(),'cmu'].values,phonemes).tolist()
palette = {phoneme: col for phoneme, col in zip(phonemes_palette,cols+cols_rest)}
phonemes_pred = df_p.loc[uniques.numpy(),'cmu'].values

uniques_c, inverse_c = torch.unique_consecutive(torch.Tensor(df['correct']),return_inverse=True)


# %%

t = df.loc[frames,'time'].values

plt.close(1)
fig = plt.figure(1)
ax_spec, ax_pred, ax_annot, ax_correct = fig.subplots(4,1,sharex=True,height_ratios=[5,1,1,1])
ax = ax_spec.twinx()
ax_spec.pcolormesh(t,f,spec[frames].T,cmap='rocket', rasterized=True)

for i_ph, phoneme in enumerate(phonemes_pred):
    if phoneme in ['sil','sp','spn']:
        lang = 'de'
    else:
        lang = 'en'
    p_index = df_p.query(f'cmu=="{phoneme}"').query(f'lang=="{lang}"').index.item()
    ax.plot(t, p_phoneme[frames,p_index],'w',linewidth=3)
    color = palette[phoneme]
    ax.plot(t, p_phoneme[frames,p_index],linewidth=2,color=color)
    frames_pred = np.where(inverse==i_ph)[0]
    if len(frames_pred)>0:
        frame = np.round(np.mean(frames_pred)).astype(int)
        rect = plt.Rectangle(
            [t[frames_pred[0]],-.35],width=len(frames_pred)*.01,
            height=.7,facecolor=color,edgecolor='k'
            )
        ax_pred.add_patch(rect)
        ax_pred.text(t[frame],[.7,-.7][i_ph%2],phoneme,color=color,ha='center',va='center')
ax.set_ylabel('Phoneme probability')
ax.yaxis.tick_left()
ax.yaxis.set_label_position("left")
ax_spec.xaxis.tick_top()
ax_spec.set_yticks([])
ax_pred.set_ylim([-.8,.8])
ax_pred.axis(False)

for i_ph, phoneme_index in enumerate(phoneme_indices):
    if phoneme_index!=-1:
        phoneme = df_p.loc[phoneme_index.item(),'cmu']
        frames_annot = np.where(annot_inverse==i_ph)[0]
        frame = np.round(np.mean(frames_annot)).astype(int)
        color = palette[phoneme]
        rect = plt.Rectangle(
            [t[frames_annot[0]],-.35],width=len(frames_annot)*.01,
            height=.7,facecolor=color,edgecolor='k'
        )
        ax_annot.add_patch(rect)
        ax_annot.text(t[frame],.7,phoneme,color=color,ha='center',va='center')
ax_annot.set_ylim([-.8,.8])
ax_annot.axis(False)

frames_cs = []
for i_correct, correct in enumerate(uniques_c):
    frames_c = np.where(inverse_c==i_correct)[0]
    frames_cs.append(int(np.round(frames_c.mean())))
    color = ['w','k','w'][int(correct)]
    hatch = ["","",r"///////"][int(correct)]
    rect = plt.Rectangle(
        [t[frames_c[0]],-.35],width=len(frames_c)*.01,
        height=.7,facecolor=color,edgecolor='k', hatch=hatch
    )
    ax_correct.add_patch(rect)
ax_correct.text(t[frames_cs[13]],.7,'incorrect',ha='center',va='center')
ax_correct.text(t[frames_cs[10]],.7,'correct',ha='center',va='center')
ax_correct.text(t[frames[-1]],.7,'not a word',ha='center',va='center')
ax_correct.text(t[frames[-1]],-.8,'Accuracy: {:0.2f}'.format(accuracy),ha='right',va='top')
ax_correct.set_ylim([-.8,.8])
ax_correct.axis(False)

ax_pred.set_ylabel('Predicted')
ax_spec.set_xlim([t[frames[0]],t[frames[-1]]])
# ax_spec.set_xlim([.9,t[frames[-1]]])

fig.set_size_inches([6.2,3.2])
fig.savefig('imgs/accuracy_method_phonemes.png',dpi=300)
fig.savefig('imgs/accuracy_method_phonemes.pdf')

# %%

specs = [
    joblib.load(os.path.join(basepath, f'example_probabilities_sentence_en.csv'))['spec'],
    joblib.load(os.path.join(basepath.replace('_en_','_fren_'), f'example_probabilities_sentence_fr.csv'))['spec']
]

plt.close(1)
fig = plt.figure(1)
axs = fig.subplots(1,2)
for ax, spec in zip(axs,specs):
    t = np.arange(spec.shape[0])/160
    f = np.arange(128)
    ax.pcolormesh(t,f,spec.T,cmap='gray',rasterized=True)
    ax.set_xlim([0,1.1])
    ax.set_aspect(.005)
fig.savefig('imgs/example_sentence_spectrograms.pdf',dpi=300)