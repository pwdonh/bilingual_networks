import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import to_hex

from params import palette, nanfiller

def figure_adjustments_rep(fig):
    for data in fig.data:
        data['mode'] = 'text'
        data['text'] = ['<b>'+text+'</b>' for text in data['text']]
        data['textposition'] = 'middle center'
        data['textfont']['color'] = data['marker']['color']
        data['textfont']['family'] = 'Arial, sans-serif'
        data['textfont']['size'] = 30
        data.showlegend = False
    return fig

def figure_adjustments_trace(fig):
    for data in fig.data:
        data['mode'] = 'lines'
        data.showlegend = False
    return fig

def figure_adjustments(fig, range=[-.7,.7]):
    fig.layout['autosize'] = False
    fig.layout['hovermode'] = False
    fig.layout['scene']['camera'] = dict(
        up=dict(x=0,y=0,z=1),
        center=dict(x=0,y=0,z=0),
        eye=dict(x=-.8, y=-1.9, z=.555)
    )
    fig.layout['height'] = 1100
    fig.layout['width'] = 1100
    fig.layout['autosize'] = False
    for axis in ['xaxis','yaxis','zaxis']:
        fig.layout['scene'][axis]['range'] = range
        fig.layout['scene'][axis]['showbackground'] = False
        fig.layout['scene'][axis]['mirror'] = True
        fig.layout['scene'][axis]['showline'] = False
        fig.layout['scene'][axis]['tickvals'] = [-.5,0,.5]
        fig.layout['scene'][axis]['showticklabels'] = False
        fig.layout['scene'][axis]['title'] = ''
        fig.layout['scene'][axis]['gridwidth'] = 5
        fig.layout['scene'][axis]['zeroline'] = False
    fig.update_layout(template='plotly_white')
    fig.update_layout(scene_aspectmode='cube')
    return fig

def normalize_columns(df, cols, df2=None):
    vrange = (df[cols].max()-df[cols].min())
    df[cols] /= vrange
    df[cols] *= .55*2
    df[cols] -= df[cols].min()
    df[cols] -= .55
    if df2 is None:
        return df
    else:
        df2[cols] /= vrange
        df2[cols] *= .55*2
        df2[cols] -= df[cols].min()
        df2[cols] -= .55
        return df, df2

def get_voicing_arrows(df, cols):
    arrows = []
    voicingpairs = df.query(f'voicingpair!="{nanfiller}"')['voicingpair_lang']
    for voicingpair in voicingpairs.unique():
        df_tmp = df.query(f'voicingpair_lang=="{voicingpair}"')
        df_tmp = df_tmp.sort_values('voicing')
        coords = df_tmp[cols].values.T
        vec = coords[:,1]-coords[:,0]
        vec /= np.linalg.norm(vec)
        c0 = coords[:,0]+vec*.05
        c1 = coords[:,1]-vec*.125
        arrows += [go.Cone(
            x=[c1[0]],y=[c1[1]],z=[c1[2]],u=[vec[0]*.2], v=[vec[1]*.2], w=[vec[2]*.2],
            anchor='tail', colorscale=['rgb(0,0,0)','rgb(0,0,0)','rgb(0,0,0)'],
            showscale=False
        )]
        arrows += [go.Scatter3d(
            x=[c0[0],c1[0]], y=[c0[1],c1[1]], z=[c0[2],c1[2]], mode='lines',
            line=dict(width=5, color='rgb(0,0,0)'), showlegend=False
        )]
    return arrows

def select_axis_sign(df, feature, lang_code, cols):
    
    if lang_code=='en':
        if feature=='sonority':
            orders = [
                ['sonority','3: vowel','1: obstruents'],
                ['placerough','labial','coronal'],
                ['voicing','voiced','voiceless']
            ]
        else:
            orders = [
                ['placerough','labial','coronal'],
                ['voicing','voiced','voiceless'],
                ['voicing','voiced','voiceless']
            ]
    elif lang_code=='fren':
        if feature=='sonority':
            orders = [
                ['lang','fr','en'],
                ['sonority','1: obstruents','3: vowel'],
                ['voicing','voiced','voiceless']
            ]
        else:
            orders = [
                ['lang','fr','en'],
                ['voicing','voiced','voiceless'],
                ['voicing','voiced','voiceless']
            ]
    elif lang_code=='defren':
        if feature=='sonority':
            orders = [
                ['lang','fr','en'],
                ['lang','fr','de'],
                ['sonority','1: obstruents','3: vowel']
            ]
        else:
            orders = [
                ['lang','fr','en'],
                ['lang','fr','de'],
                ['voicing','voiced','voiceless']
            ]

    signs = np.ones(3)
    for axis, (column, order) in enumerate(zip(cols,orders)):
        df_agg = df.groupby(order[0]).agg({col:'mean' for col in cols})
        signs[axis] = np.sign(df_agg.loc[order[1],column]-df_agg.loc[order[2],column])
    df[cols] *= signs

    return df, signs

if __name__=='__main__':

    config_labels = [
        'en_phonemes_words','fren_phonemes_words','defren_phonemes_words',
        'fren_homologes_words', 'fren_words'
    ]
    for config_label in config_labels:

        config_path = f'experiments/commonvoice_{config_label}.cfg'
        lang_code = config_label.split('_')[0].strip('12345')
        df = pd.read_csv(f'{config_path[:-4]}/pretraining/phonemes.csv',index_col=0)
        df = df.query('select').copy()

        for feature in ['sonority','voicing']:

            if feature in ['sonority','gray']:
                cols = ['pca_0','pca_1','pca_2']
            else:
                cols = ['pca_o_0','pca_o_1','pca_o_2']

            df['sonority_lang'] = df['sonority'] + nanfiller + df['lang']
            df['voicing_lang'] = df['voicing'] + nanfiller + df['lang']
            df['voicingpair_lang'] = df['voicingpair'] + nanfiller + df['lang']
            df = normalize_columns(df, cols)
            df, _ = select_axis_sign(df, feature, lang_code, cols)

            if feature=='voicing':
                df_plot = df.query('obstruents')
            else:
                df_plot = df

            fig = px.scatter_3d(
                df_plot, x=cols[0], y=cols[1], z=cols[2],
                color = feature+'_lang', text='ipa',
                color_discrete_map=palette
            )
            figure_adjustments_rep(fig)
            figure_adjustments(fig)

            if feature=='voicing':
                fig.add_traces(get_voicing_arrows(df,cols))

            img_path = config_path.replace('experiments/',f'imgs/representation_{feature}_').replace('.cfg','.png')
            pio.write_image(fig, img_path, format='png', scale=3)

