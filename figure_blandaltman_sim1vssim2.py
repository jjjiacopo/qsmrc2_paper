"""
QSM Reconstruction Challenge 2019

Bland-Altman plots comparing Sim1 and Sim2

author: jakob.meineke@philips.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from translation import translation

if __name__=="__main__":

    datapath = Path('data')
    imagepath = Path('images')
    if not imagepath.exists():
        imagepath.mkdir()

    df = pd.read_csv(datapath / 'stage1_sim1sim2_rmse.csv', encoding='latin-1')

    df = df[df.rmse1 < 100]

    metrics_1 = ['rmse1']
    metrics_2 = ['rmse2']
    avgmet = ['average NRMSE: Sim1 + Sim2']
    diffmet = ['NRMSE difference: Sim1 - Sim2']

    df['algorithm type'] = np.array([translation['algorithm type'][val] for val in df['algo_type']])
    for m1, m2, am, dm in zip(metrics_1, metrics_2, avgmet, diffmet):
        df[am] = 0.5 * (df[m1] + df[m2])
        df[dm] = df[m1] - df[m2]

    for i in range(len(avgmet)):
        ymean, ystd = df[diffmet[i]].mean(), df[diffmet[i]].std()
        sns.scatterplot(x=avgmet[i], y=diffmet[i], hue='algorithm type', style='algorithm type', palette='bright',
                        data=df)
        plt.legend(loc=(1.1, 0.5))
        plt.hlines((ymean-ystd, ymean+ystd), df[avgmet[i]].min(), df[avgmet[i]].max(), 'lightgray', '--', zorder=-1)
        plt.hlines(ymean, df[avgmet[i]].min(), df[avgmet[i]].max(), 'lightgray', '-', zorder=-1)
        plt.savefig(imagepath / 'blandaltman_{}.png'.format('rmse'), bbox_inches='tight', dpi=300)
        plt.savefig(imagepath / 'blandaltman_{}.svg'.format('rmse'))
        plt.close()

    df['NRMSE Sim1'] = df['rmse1']
    df['NRMSE Sim2'] = df['rmse2']

    plt.figure()
    plt.plot(np.sort(df['NRMSE Sim1'].values), np.sort(df['NRMSE Sim1'].values), 'lightgray', zorder=-1)
    sns.scatterplot(x='NRMSE Sim1', y='NRMSE Sim2', hue='algorithm type', style='algorithm type', palette='bright',
                    data=df)
    plt.savefig(imagepath / 'nrmse_Sim1_vs_Sim2.png', bbox_inches='tight', dpi=300)
    plt.savefig(imagepath / 'nrmse_Sim1_vs_Sim2.svg')
    plt.close()
