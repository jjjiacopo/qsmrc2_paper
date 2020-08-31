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

if __name__=="__main__":

    datapath = Path('data')
    imagepath = Path('images')
    if not imagepath.exists():
        imagepath.mkdir()

    df = pd.read_csv(datapath / 'stage1_sim1sim2_rmse.csv', encoding='latin-1')

    df = df[df.rmse1 < 100]

    metrics_1 = ['rmse1']
    metrics_2 = ['rmse2']
    avgmet = ['avg NRMSE Sim1 + Sim2']
    diffmet = ['diff NRMSE |Sim1 - Sim2|']

    for m1, m2, am, dm in zip(metrics_1, metrics_2, avgmet, diffmet):
        df[am] = 0.5 * (df[m1] + df[m2])
        df[dm] = df[m1] - df[m2]

    for i in range(len(avgmet)):

        sns.scatterplot(x=avgmet[i], y=diffmet[i], hue='algo_type', data=df)
        plt.legend(loc=(1.1, 0.5))
        plt.savefig(imagepath / 'blandaltman_{}.png'.format('rmse'), bbox_inches='tight', dpi=300)
        plt.savefig(imagepath / 'blandaltman_{}.svg'.format('rmse'))