"""
QSM Reconstruction Challenge 2019

Create pairplot with hue = algorithm_type

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

    df1 = pd.read_pickle(datapath / 'stage2_snr1.pkl')
    df2 = pd.read_pickle(datapath / 'stage2_snr2.pkl')

    df_stage1 = pd.read_pickle(datapath / 'stage1.pkl')

    rmse_cutoff = 100
    df1 = df1[df1.rmse < rmse_cutoff]
    df2 = df2[df2.rmse < rmse_cutoff]

    metrics = ['rmse', 'CalcStreak', 'DeviationFromCalcMoment']
    metrics_renamed = ['NRMSE', 'Calcif. Streaking', 'Calcif. Moment Error']

    avgtemplate = 'avg {} SNR1 + SNR2'
    difftemplate = 'diff {} |SNR1 - SNR2|'

    df = pd.DataFrame([])
    intersection_submissions = set(df1['Submission Identifier of the corresponding Stage 1 submission']).intersection(set(df2['Submission Identifier of the corresponding Stage 1 submission']))
    df_stage1 = df_stage1.set_index('Submission Identifier')
    df_stage1 = df_stage1.loc[df_stage1.index.intersection(intersection_submissions)]
    df1 = df1.set_index('Submission Identifier of the corresponding Stage 1 submission')
    df2 = df2.set_index('Submission Identifier of the corresponding Stage 1 submission')
    df1 = df1.loc[df1.index.intersection(intersection_submissions)]
    df2 = df2.loc[df2.index.intersection(intersection_submissions)]
    df['algorithm type'] = df_stage1['algorithm type']
    for met, newname in zip(metrics, metrics_renamed):
        df[avgtemplate.format(newname)] = 0.5 * (df1[met] + df2[met])
        df[difftemplate.format(newname)] = df1[met] - df2[met]

        sns.scatterplot(x=avgtemplate.format(newname), y=difftemplate.format(newname), hue='algorithm type', data=df)
        xs = df[avgtemplate.format(newname)].min(), df[avgtemplate.format(newname)].max()
        ymean, ystd = df[difftemplate.format(newname)].mean(), df[difftemplate.format(newname)].std()
        plt.hlines(ymean - ystd, xs[0], xs[1], 'k', '--')
        plt.hlines(ymean + ystd, xs[0], xs[1], 'k', '--')
        #plt.legend(loc=(1.1, 0.5))

        plt.savefig(imagepath / 'stage2_snr1vssnr2_blandaltman_{}.png'.format(met), bbox_inches='tight', dpi=300)
        #plt.savefig(imagepath / 'stage2_snr1vssnr2_blandaltman_{}.svg'.format(met))
        plt.close()


    df['NRMSE_Stage1'] = df_stage1['rmse']
    df['NRMSE_Stage2'] = df[avgtemplate.format('NRMSE')]
    sns.scatterplot(x='NRMSE_Stage1', y='NRMSE_Stage2', hue='algorithm type', data=df)
    plt.plot(np.sort(df['NRMSE_Stage1']), np.sort(df['NRMSE_Stage1']), 'lightgray')
    plt.savefig(imagepath / 'NRMSE_stage1_vs_stage2.png', bbox_inches='tight', dpi=300)
    print()