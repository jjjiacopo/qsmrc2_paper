"""
QSM Reconstruction Challenge 2019

Create pairplot with hue = input

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

    inputfile = 'stage1'
    df = pd.read_pickle(datapath / (inputfile+'.pkl'))

    hue = "input"
    magmets = ['rmse', 'Visual', #'rmse_detrend_Blood',
               #'rmse_detrend_DGM',
               'CalcStreak',
               'DeviationFromCalcMoment']
    rename = ['NRMSE', 'Visual', #'NRMSEd Blood',
              #'NRMSEd DeepGM',
              'Calcif. Streaking',
              'Calcif. Moment Error']

    for met, remet in zip(magmets, rename):
        df[remet] = df[met]

    dfmm = pd.concat(
        [df[df[hue] == 'frequency'].sort_values('rmse')[:20], df[df[hue] == "multi-echo"].sort_values('rmse')[:20]])
    sns.pairplot(dfmm[dfmm["magnitude info"] == True][rename + [hue]], hue=hue, palette='bright')
    plt.savefig(imagepath /'pairplot_hueinput_usemag_{}.png'.format(inputfile), bbox_inches='tight', dpi=300)
    sns.pairplot(dfmm[dfmm["magnitude info"] == False][rename + [hue]], hue=hue, palette='bright')
    plt.savefig(imagepath /'pairplot_hueinput_nomag_{}.png'.format(inputfile), bbox_inches='tight', dpi=300)
    sns.pairplot(dfmm[rename + [hue]], hue=hue, palette='bright')
    plt.savefig(imagepath /'pairplot_hueinput_{}.png'.format(inputfile), bbox_inches='tight', dpi=300)
    plt.savefig(imagepath /'pairplot_hueinput_{}.svg'.format(inputfile))

    huemag = "magnitude info"
    sns.pairplot(dfmm[rename + [huemag]], hue=huemag, palette='bright')
    plt.savefig(imagepath /'pairplot_huemag.png'.format(inputfile), bbox_inches='tight', dpi=300)

    print()