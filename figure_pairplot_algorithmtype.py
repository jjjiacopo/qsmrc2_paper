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

    inputfile = 'stage1'
    df = pd.read_pickle(datapath / (inputfile+'.pkl'))

    rmse_cutoff = 110
    df = df[df.rmse < rmse_cutoff]

    markers = ['o', 'X', 's', 'P']

    hue = "algorithm type"
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

    dfmm = pd.concat([df[df[hue]=='iterative'].sort_values('rmse')[:20],
                      df[df[hue]=="direct"].sort_values('rmse')[:20],
                      df[df[hue]=="deep learning"].sort_values('rmse')[:20],
                      #df[df[hue]=="hybrid"].sort_values('rmse')[:20]
                     ])
    sns.pairplot(dfmm[rename + [hue]], hue=hue, markers=markers[:dfmm[hue].unique().size], palette='bright')
    plt.savefig(imagepath / 'pairplot_algotype_best20_rmse_visual_rmseddgm_calc_{}.png'.format(inputfile), bbox_inches='tight', dpi=300)
    plt.savefig(imagepath / 'pairplot_algotype_best20_rmse_visual_rmseddgm_calc_{}.svg'.format(inputfile))

    # more pairplots for additional metrics
    magmets_alt = ['rmse', 'XSIM', 'HFEN', 'CC']
    rename_alt = ['NRMSE', 'XSIM', 'HFEN', 'Corr Coeff']

    for met, remet in zip(magmets_alt, rename_alt):
        df[remet] = df[met]

    dfmm_alt = pd.concat([df[df[hue]=='iterative'].sort_values('rmse')[:20],
                      df[df[hue]=="direct"].sort_values('rmse')[:20],
                      df[df[hue]=="deep learning"].sort_values('rmse')[:20],
                      #df[df[hue]=="hybrid"].sort_values('rmse')[:20]
                     ])
    sns.pairplot(dfmm_alt[rename_alt + [hue]], hue=hue, markers=markers[:dfmm[hue].unique().size], palette='bright')
    plt.savefig(imagepath / 'pairplot_algotype_best20_rmse_altmetrics_{}.png'.format(inputfile), bbox_inches='tight', dpi=300)
    plt.savefig(imagepath / 'pairplot_algotype_best20_rmse_altmetrics_{}.svg'.format(inputfile))

    # still more pairplots for additional metrics
    magmets_alt = ['rmse', 'MI', 'MAD', 'GXE']
    rename_alt = ['NRMSE', 'Mutual Inform', 'MAD', 'GXE']

    for met, remet in zip(magmets_alt, rename_alt):
        df[remet] = df[met]

    dfmm_alt = pd.concat([df[df[hue]=='iterative'].sort_values('rmse')[:20],
                      df[df[hue]=="direct"].sort_values('rmse')[:20],
                      df[df[hue]=="deep learning"].sort_values('rmse')[:20],
                      #df[df[hue]=="hybrid"].sort_values('rmse')[:20]
                     ])
    sns.pairplot(dfmm_alt[rename_alt + [hue]], hue=hue, markers=markers[:dfmm[hue].unique().size], palette='bright')
    plt.savefig(imagepath / 'pairplot_algotype_best20_rmse_morealtmetrics_{}.png'.format(inputfile), bbox_inches='tight', dpi=300)
    plt.savefig(imagepath / 'pairplot_algotype_best20_rmse_morealtmetrics_{}.svg'.format(inputfile))

    print()