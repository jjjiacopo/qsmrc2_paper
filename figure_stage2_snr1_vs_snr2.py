"""
QSM Reconstruction Challenge 2019

Create pairplot with hue = algorithm_type

author: jakob.meineke@philips.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

    avgtemplate = 'average {} SNR1 + SNR2'
    difftemplate = '{} difference: SNR1 - SNR2'

    markers = ['o', 'X', 's', 'P']

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

        ax = plt.subplot(111)
        xs = df[avgtemplate.format(newname)].min(), df[avgtemplate.format(newname)].max()
        ymean, ystd = df[difftemplate.format(newname)].mean(), df[difftemplate.format(newname)].std()
        plt.hlines(ymean - ystd, xs[0], xs[1], 'lightgray', '--', zorder=-1)
        plt.hlines(ymean, xs[0], xs[1], 'lightgray', zorder=-1)
        plt.hlines(ymean + ystd, xs[0], xs[1], 'lightgray', '--', zorder=-1)
        sns.scatterplot(ax=ax, x=avgtemplate.format(newname), y=difftemplate.format(newname),
                        hue='algorithm type', style='algorithm type', palette='bright', data=df)
        #plt.legend(loc=(1.1, 0.5))

        plt.savefig(imagepath / 'stage2_snr1vssnr2_blandaltman_{}.png'.format(met), bbox_inches='tight', dpi=300)
        #plt.savefig(imagepath / 'stage2_snr1vssnr2_blandaltman_{}.svg'.format(met))
        plt.close()

    fig = plt.figure(figsize=(16, 4))
    for num, (met, met_renamed) in enumerate(zip(metrics, metrics_renamed)):
        ax = fig.add_subplot(1, 3, num+1)
        df[f'{met_renamed} Stage1'] = df_stage1[met]
        df[f'{met_renamed} Stage2'] = df[avgtemplate.format(met_renamed)]
        if num>0:
            legend = False
        else:
            legend = 'brief'
        sns.scatterplot(x=f'{met_renamed} Stage1', y=f'{met_renamed} Stage2',
                        hue='algorithm type', style='algorithm type', palette='bright',
                        ax=ax, data=df,
                        legend=legend)
        if met == 'CalcStreak':
            #ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x)))
            #ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x)))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        plt.plot(np.sort(df[f'{met_renamed} Stage1']), np.sort(df[f'{met_renamed} Stage1']), 'lightgray', zorder=-1)
    plt.savefig(imagepath / f'stage1_vs_stage2.png', bbox_inches='tight', dpi=300)

    # pairplots for stage2, this should be in another file, but here we have everything together already.
    hue = "algorithm type"
    magmets = ['rmse', #'rmse_detrend_Blood',
               #'rmse_detrend_DGM',
               'CalcStreak',
               'DeviationFromCalcMoment']
    rename = ['NRMSE', #'NRMSEd Blood',
              #'NRMSEd DeepGM',
              'Calcif. Streaking',
              'Calcif. Moment Error']

    df['rmse'] = 0.5 * (df1['rmse'] + df2['rmse'])
    for met, remet in zip(magmets, rename):
        df[remet] = 0.5 * (df1[met] + df2[met])

    dfmm = pd.concat([df[df[hue]=='iterative'].sort_values('rmse')[:20],
                      df[df[hue]=="direct"].sort_values('rmse')[:20],
                      df[df[hue]=="deep learning"].sort_values('rmse')[:20],
                      #df[df[hue]=="hybrid"].sort_values('rmse')[:20]
                     ])
    sns.pairplot(dfmm[rename + [hue]], hue=hue, markers=markers[:dfmm[hue].unique().size], palette='bright')
    plt.savefig(imagepath / 'pairplot_algotype_best20_rmse_visual_rmseddgm_calc_stage2.png', bbox_inches='tight', dpi=300)
    plt.savefig(imagepath / 'pairplot_algotype_best20_rmse_visual_rmseddgm_calc_stage2.svg')

    # more pairplots for additional metrics
    magmets_alt = ['rmse', 'XSIM', 'HFEN', 'CC']
    rename_alt = ['NRMSE', 'XSIM', 'HFEN', 'Corr Coeff']

    for met, remet in zip(magmets_alt, rename_alt):
        df[remet] = 0.5 * (df1[met] + df2[met])

    dfmm_alt = pd.concat([df[df[hue]=='iterative'].sort_values('rmse')[:20],
                      df[df[hue]=="direct"].sort_values('rmse')[:20],
                      df[df[hue]=="deep learning"].sort_values('rmse')[:20],
                      #df[df[hue]=="hybrid"].sort_values('rmse')[:20]
                     ])
    sns.pairplot(dfmm_alt[rename_alt + [hue]], hue=hue, markers=markers[:dfmm[hue].unique().size], palette='bright')
    plt.savefig(imagepath / 'pairplot_algotype_best20_rmse_altmetrics_stage2.png', bbox_inches='tight', dpi=300)
    plt.savefig(imagepath / 'pairplot_algotype_best20_rmse_altmetrics_stage2.svg')

    # still more pairplots for additional metrics
    magmets_alt = ['rmse', 'MI', 'MAD', 'GXE']
    rename_alt = ['NRMSE', 'Mutual Inform', 'MAD', 'GXE']

    for met, remet in zip(magmets_alt, rename_alt):
        df[remet] = 0.5 * (df1[met] + df2[met])

    dfmm_alt = pd.concat([df[df[hue]=='iterative'].sort_values('rmse')[:20],
                      df[df[hue]=="direct"].sort_values('rmse')[:20],
                      df[df[hue]=="deep learning"].sort_values('rmse')[:20],
                      #df[df[hue]=="hybrid"].sort_values('rmse')[:20]
                     ])
    sns.pairplot(dfmm_alt[rename_alt + [hue]], hue=hue, markers=markers[:dfmm[hue].unique().size], palette='bright')
    plt.savefig(imagepath / 'pairplot_algotype_best20_rmse_morealtmetrics_stage2.png', bbox_inches='tight', dpi=300)
    plt.savefig(imagepath / 'pairplot_algotype_best20_rmse_morealtmetrics_stage2.svg')


    print()