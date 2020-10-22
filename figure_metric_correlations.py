"""
QSM Reconstruction Challenge 2019

Create figure showing correlations between metrics

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

    df['Mean Visual'] = 1/3 * (df['Streaking'] + df['Unnaturalness'] + df['NoiseVisual'])
    df['Summed Discrepancy'] = df[['DGM Dissimilarity to GT', 'WMGM Dissimilarity to GT', 'CALC Dissimilarity to GT', 'VEINS Dissimilarity to GT']].sum(axis=1)

    all_metrics = ['rmse', 'rmse_detrend', 'rmse_detrend_Tissue',
                   'rmse_detrend_Blood', 'rmse_detrend_DGM',
                   'DeviationFromLinearSlope', 'CalcStreak', 'DeviationFromCalcMoment',
                   'Streaking', 'Unnaturalness', 'NoiseVisual',
                   #'Mean Visual',
                   #'Summed Discrepancy'
                   'DGM Dissimilarity to GT', 'WMGM Dissimilarity to GT', 'CALC Dissimilarity to GT',
                   'VEINS Dissimilarity to GT'
                   ]

    toplength = 5
    df['topgroupinany'] = np.array(
        [0 for x in df['Does your algorithm incorporate information derived from magnitude images?']])
    for met in all_metrics:
        df = df.sort_values(met)
        df['topgroupinany'].values[:toplength] += 1

    top = df[df.topgroupinany > 0]

    mymetrics = ['rmse',
                 'rmse_detrend',
                 'rmse_detrend_Tissue',
                 'rmse_detrend_Blood',
                 'rmse_detrend_DGM',
                 'DeviationFromLinearSlope',
                 'CalcStreak',
                 'DeviationFromCalcMoment',
                 'Streaking', 'Unnaturalness', 'NoiseVisual',
                 #'Mean Visual',
                 #'Summed Discrepancy'
                 'DGM Dissimilarity to GT',
                 'WMGM Dissimilarity to GT',
                 'CALC Dissimilarity to GT',
                 'VEINS Dissimilarity to GT']
    myrenamed = ['NRMSE',
                 'dNRMSE',
                 'dNRMSE Tissue',
                 'dNRMSE Blood',
                 'dNRMSE DeepGM',
                 'Slope Error',
                 'Calcif. Streaking',
                 'Calcif. Moment Error',
                 'Visual Streaking', 'Visual Unnaturalness', 'Visual Noise',
                 #'Mean Visual',
                 #'Summed Discrepancy'
                 'DGM Dissimilarity',
                 'WMGM Dissimilarity',
                 'Calcif. Dissimilarity',
                 'Veins Dissimilarity'
                 ]
    for m, n in zip(mymetrics, myrenamed):
        top[n] = top[m]
        df[n] = df[m]
    dfscorr = top[myrenamed].corr()
    dfscorr_all = df[myrenamed].corr()
    dfscorr_80 = df[df.rmse<80][myrenamed].corr()

    f = plt.figure(figsize=(8, 8))
    plt.matshow(dfscorr, fignum=f.number)
    plt.xticks(range(dfscorr.shape[1]), dfscorr.columns, fontsize=12, rotation=90)
    plt.yticks(range(dfscorr.shape[1]), dfscorr.columns, fontsize=12)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.savefig(imagepath / 'category_correlations_within_top{}_{}.png'.format(toplength, inputfile), bbox_inches='tight', dpi=300)

    f = plt.figure(figsize=(8, 8))
    plt.matshow(dfscorr_all, fignum=f.number)
    plt.xticks(range(dfscorr_all.shape[1]), dfscorr_all.columns, fontsize=12, rotation=90)
    plt.yticks(range(dfscorr_all.shape[1]), dfscorr_all.columns, fontsize=12)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.savefig(imagepath / 'category_correlations_all_{}.png'.format(inputfile), bbox_inches='tight', dpi=300)

    f = plt.figure(figsize=(8, 8))
    plt.matshow(dfscorr_80, fignum=f.number)
    plt.xticks(range(dfscorr_80.shape[1]), dfscorr_80.columns, fontsize=12, rotation=90)
    plt.yticks(range(dfscorr_80.shape[1]), dfscorr_80.columns, fontsize=12)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.savefig(imagepath / 'category_correlations_80_{}.png'.format(inputfile), bbox_inches='tight', dpi=300)

    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(dfscorr_80, annot=True, ax=ax, vmin=0, fmt=".2f")
    plt.savefig(imagepath / 'sns_category_correlations_80_{}.png'.format(inputfile), bbox_inches='tight', dpi=300)

    sns.clustermap(dfscorr_80, center=0, cmap="vlag", linewidths = 1, figsize = (10, 10))
    plt.savefig(imagepath / 'metrics_clustermap_80_{}.png'.format(inputfile), bbox_inches='tight', dpi=300)

    # version with all metrics
    mymetrics_ = ['Streaking', 'Unnaturalness', 'NoiseVisual',
                  'DGM Dissimilarity to GT', 'WMGM Dissimilarity to GT', 'CALC Dissimilarity to GT', 'VEINS Dissimilarity to GT',
                  'mean visual', 'rmse', 'rmse_detrend', 'rmse_detrend_Tissue', 'rmse_detrend_Blood', 'rmse_detrend_DGM',
                  'DeviationFromLinearSlope', 'CalcStreak', 'DeviationFromCalcMoment',
                  'SSIM', 'XSIM', 'HFEN', 'CC', 'MI',
                  'MAD', 'GXE']
    myrenamed_ = ['Visual Streaking', 'Visual Unnaturalness', 'Visual Noise',
                  'DGM Dissimilarity', 'WMGM Dissimilarity', 'Calcif. Dissimilarity', 'Veins Dissimilarity',
                  'Mean Visual', 'NRMSE', 'dNRMSE', 'dNRMSE Tissue', 'dNRMSE Blood', 'dNRMSE DGM',
                  'Slope Error', 'Calcif. Streaking', 'Calcif. Moment Error',
                  '1-SSIM', '1-XSIM', 'HFEN', '1-CC', '1-MI',
                  'MAD', 'GXE']

    ddf = df.copy()
    for m, n in zip(mymetrics_, myrenamed_):
        if m in ['SSIM', 'XSIM', 'CC', 'MI']:
            ddf[n] = 1-ddf[m]
        else:
            ddf[n] = ddf[m]
    ddfscorr_80 = ddf[ddf.rmse<80][myrenamed_].corr()

    f = plt.figure(figsize=(8, 8))
    plt.matshow(ddfscorr_80, fignum=f.number)
    plt.xticks(range(ddfscorr_80.shape[1]), ddfscorr_80.columns, fontsize=12, rotation=90)
    plt.yticks(range(ddfscorr_80.shape[1]), ddfscorr_80.columns, fontsize=12)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.savefig(imagepath / 'category_correlations_ALL_80_{}.png'.format(inputfile), bbox_inches='tight', dpi=300)

    f, ax = plt.subplots(figsize=(18, 12))
    sns.heatmap(ddfscorr_80, annot=True, ax=ax, vmin=0, fmt=".2f")
    plt.savefig(imagepath / 'sns_category_correlations_ALL_80_{}.png'.format(inputfile), bbox_inches='tight', dpi=300)

    sns.clustermap(ddfscorr_80, center=0, cmap="vlag", linewidths = 1, figsize = (10, 10))
    plt.savefig(imagepath / 'metrics_clustermap_ALL_80_{}.png'.format(inputfile), bbox_inches='tight', dpi=300)

    # for comparison between correlations from stage1 and stage2 reduce the data to those present in both stages

    # data for stage2
    df2_1 = pd.read_pickle(datapath / 'stage2_snr1.pkl')
    df2_2 = pd.read_pickle(datapath / 'stage2_snr2.pkl')

    intersection_submissions = set(df2_1['Submission Identifier of the corresponding Stage 1 submission']).intersection(set(df2_2['Submission Identifier of the corresponding Stage 1 submission']))
    df1 = df.copy().set_index('Submission Identifier')
    df1 = df1.loc[df1.index.intersection(intersection_submissions)]
    df2_1 = df2_1.set_index('Submission Identifier of the corresponding Stage 1 submission')
    df2_2 = df2_2.set_index('Submission Identifier of the corresponding Stage 1 submission')
    df2_1 = df2_1.loc[df2_1.index.intersection(intersection_submissions)]
    df2_2 = df2_2.loc[df2_2.index.intersection(intersection_submissions)]

    # for stage1 without visual
    # version with all metrics but without visual
    mymetrics__ = ['rmse', 'rmse_detrend', 'rmse_detrend_Tissue', 'rmse_detrend_Blood', 'rmse_detrend_DGM',
                  'DeviationFromLinearSlope', 'CalcStreak', 'DeviationFromCalcMoment',
                  'SSIM', 'XSIM', 'HFEN', 'CC', 'MI',
                  'MAD', 'GXE']
    myrenamed__ = ['NRMSE', 'dNRMSE', 'dNRMSE Tissue', 'dNRMSE Blood', 'dNRMSE DGM',
                  'Slope Error', 'Calcif. Streaking', 'Calcif. Moment Error',
                  '1-SSIM', '1-XSIM', 'HFEN', '1-CC', '1-MI',
                  'MAD', 'GXE']

    # stage1
    ddf1 = df1.copy()
    for m, n in zip(mymetrics_, myrenamed_):
        if m in ['SSIM', 'XSIM', 'CC', 'MI']:
            ddf1[n] = 1-ddf1[m]
        else:
            ddf1[n] = ddf1[m]

    ddf1corr_novis = ddf1[myrenamed__].corr()

    f, ax = plt.subplots(figsize=(18, 12))
    sns.heatmap(ddf1corr_novis, annot=True, ax=ax, vmin=0, fmt=".2f")
    plt.savefig(imagepath / 'sns_STAGE1_category_correlations_ALLNOVISUAL_{}.png'.format(inputfile), bbox_inches='tight', dpi=300)

    sns.clustermap(ddf1corr_novis, center=0, cmap="vlag", linewidths = 1, figsize = (10, 10))
    plt.savefig(imagepath / 'metrics_STAGE1_clustermap_ALLNOVISUAL_{}.png'.format(inputfile), bbox_inches='tight', dpi=300)

    # prepare correlation data for stage2
    ddf2 = pd.DataFrame([])
    for m, n in zip(mymetrics__, myrenamed__):
        if m in ['SSIM', 'XSIM', 'CC', 'MI']:
            ddf2[n] = 1-0.5*(df2_1[m] + df2_2[m])
        else:
            ddf2[n] = 0.5*(df2_1[m] + df2_2[m])
    ddf2corr = ddf2[myrenamed__].corr()

    f, ax = plt.subplots(figsize=(18, 12))
    sns.heatmap(ddf2corr, annot=True, ax=ax, vmin=0, fmt=".2f")
    plt.savefig(imagepath / 'sns_STAGE2_category_correlations_ALL_{}.png'.format(inputfile), bbox_inches='tight', dpi=300)

    sns.clustermap(ddf2corr, center=0, cmap="vlag", linewidths=1, figsize=(10, 10))
    plt.savefig(imagepath / 'metrics_STAGE2_clustermap_ALL_{}.png'.format(inputfile), bbox_inches='tight', dpi=300)


    print()