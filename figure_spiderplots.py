"""
QSM Reconstruction Challenge 2019

Create spiderplots for top 5 submissions

author: jakob.meineke@philips.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from radarplot import radar_factory

if __name__=="__main__":

    datapath = Path('data')
    imagepath = Path('images')
    if not imagepath.exists():
        imagepath.mkdir()

    inputfile = 'stage1'
    df = pd.read_pickle(datapath / (inputfile+'.pkl'))

    all_metrics = ['rmse', 'rmse_detrend', 'rmse_detrend_Tissue',
                   'rmse_detrend_Blood', 'rmse_detrend_DGM',
                   'DeviationFromLinearSlope', 'CalcStreak', 'DeviationFromCalcMoment',
                   'Visual'
                   ]

    toplength = 5
    df['topgroupinany'] = np.array(
        [0 for x in df['Does your algorithm incorporate information derived from magnitude images?']])
    for met in all_metrics:
        df = df.sort_values(met)
        df['topgroupinany'].values[:toplength] += 1

    top = df[df.topgroupinany > 0]
    top['descr'] = top['Regularization terms'] + ', ' + top['input'] + ', usemag' + top['magnitude info'].apply(str)
    top[all_metrics + ['Submission Identifier', 'descr', 'topgroupinany']].sort_values('topgroupinany', ascending=False)
    top = top.set_index('Submission Identifier')

    winners = {}
    for met in all_metrics:  # + ['visual']:
        win = df.sort_values(met).iloc[0]
        sub = win['Submission Identifier']
        if sub in winners.keys():
            winners[sub].append(met)
        else:
            winners[sub] = [met]
    print('winners')
    print(winners)

    intop5 = {}
    for met in all_metrics:  # + ['visual']:
        for i in range(toplength):
            win = df.sort_values(met).iloc[i]
            sub = win['Submission Identifier']
            if sub in intop5.keys():
                intop5[sub].append(met)
            else:
                intop5[sub] = [met]

    ordered_keys = sorted(list(intop5.keys()), key=lambda k: len(intop5[k]), reverse=True)
    for ok in ordered_keys:
        print(ok, len(intop5[ok]), intop5[ok])

    # compute scaled range for spiderplots
    spider_metrics = ['rmse', 'Visual', 'DeviationFromLinearSlope', 'CalcStreak', 'DeviationFromCalcMoment', 'rmse_detrend_DGM', 'rmse_detrend_Blood', ]
    dfmin = df.dropna().groupby('Regularization terms')[spider_metrics].min()
    datamax = dfmin.values.max(axis=0)

    print('dfmin.max(axis=0)')
    print(dfmin.max(axis=0))

    datamax = np.array([100.0, 3.0, 0.1, 0.1, 30.0, 100.0, 100.0])
    print('all_metrics')
    print(all_metrics)

    colors = ['#1f77b4',
              '#aec7e8',
              '#ff7f0e',
              '#ffbb78',
              '#2ca02c',
              '#98df8a',
              '#d62728',
              '#ff9896',
              '#9467bd',
              '#c5b0d5',
              '#8c564b',
              '#c49c94',
              '#e377c2',
              '#f7b6d2',
              '#7f7f7f',
              '#c7c7c7',
              '#bcbd22',
              '#dbdb8d',
              '#17becf',
              '#9edae5']

    name_to_color = {}
    for num, name in enumerate(top.index):
        name_to_color[name] = colors[(num) % len(colors)]

    # Sort by different metrics and plot
    for sortingmetric in ['rmse', 'Visual', 'CalcStreak']:
        dftop = top[top.rmse_detrend_Blood < 100].sort_values(sortingmetric)[spider_metrics + ['Preferred Acronym']][:5]

        N = len(spider_metrics)
        spoke_labels = spider_metrics
        theta = radar_factory(N, frame='polygon')

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(top=0.85, bottom=0.05)

        ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
        # ax.set_title("top 5 in '{}'".format(metric),  position=(0.5, 1.1), ha='center')

        data = dftop[spider_metrics].values
        names = dftop.index
        dataperc = 1 * data / datamax
        for d, name in zip(dataperc, names):
            line = ax.plot(theta, d, color=name_to_color[name], lw=2)
            # ax.fill(theta, d,  alpha=0.1)
        ax.set_rmax(1.0)
        ax.set_rmin(0.0)
        display_metrics = ['NRMSE', 'Visual', 'Slope Error', 'Calc. Streaking', 'Calcification Error', 'NRMSEd DeepGM',
                           'NRMSEd Blood']
        assert len(display_metrics) == len(spoke_labels)
        ax.set_varlabels(display_metrics)

        legend = ax.legend(dftop['Preferred Acronym'], loc=(1.1, .5),
                           labelspacing=0.1, fontsize='small')
        plt.savefig(imagepath / f"radarplot_top5any_{sortingmetric.replace(' ', '_')}.png", bbox_inches='tight', dpi=300)
        plt.savefig(imagepath / f"radarplot_top5any_{sortingmetric.replace(' ', '_')}.svg")