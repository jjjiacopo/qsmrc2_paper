"""
QSM Reconstruction Challenge 2019

Preprocessing of csv-files before further analysis

author: jakob.meineke@philips.com
"""

import numpy as np
import pandas as pd
from pathlib import Path

from translation import translation

def add_translation_columns(df):
    for key in ['algorithm type', 'solution space']:
        df[key] = np.array([translation[key][x] for x in df['Algorithm-type']])
    for key in ['regularization_class', 'regularization_tv', 'regularization_tgv', 'regularization_l2']:
        df[key] = np.array([translation[key][x] for x in df['Regularization terms']])
    df['input'] = np.array([translation['input'].get(x, 'unknown') for x in df[
        'Did your algorithm use the provided frequency map or the four individual echo phase images?']])
    df['magnitude info'] = np.array(
        [x == 'Yes' for x in df['Does your algorithm incorporate information derived from magnitude images?']])
    return df

if __name__=="__main__":

    datapath = Path('data')
    if not datapath.exists():
        datapath.mkdir(parents=True)

    stage1 = pd.read_csv(datapath / 'master_database_stage1_final.csv', encoding='latin-1')
    # throw out GT from plotting
    stage1 = stage1[stage1.Sim2 != 'GT']
    stage1 = add_translation_columns(stage1)
    stage1['Visual'] = stage1[['Streaking', 'Unnaturalness', 'NoiseVisual']].mean(axis=1)

    # sort by submission ID and reindex
    stage1 = stage1.sort_values('Submission Identifier').reset_index(drop=True)
    stage1.to_pickle(datapath / 'stage1.pkl')

    print(1)