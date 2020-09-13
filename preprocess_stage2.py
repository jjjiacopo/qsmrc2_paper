"""
QSM Reconstruction Challenge 2019

Preprocessing of csv-files before further analysis

author: jakob.meineke@philips.com
"""

import numpy as np
import pandas as pd
from pathlib import Path


if __name__=="__main__":

    datapath = Path('data')
    if not datapath.exists():
        datapath.mkdir(parents=True)

    stage2_snr1 = pd.read_csv(datapath / 'master_database_stage2_snr1__final.csv', encoding='latin-1')
    stage2_snr2 = pd.read_csv(datapath / 'master_database_stage2_snr2__final.csv', encoding='latin-1')

    # sort by submission ID and reindex
    stage2_snr1 = stage2_snr1.sort_values('Submission Identifier of the corresponding Stage 1 submission').reset_index(drop=True)
    stage2_snr1.to_pickle(datapath / 'stage2_snr1.pkl')

    stage2_snr2 = stage2_snr2.sort_values('Submission Identifier of the corresponding Stage 1 submission').reset_index(drop=True)
    stage2_snr2.to_pickle(datapath / 'stage2_snr2.pkl')

    print(1)