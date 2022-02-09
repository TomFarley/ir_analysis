#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from fire.interfaces import io_utils, io_basic

def compile_jet_lambda_q_values_inner():
    limiter = 'Inner'
    # limiter = 'Outer'
    path = Path(f'/home/tfarley/repos/ir_analysis/jet_2012_lamda_q_values/data/IndividualPulseSummaries/{limiter}'
                'Lim_IndividualPulseSummaries')

    fns = path.glob('*.txt')
    fns = sorted(list(fns))
    print(fns)

    col_names = [
        'pulse', 'cutoff', 'pitch_angle_filter_0', 'pitch_angle_filter_1',
        'I_p', 'I_p_error', 'n_e ', 'n_e_error', 'P_sol', 'P_sol_error', 'P_nbi ', 'P_nbi_error',
        'q_para0_far', 'q_para0_far_error', 'lambda_q_far', 'lambda_q_far_error', 'n_points_far', 'Chi2_far',
        'q_para0_near', 'q_para0_near_error', 'lambda_q_near', 'lambda_q_near_error', 'n_points_near', 'Chi2_near',
        'q_para0_ratio', 'lambda_q_ratio',]
    data = pd.DataFrame()
    for fn in fns:
        print(fn)
        df = io_basic.read_csv(fn, skiprows=2, names=col_names)
        data = data.append(df)

    data = data.set_index('pulse')
    print(data)

    cols_reduced = ['lambda_q_far', 'lambda_q_far_error', 'q_para0_far', 'q_para0_far_error', 'n_points_far', 'Chi2_far',
                    'lambda_q_near', 'lambda_q_near_error', 'q_para0_near', 'q_para0_near_error', 'n_points_near', 'Chi2_near',
                    'q_para0_ratio', 'lambda_q_ratio',
                    'I_p', 'I_p_error', 'n_e ', 'n_e_error', 'P_sol', 'P_sol_error', 'P_nbi ', 'P_nbi_error']
    data_reduced = data[cols_reduced]

    fn_out = f'lambda_q-JET_2012-tfarley-{limiter.lower()}_limter.csv'
    data_reduced.to_csv(fn_out, sep=',')
    data_reduced.plot('I_p', 'lambda_q_far')
    pass

def compile_jet_lambda_q_values_outer():
    limiter = 'Outer'
    path = Path(f'/home/tfarley/repos/ir_analysis/jet_2012_lamda_q_values/data/IndividualPulseSummaries/{limiter}'
                'Lim_IndividualPulseSummaries')

    fns = path.glob('*.txt')
    fns = sorted(list(fns))
    print(fns)

    col_names = [
        'pulse', 'cutoff', 'pitch_angle_filter_0', 'pitch_angle_filter_1',
        'I_p', 'I_p_error', 'n_e ', 'n_e_error', 'P_sol', 'P_sol_error', 'P_nbi ', 'P_nbi_error',
        'q_para0_far', 'q_para0_far_error', 'lambda_q_far', 'lambda_q_far_error', 'n_points_far', 'Chi2_far',
        'q_para0_near', 'q_para0_near_error', 'lambda_q_near', 'lambda_q_near_error', 'n_points_near', 'Chi2_near',
        'q_para0_ratio', 'lambda_q_ratio']
    data = pd.DataFrame()
    for fn in fns:
        print(fn)
        df = io_basic.read_csv(fn, skiprows=2, names=col_names)
        data = data.append(df)

    data = data.set_index('pulse')
    print(data)

    cols_reduced = ['lambda_q_far', 'lambda_q_far_error', 'q_para0_far', 'q_para0_far_error', 'n_points_far', 'Chi2_far',
                    # 'lambda_q_near', 'lambda_q_near_error', 'q_para0_near', 'q_para0_near_error', 'n_points_near', 'Chi2_near',
                    # 'q_para0_ratio', 'lambda_q_ratio',
                    'I_p', 'I_p_error', 'n_e ', 'n_e_error', 'P_sol', 'P_sol_error', 'P_nbi ', 'P_nbi_error']
    data_reduced = data[cols_reduced]

    fn_out = f'lambda_q-JET_2012-tfarley-{limiter.lower()}_limter.csv'
    data_reduced.to_csv(fn_out, sep=',')

    data_reduced.plot('I_p', 'lambda_q_far')
    pass


logger = logging.getLogger(__name__)


if __name__ == '__main__':
    compile_jet_lambda_q_values_outer()
    compile_jet_lambda_q_values_inner()
    plt.show()
    pass