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

from fire.scripts.read_pickled_ir_data import read_data_for_pulses_pickle
from fire.plugins.plugins_movie import MovieReader
from fire.misc import utils
from fire.plotting import plot_tools
from fire.interfaces import io_utils, io_basic

logger = logging.getLogger(__name__)


def load_analysed_shot_pickle(pulse, camera='rit', machine='mast_u', recompute=False):
    logger.info(f'Reviewing {machine}, {camera}, {pulse}')
    print(f'Reviewing {machine}, {camera}, {pulse}')

    data, data_unpacked = read_data_for_pulses_pickle(diag_tag_raw=camera, pulses=pulse, machine=machine,
                                                      generate=True, recompute=recompute)[pulse]

    image_data = data['image_data']
    path_data = data['path_data']

    meta = data['meta_data']
    # meta = dict(pulse=pulse, camera=camera, machine=machine)

def extract_signals_from_shot_range(shots, camera='rit', machine='mast_u', signals=(), recompute=False):
    signals = dict(signals)
    data_scan = pd.DataFrame(data=None, index=(), columns=list(signals.keys()))

    for shot in shots:
        data, data_unpacked = read_data_for_pulses_pickle(diag_tag_raw=camera, pulses=shot, machine=machine,
                                                          generate=True, recompute=recompute)[shot]
        for signal, func in signals.items():
            data_scan.loc[shot, signal] = func(data)

    return data_scan

def extract_movie_meta_for_shot_range(shots, diag_tag_raw='rit', machine='mast_u', signals=(), plugin_filter=None,
                                      shots_skip=()):
    data_scan = pd.DataFrame(data=None, index=(), columns=list(signals))
    data_scan.index.name = 'shot'

    movie_reader = MovieReader(plugin_filter=plugin_filter)

    shots_read = []
    shots_missing = []
    shots_skipped = []
    for shot in shots:
        if shot in shots_skip:
            shots_skipped.append(shot)
            continue
        try:
            meta_data, origin = movie_reader.read_movie_meta_data(pulse=shot, diag_tag_raw=diag_tag_raw,
                                                                  machine=machine, )
        except IOError as e:
            logger.warning(f'Failed to read meta data for shot {shot}')
            data_scan.loc[shot, :] = np.nan  # Note will cast int columns to float, unless specify nan int dtype above
            shots_missing.append(shot)
        else:
            for signal in signals:
                value = meta_data.get(signal)
                if isinstance(value, str):
                    value = value.strip(' "\'')
                # if utils.is_numeric(value):
                #     dtype = type(value)
                #     data_scan.astype({signal: dtype})
                data_scan.loc[shot, signal] = value
            shots_read.append(shot)
            print(f'Read values for shot {shot}')

    return data_scan

def heat_flux_min(data):
    out = data['path_data']['heat_flux_path0'].min()
    return out

def heat_flux_max(data):
    out = data['path_data']['heat_flux_path0'].max()
    return out


if __name__ == '__main__':
    # shot_start = 44963
    shot_start = 29541  # Ryse  29435 has height=71
    # shot_start = 2800  #
    n_shots = 20
    stride = 1
    shots_trawl = np.arange(shot_start, shot_start - (n_shots + 1), -stride)
    # shots = np.arange(shot_start, shot_start+(n_shots+1), stride)
    print(shots_trawl)

    # camera = 'rir'
    diag_tag_raw = 'rit'
    # machine = 'mast_u'
    machine = 'mast'

    signals = ('view', 'width', 'height')
    signals += ('left', 'top', 'fps', 'exposure_in_us', 'lens', 'filter')
    signals += ('date_time',)
    signals += ('n_frames',)
    signals += ('sensor_temperature',)
    signals += ('trigger',)

    path_fn = f'./{machine}-{diag_tag_raw}.csv'

    overwrite_file = False  # Overwrite saved file with new data (shots outside current range will be lost)
    update_file = False  # Overwrite previously saved values

    if overwrite_file:
        io_basic.delete_file(path_fn)

    try:
        meta_read = pd.read_csv(path_fn, index_col='shot')
    except FileNotFoundError as e:
        print(f'No existing file for: {path_fn}')
        meta_read = None
        shots_read = ()
    else:
        shots_read = np.array(meta_read.index, dtype=int)
        print(f'Read data from {path_fn} for {len(shots_read)} from {np.min(shots_read)} to {np.max(shots_read)}')

    shots_skip = shots_read if ((not update_file) and (not overwrite_file)) else ()

    meta_trawled = extract_movie_meta_for_shot_range(shots_trawl, diag_tag_raw=diag_tag_raw, machine=machine,
                                                     plugin_filter=['ipx', 'uda'],
                                                     signals=signals, shots_skip=shots_skip)
    # print(meta_trawled)

    if (meta_read is not None) and (not overwrite_file):
        meta_all = pd.concat([meta_read, meta_trawled], axis=0, verify_integrity=True, sort=True)
        # meta_all = pd.merge(meta_read, meta_trawled, on='shot', sort=True, how='outer')
        shots_all = np.array(meta_all.index, dtype=int)
        print(f'Data combined for {len(shots_read)} from {np.min(shots_read)} to {np.max(shots_read)}')
    else:
        meta_all = meta_trawled

    meta_all.to_csv(path_fn, sep=',')
    print(f'Saved combined data to "{path_fn}"')

    height_max = meta_all["height"].max()
    # print(f'Max height: {height_max}')
    ind = np.where(meta_all['height'] == height_max)
    shots_max_height = meta_all.index[ind]
    print(f'Shots max height ({height_max}): {shots_max_height}')

    fig, axes, ax_passed = plot_tools.get_fig_ax(ax_grid_dims=(1, 1), axes_flatten=True)
    ax = axes[0]
    meta_all['height'].plot(marker='x', ls=':')
    plot_tools.show_if(True)


    # print(f'Shot for max: {meta["height"].argmax()}')


    # signals = dict(heat_flux_min=heat_flux_min, heat_flux_max=heat_flux_max)
    # data = extract_signals_from_shot_range(shots, signals=signals)

    pass