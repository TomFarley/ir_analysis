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

from fire import fire_paths, copy_default_user_settings
from fire.scripts.scheduler_workflow import scheduler_workflow, copy_output

logger = logging.getLogger(__name__)
logger.propagate = False



def run_jet():  # pragma: no cover
    pulse = 94935  # Split view example 715285
    camera = 'kldt'
    pass_no = 0
    machine = 'JET'
    scheduler = False
    magnetics = False
    update_checkpoints = False
    # update_checkpoints = True
    debug = {'debug_detector_window': False, 'camera_shake': True,
             'movie_data_animation': False, 'movie_data_nuc_animation': False,
             'spatial_coords': True, 'spatial_res': False, 'movie_data_nuc': False,
             'temperature_im': False, 'surfaces': False, 'analysis_path': True}
    # debug = {k: True for k in debug}
    # debug = {k: False for k in debug}
    figures = {'spatial_res': False}
    logger.info(f'Running JET scheduler workflow...')
    status = scheduler_workflow(pulse=pulse, camera=camera, pass_no=pass_no, machine=machine, scheduler=scheduler,
                       equilibrium=magnetics, update_checkpoints=update_checkpoints, debug=debug, figures=figures)
    return status

def run_mast_rir():  # pragma: no cover
    # pulse = 23586  # Full frame with clear spatial calibration
    pulse = 26505  # Full frame OSP only louvre12d, 1D analysis profile, HIGH current - REQUIRES NEW CALCAM CALIBRATION
    # pulse = 26489  # Full frame OSP only, 1D analysis profile, MODERATE current - REQUIRES NEW CALCAM CALIBRATION
    # pulse = 28866  # Low power, (8x320)
    # pulse = 29210  # High power, (8x320) - Lots of bad frames/missing data?
    # pulse = 29936  # Full frame, has good calcam calib
    # pulse = 30378  # High ELM surface temperatures ~450 C

    # pulse = 24688  # full frame - requires new callibration - looking at inner strike point only
    # pulse = 26098  # full frame - no?
    # pulse = 30012  # full frame
    # pulse = 29945  # [  0  80 320  88] TODO: Further check detector window aligned correctly?

    # pulse = 28623  # 700kA

    # Pulses with bad detector window meta data (512 widths?): 23775, 26935

    # # pulse_range_rand = [23586, 28840]
    # # pulse_range_rand = [28840, 29936]
    # pulse_range_rand = [29936, 30471]
    # pulse = int(pulse_range_rand[0] + np.diff(pulse_range_rand)[0] * np.random.rand())

    camera = 'rir'
    pass_no = 0
    machine = 'MAST'
    scheduler = False
    magnetics = False
    update_checkpoints = False
    # update_checkpoints = True
    debug = {'debug_detector_window': False, 'camera_shake': False,
             'movie_data_animation': False, 'movie_data_nuc_animation': False,
             'spatial_coords': True, 'spatial_res': False, 'movie_data_nuc': False,
             'surfaces': False, 'analysis_path': True, 'temperature_im': False}
    # debug = {k: True for k in debug}
    debug = {k: False for k in debug}
    figures = {'spatial_res': False}
    logger.info(f'Running MAST scheduler workflow...')
    status = scheduler_workflow(pulse=pulse, camera=camera, pass_no=pass_no, machine=machine, scheduler=scheduler,
                       equilibrium=magnetics, update_checkpoints=update_checkpoints, debug=debug, figures=figures)
    return status

def run_mast_rit():  # pragma: no cover
    # pulse = 30378
    # pulse = 29936
    # pulse = 27880  # Full frame upper divertor
    # pulse = 29000  # 80x256 upper divertor
    # pulse = 28623   # 700kA - no data?
    pulse = 26798   # MAST 400kA
    camera = 'rit'

    pass_no = 0
    machine = 'MAST'
    scheduler = False
    magnetics = False
    update_checkpoints = False
    # update_checkpoints = True
    debug = {'debug_detector_window': True, 'camera_shake': True,
             'movie_data_animation': False, 'movie_data_nuc_animation': False,
             'spatial_coords': True, 'spatial_res': False, 'movie_data_nuc': False,
             'surfaces': False, 'analysis_path': True, 'temperature_im': False,}
    # debug = {k: True for k in debug}
    debug = {k: False for k in debug}
    figures = {'spatial_res': False}
    logger.info(f'Running MAST scheduler workflow...')
    status = scheduler_workflow(pulse=pulse, camera=camera, pass_no=pass_no, machine=machine, scheduler=scheduler,
                       equilibrium=magnetics, update_checkpoints=update_checkpoints, debug=debug, figures=figures)
    return status

def run_mastu_rir():  # pragma: no cover
    # pulse = 50000  # CAD view calibrated from test installation images - no plasma
    # pulse = 50001  # Test movie consisting of black body cavity calibration images

    # pulse = 44726  #
    # pulse = 44673  # DN-700-SXD-OH
    # pulse = 43952  # early focus

    # pulse = 44677  # Standard pulse JH suggests comparing with all diagnostics - RT18 slack, time to eurofusion
    pulse = 44982  # Error field shot

    camera = 'rir'
    pass_no = 0
    machine = 'MAST_U'

    scheduler = False
    magnetics = False
    update_checkpoints = False
    # update_checkpoints = True

    # TODO: Remove redundant movie_data step
    debug = {'calcam_calib_image': False, 'debug_detector_window': False,
             'movie_intensity_stats-raw': True,
             'movie_intensity_stats-corrected': True,
             'movie_intensity_stats-nuc': True,
             'dark_level': False,
             'movie_data_animation': False, 'movie_data_nuc_animation': False,
             'movie_temperature_animation': False,
             'spatial_coords': False,
             'spatial_res': False,
             'movie_data_nuc': True, 'specific_frames': False, 'camera_shake': False, 'temperature_im': False,
             'surfaces': False, 'analysis_path': False,
             'path_cross_sections': False,
             'temperature_vs_R_t': True,
             'heat_flux_vs_R_t-raw': True,
             'heat_flux_vs_R_t-robust': True,
             'timings': True,
             'strike_point_loc': True,
             # 'heat_flux_path_1d': True,
             }

    output = {'strike_point_loc': True, 'raw_frame_image': False}

    # debug = {k: True for k in debug}
    # debug = {k: False for k in debug}
    figures = {'spatial_res': False, 'heat_flux_vs_R_t-robust': True}
    logger.info(f'Running {machine} {camera} scheduler workflow...')
    status = scheduler_workflow(pulse=pulse, camera=camera, pass_no=pass_no, machine=machine, scheduler=scheduler,
                       equilibrium=magnetics, update_checkpoints=update_checkpoints, debug=debug, figures=figures,
                                output=output)
    return status

def run_mastu_rit():  # pragma: no cover
    # pulse = 50000  # CAD view calibrated from test installation images - no plasma
    # pulse = 50001  # Test movie consisting of black body cavity calibration images
    # pulse = 50002  # IRCAM Works raw file for debugging
    # pulse = 43141  # Early diverted plasma on T2-T4
    pulse = 43183  # Early diverted plasma on T2-T5
    # pulse = 43163  # TODO: check frame rate and expsure meta data on LWIR PC
    # pulse = 43412  # Peter Ryan's strike point sweep based on 43391 for LP checks
    # pulse = 43413  # Peter Ryan's strike point sweep based on 43391 for LP checks
    # pulse = 43415  # Peter Ryan's strike point sweep for LP checks
    # pulse = 43524  # Double NBI in
    # pulse = 43530  # Peter Ryan's strike point sweep for LP checks
    # pulse = 43534  # T5 sweep
    # pulse = 43535  # T5 sweep with NBI
    # pulse = 43547  # T5 sweep with NBI
    # pulse = 43561  # NBI
    # pulse = 43575  # NBI
    # pulse = 43583  # NBI
    # pulse = 43584  # NBI
    # pulse = 43591
    # pulse = 43587
    # pulse = 43610
    # pulse = 43643
    # pulse = 43644
    # pulse = 43648
    # pulse = 43685

    # pulse = 43610  # MU KPI
    # pulse = 43611
    # pulse = 43613
    # pulse = 43614
    # pulse = 43591
    # pulse = 43596
    # pulse = 43415
    # pulse = 43644
    # pulse = 43587

    # Peter's list of shots with a strike point sweep to T5:
    # pulse = 43756  # LP, but NO IR data
    # pulse = 43755  # NO LP or IR data
    # pulse = 43529  # LP, but NO IR data
    # pulse = 43415  # LP and IR data --
    # pulse = 43412  # LP and IR data --
    # pulse = 43391  # no LP or IR data

    # pulse = 43753  # Lidia strike point splitting request - no data

    # pulse = 43805  # Strike point sweep to T5 - good data for IR and LP

    # pulse = 43823
    # pulse = 43835  # Lidia strike point splitting request - good data
    # pulse = 44004  # LM
    # pulse = 43835
    # pulse = 43852
    # pulse = 43836

    # pulse = 43839
    # pulse = 43996

    # pulse = 43998  # Super-X

    # pulse = 44463  # first irircam automatic aquisition

    # pulse = 44386

    # pulse = 44628  # DATAC software acquisition ftp'd straight to data store - access via UDA
    # pulse = 44697  # DATAC software acquisition ftp'd straight to data store - access via UDA
    # pulse = 44628  # IPX file written by datac software

    # pulse = 44677   # RT18 - Swept (attached?) super-x
    # pulse = 44683   # RT18 - Attached T5 super-x

    # pulse = 44695   # 400 kA
    # pulse = 44613   # 400 kA

    # pulse = 44550   # error field exp
    # pulse = 44776   # error field exp
    # pulse = 44777   # error field exp
    # pulse = 44778   # error field exp
    # pulse = 44779   # error field exp

    # pulse = 44815   # RT13

    # pulse = 44852  #  TF test shot with gas

    # pulse = 44677  # Standard pulse JH suggests comparing with all diagnostics - RT18 slack, time to eurofusion

    # pulse = 44865  #

    # pulse = 44896


    # 44849 onwards should have uda efit

    camera = 'rit'
    pass_no = 0
    machine = 'MAST_U'

    scheduler = False
    magnetics = False
    update_checkpoints = False
    # update_checkpoints = True

    # TODO: Remove redundant movie_data step
    debug = {'calcam_calib_image': False, 'debug_detector_window': True,
             'movie_intensity_stats-raw': False,
             'movie_intensity_stats-corrected': True,
             'movie_intensity_stats-nuc': False,
             'dark_level': False,
             'movie_data_animation': False, 'movie_data_nuc_animation': False,
             'movie_temperature_animation': False,
             'spatial_coords': False,
             'spatial_res': False,
             'movie_data_nuc': False, 'specific_frames': False, 'camera_shake': False, 'temperature_im': False,
             'surfaces': False, 'analysis_path': False,
             'path_cross_sections': False,
             'temperature_vs_R_t': True,
             'heat_flux_vs_R_t-raw': True,
             'heat_flux_vs_R_t-robust': True,
             'timings': True,
             'strike_point_loc': False,
             # 'heat_flux_path_1d': True,
             }

    output = {'strike_point_loc': True, 'raw_frame_image': False}

    # debug = {k: True for k in debug}
    debug = {k: False for k in debug}
    figures = {'spatial_res': False, 'heat_flux_vs_R_t-robust': True}
    logger.info(f'Running MAST-U ait scheduler workflow...')
    status = scheduler_workflow(pulse=pulse, camera=camera, pass_no=pass_no, machine=machine,
                                scheduler=scheduler, equilibrium=magnetics, update_checkpoints=update_checkpoints,
                                debug=debug, figures=figures, output=output)
    return status


if __name__ == '__main__':
    # delete_file('~/.fire_config.json', verbose=True, raise_on_fail=True)
    copy_default_user_settings(replace_existing=True)

    # AIR, AIT, AIS, AIU, AIV
    # outputs = run_mast_rir()
    # outputs = run_mast_rit()
    outputs = run_mastu_rir()
    # outputs = run_mastu_rit()
    # outputs = run_jet()

    clean_netcdf = True
    copy_to_uda_scrach = True

    copy_output(outputs, copy_to_uda_scrach=copy_to_uda_scrach, clean_netcdf=clean_netcdf)