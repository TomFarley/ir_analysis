#!/usr/bin/env python

"""


Created: 
"""

import logging, datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import calcam

import fire
import fire.interfaces.io_utils

from fire.interfaces import interfaces, user_config
from fire.plugins import plugins
from fire.plotting import debug_plots, image_figures, spatial_figures, temporal_figures, plot_tools
from fire.scripts.read_pickled_ir_data import read_data_for_pulses_pickle
from fire.physics import physics_parameters
from fire.misc import utils

logger = logging.getLogger(__name__)


# paths_figures = (fire_paths['user'] / 'figures/').resolve()

def review_analysed_shot_pickle(pulse, diag_tag_raw='rit', machine='mast_u', debug_figures=None, recompute=False,
                                show=True):
    logger.info(f'Reviewing {machine}, {diag_tag_raw}, {pulse}')
    print(f'Reviewing {machine}, {diag_tag_raw}, {pulse}')

    data, data_unpacked = read_data_for_pulses_pickle(diag_tag_raw=diag_tag_raw, pulses=pulse, machine=machine,
                                                      generate=True, recompute=recompute)[pulse]

    image_data = data['image_data']
    path_data = data['path_data']

    meta = data['meta_data']
    # meta = dict(pulse=pulse, camera=camera, machine=machine)

    review_analysed_shot(image_data, path_data, meta=meta, debug=debug_figures)

def review_analysed_shot(image_data, path_data, meta, debug=None, output=None):
    if debug is None:
        debug = {}
    if output is None:
        output = {}


    meta_data = meta

    # Required meta data
    pulse = meta['pulse']
    diag_tag_analysed = meta['diag_tag_analysed']
    diag_tag_raw = meta['diag_tag_raw']
    machine = meta['machine']
    files = meta['files']
    analysis_path_names = meta['analysis_path_names']
    analysis_path_keys = meta['analysis_path_keys']
    analysis_path_labels = meta.get('analysis_path_labels', analysis_path_names)

    # Optional data
    frame_data = image_data.get('frame_data')
    frame_times = image_data.get('t')

    calcam_calib = calcam.Calibration(load_filename=str(files['calcam_calib']))

    config, config_groups, config_path_fn = user_config.get_user_fire_config()
    base_paths = config_groups['fire_paths']

    paths_input = config['paths_input']['input_files']
    paths_output = {key: Path(str(path).format(**base_paths)) for key, path in config['paths_output'].items()}
    paths_figures = utils.make_iterable(config['user']['paths']['figures'])

    # Load machine plugins
    machine_plugin_paths = config['paths_input']['plugins']['machine']
    machine_plugin_attrs = config['plugins']['machine']['module_attributes']
    machine_plugins, machine_plugins_info = plugins.get_compatible_plugins(machine_plugin_paths,
                                                attributes_required=machine_plugin_attrs['required'],
                                                attributes_optional=machine_plugin_attrs['optional'],
                                                plugins_required=machine, plugin_type='machine', base_paths=base_paths)
    machine_plugins, machine_plugins_info = machine_plugins[machine], machine_plugins_info[machine]
    fire.active_machine_plugin = (machine_plugins, machine_plugins_info)

    t_end = physics_parameters.get_t_end_shot(path_data['power_total_vs_t_path0'], plot=False, t_end_min=0.1)
    # r_end = path_data['t'][path_data['heat_flux_total_path0'].where(path_data['t']> 0.05).argmin(dim='t')] + 0.05
    t_range = [0, t_end]

    # r_range = [None, 1.0]
    # r_range = [None, 1.65]
    r_range = None

    n_middle = int(np.floor(len(frame_data)/2))  # Frame number in middle of movie

    if debug.get('calcam_calib_image', False):
        frame_ref_pre_transforms = None
        debug_plots.debug_calcam_calib_image(calcam_calib, frame_data=frame_data, frame_ref=frame_ref_pre_transforms,
                                             n_frame_ref=n_middle, wire_frame=image_data['wire_frame'])

    # if debug.get('camera_shake', False):
    #     debug_plots.debug_camera_shake(pixel_displacements=pixel_displacemnts, n_shake_ref=n_shake_ref)

    # if debug.get('debug_detector_window', False):  # Need to plot before detector window applied to calibration
    #     debug_plots.debug_detector_window(detector_window=detector_window, frame_data=image_data,
    #                                       calcam_calib=calcam_calib, image_full_frame=calcam_calib_image_full_frame,
    #                                       image_coords=image_coords)

    if debug.get('movie_data_animation', False):
        image_figures.animate_frame_data(image_data, key='frame_data', nth_frame=1)
                        #                  n_start=40, n_end=350,
                        # save_path_fn=paths_output['gifs'] / f'{machine}-{pulse}-{camera}-frame_data.gif')

    if debug.get('movie_data_nuc_animation', False):
        image_figures.animate_frame_data(image_data, key='frame_data_nuc', nth_frame=1)

    if debug.get('movie_data_nuc', False):
        debug_plots.debug_movie_data(image_data, key='frame_data_nuc')

    if debug.get('specific_frames', False):
        n_check = 218
        debug_plots.debug_movie_data(image_data, frame_nos=np.arange(n_check, n_check+4), key='frame_data') #
        # frame_data_nuc

    if (debug.get('movie_intensity_stats', False)):
        # Force plot if there are any saturated frames
        temporal_figures.plot_movie_intensity_stats(frame_data, meta_data=meta_data)

    if (debug.get('movie_temperature_stats', False)):
        # Force plot if there are any saturated frames
        temporal_figures.plot_movie_intensity_stats(image_data.get('temperature_im'), meta_data=meta_data,
                                                    bit_depth=None)

    if debug.get('spatial_res', False):
        debug_plots.debug_spatial_res(image_data)

    # if figures.get('spatial_res', False):
    #     fn_spatial_res = path_figures / f'spatial_res_{pulse}_{camera}.png'
    #     image_figures.figure_spatial_res_max(image_data, clip_range=[None, 20], save_fn=fn_spatial_res, show=True)

    if False and debug.get('spatial_coords', False):
        phi_offset = 112.3
        # points_rzphi = [[0.80, -1.825, phi_offset-42.5], [1.55, -1.825, phi_offset-40]]  # R, z, phi - old IDL
        points_rzphi = None  # R, z, phi
        # analysis path
        # points_rzphi = [[0.80, -1.825, -42.5]]  # R, z, phi - old IDL analysis path
        # points_pix = np.array([[256-137, 320-129], [256-143, 320-0]]) -1.5  # x_pix, y_pix - old IDL analysis path
        points_pix = np.array([[256-133, 320-256], [256-150, 320-41]]) -1.5  # x_pix, y_pix - old IDL analysis path: [256, 133], [41, 150]
        # points_pix = None
        # points_pix plotted as green crosses
        debug_plots.debug_spatial_coords(image_data, points_rzphi=points_rzphi, points_pix=points_pix)

    if debug.get('surfaces', False):
        debug_plots.debug_surfaces(image_data)

    if debug.get('temperature_im', False):
        debug_plots.debug_temperature_image(image_data)

    if (debug.get('movie_temperature_animation', False) or debug.get('movie_temperature_animation_gif', False)):
        overwrite_gif = False
        if debug.get('movie_temperature_animation_gif', False):
            fn = f'{machine}-{diag_tag_raw}-{pulse}_temperature_movie.gif'
            save_path_fn = paths_output['gifs'] / diag_tag_raw / fn
            if save_path_fn.is_file() and (not overwrite_gif):  # As writing gif is slow don't repeat write
                save_path_fn = None
        else:
            save_path_fn = None
        show = debug.get('movie_temperature_animation', False)

        # duration = 15
        duration = 25
        # cbar_range = [0, 99.8]  # percentile of range
        cbar_range = [0, 99.7]  # percentile of range
        # cbar_range = [0, 99.95]  # percentile of range
        # cbar_range = [0, 100]  # percentile of range
        # cbar_range = None
        # frame_range = [40, 270]
        frame_range = [40, 410]
        # frame_range = [40, 470]
        frame_range = np.clip(frame_range, *meta_data['frame_range'])
        image_figures.animate_frame_data(image_data, key='temperature_im', nth_frame=1, duration=duration,
                                         n_start=frame_range[0], n_end=frame_range[1], save_path_fn=save_path_fn,
                                         cbar_range=cbar_range,
                                         frame_label=f'{diag_tag_analysed.upper()} {pulse} $t=${{t:0.1f}} ms',
                                         cbar_label='$T$ [$^\circ$C]',
                                         label_values={'t': frame_times.values*1e3}, show=show)
        if (debug.get('movie_temperature_animation_gif', False) and
                (not debug.get('movie_temperature_animation', False)) and ()):
            plot_tools.close_all_mpl_plots(close_all=True, verbose=True)

    for i_path, (analysis_path_key, analysis_path_name) in enumerate(zip(analysis_path_keys, analysis_path_names)):
        path = analysis_path_key
        meta_data['path_label'] = analysis_path_labels[i_path]

        if debug.get('poloidal_cross_sec', False):
            spatial_figures.figure_poloidal_cross_section(image_data=image_data, path_data=path_data, pulse=pulse, no_cal=True,
                                                                        show=True)
        if debug.get('spatial_coords_raw', False):
            coord_keys = ('x_im', 'y_im',
                          'R_im', 'phi_deg_im', 'z_im',
                          'ray_lengths_im', 'sector_im', 'wire_frame')
            debug_plots.debug_spatial_coords(image_data, path_data=path_data, path_name=analysis_path_key,
                                             coord_keys=coord_keys)

        if debug.get('spatial_coords_derived', False):
            coord_keys = ('s_global_im', 'wire_frame',
                          'R_im', 'phi_deg_im', 'z_im',
                          'ray_lengths_im', 'R_im', 'R_im')
            debug_plots.debug_spatial_coords(image_data, path_data=path_data, path_name=analysis_path_key,
                                             coord_keys=coord_keys)

        if debug.get('temperature_vs_R_t-robust', False):
            save_path_fn = None
            robust_percentiles = (35, 99)
            # robust_percentiles = (35, 99.5)
            debug_plots.debug_plot_profile_2d(path_data, param='temperature', path_names=analysis_path_key,
                                              extend='both',
                                              robust=True, meta=meta_data, machine_plugins=machine_plugins,
                                              label_tiles=True, t_range=t_range, x_range=r_range,
                                              robust_percentiles=robust_percentiles,
                                              set_data_coord_lims_with_ranges=True, save_path_fn=save_path_fn,
                                              show=True)
        if debug.get('temperature_vs_R_t-raw', False):
            save_path_fn = None
            debug_plots.debug_plot_profile_2d(path_data, param='temperature', path_names=analysis_path_key,
                                              robust=False, meta=meta_data, machine_plugins=machine_plugins,
                                              label_tiles=True, t_range=t_range, x_range=r_range,
                                              robust_percentiles=None,
                                              set_data_coord_lims_with_ranges=True, save_path_fn=save_path_fn,
                                              show=True)


        if debug.get('heat_flux_vs_R_t-robust', False) or debug.get('heat_flux_vs_R_t-robust-save', False):
            if debug.get('heat_flux_vs_R_t-robust-save', False):
                fn = f'heat_flux_vs_R_t-robust-{machine}_{diag_tag_analysed}_{pulse}.png'
                save_path_fn = (paths_output['figures'] / 'heat_flux_vs_R_t-robust' / diag_tag_analysed / fn)
            else:
                save_path_fn = None
            show = debug.get('heat_flux_vs_R_t-robust', False)

            # robust = False
            # robust_percentiles = (30, 90)
            # robust_percentiles = (30, 98)
            robust_percentiles = (35, 99.5)
            # robust_percentiles = (45, 99.7)
            # robust_percentiles = (50, 99.8)
            # robust_percentiles = (2, 98)
            # robust_percentiles = (2, 99)
            # robust_percentiles = (2, 99.5)
            # robust_percentiles = (2, 100)
            extend = 'both'
            # extend = 'min'
            # extend = 'neither'
            num = 'heat_flux_vs_R_t-robust'
            debug_plots.debug_plot_profile_2d(path_data, param='heat_flux', path_names=analysis_path_key, extend=extend,
                                              robust=True, meta=meta_data, machine_plugins=machine_plugins, num=num,
                                              label_tiles=True, t_range=t_range, x_range=r_range,
                                              robust_percentiles=robust_percentiles,
                                              set_data_coord_lims_with_ranges=True, save_path_fn=save_path_fn,
                                              mark_peak=False,
                                              show=show)
            if (debug.get('heat_flux_vs_R_t-robust-save', False) and (not debug.get('heat_flux_vs_R_t-robust', False))):
                plot_tools.close_all_mpl_plots(close_all=True, verbose=True)

        if (debug.get('heat_flux_vs_R_t-raw', False) or debug.get('heat_flux_vs_R_t-raw-save', False)):
            if debug.get('heat_flux_vs_R_t-raw-save', False):
                fn = f'heat_flux_vs_R_t-raw-{machine}_{diag_tag_analysed}_{pulse}.png'
                save_path_fn = (paths_output['figures'] / 'heat_flux_vs_R_t-raw' / diag_tag_analysed / fn)
            else:
                save_path_fn = None
            show = debug.get('heat_flux_vs_R_t-raw', False)
            robust = False
            num = 'heat_flux_vs_R_t-raw'

            debug_plots.debug_plot_profile_2d(path_data, param='heat_flux', path_names=analysis_path_key, num=num,
                                              extend='neither', robust=robust, meta=meta_data,
                                              t_range=t_range, x_range=r_range, mark_peak=False,
                                              machine_plugins=machine_plugins, show=show, save_path_fn=save_path_fn)
            if (debug.get('heat_flux_vs_R_t-raw-save', False) and (not debug.get('heat_flux_vs_R_t-raw', False))):
                plot_tools.close_all_mpl_plots(close_all=True, verbose=True)

        if debug.get('analysis_path', False):
            # TODO: Finish  debug_analysis_path_2d
            # debug_plots.debug_analysis_path_2d(image_data, path_data, path_names=analysis_path_key,
            #                        image_data_in_cross_sections=True, machine_plugins=machine_plugins)
            debug_plots.debug_analysis_path_1d(image_data, path_data, path_names=analysis_path_key,
                           image_data_in_cross_sections=True, machine_plugins=machine_plugins,
                           pupil_coords=meta.get('calcam_pupilpos'),
                           keys_profiles=(
                           ('frame_data_min(i)_{path}', 'frame_data_mean(i)_{path}', 'frame_data_max(i)_{path}'), #  TODO uncomment when pickle fixed
                           ('temperature_min(i)_{path}', 'temperature_mean(i)_{path}', 'temperature_max(i)_{path}'),
                           ('heat_flux_min(i)_{path}', 'heat_flux_mean(i)_{path}', 'heat_flux_max(i)_{path}'),
                           ('s_global_{path}', 'R_{path}'),
                           ('ray_lengths_{path}',),
                           ('spatial_res_x_{path}', 'spatial_res_y_{path}', 'spatial_res_linear_{path}')
                           ))

        if debug.get('timings', False):
            debug_plots.debug_plot_timings(path_data, pulse=pulse, meta_data=meta_data)

        if debug.get('strike_point_loc', False):
            heat_flux = path_data[f'heat_flux_amplitude_global_peak_{path}'].values
            heat_flux_thresh = np.nanmin(heat_flux) + 0.03 * (np.nanmax(heat_flux)-np.nanargmin(heat_flux))
            debug_plots.debug_plot_temporal_profile_1d(path_data, params=('heat_flux_R_peak',),
                                                       path_name=analysis_path_keys, x_var='t',
                                                       heat_flux_thresh=heat_flux_thresh, meta_data=meta_data,
                                                       machine_plugins=machine_plugins)
            debug_plots.debug_plot_temporal_profile_1d(path_data, params=('heat_flux_R_peak', 'heat_flux_amplitude_global_peak'),
                                                       path_name=analysis_path_keys, x_var='t',
                                                       heat_flux_thresh=heat_flux_thresh, meta_data=meta_data,
                                                       machine_plugins=machine_plugins)
        if output.get('strike_point_loc', False):
            path_fn = Path(paths_output['csv_data']) / f'strike_point_loc-{machine}-{diag_tag_analysed}-{pulse}.csv'
            fire.interfaces.io_utils.to_csv(path_fn, path_data, cols=f'heat_flux_R_peak_{path}', index='t',
                                            x_range=[0, 0.6], drop_other_coords=True, verbose=True)

        if True:   # debug.get('power_to_target', False):
            debug_plots.plot_energy_to_target(path_data, params=('power_total_vs_t', 'energy_total_vs_R',
                                                                 'cumulative_energy_vs_t', 'cumulative_energy_vs_R'),
                                       path_name=analysis_path_keys[0], meta_data=meta_data,
                                       machine_plugins=machine_plugins)




def review_shot(pulse=None, camera = 'rit', machine='mast_u', recompute=False, show=True):
    import pyuda
    client = pyuda.Client()

    if pulse is None:
        # pulse = 43183  # Nice strike point sweep to T5, but negative heat fluxes
        # pulse = 43177
        # pulse = 43530
        # pulse = 43534
        # pulse = 43547
        # pulse = 43415  # Peter Ryan's strike point sweep for LP checks

        # pulse = 43583  # 2xNBI - Kevin choice
        # pulse = 43587  #
        # pulse = 43591  #
        # pulse = 43596  #
        # pulse = 43610  #
        # pulse = 43620  #
        # pulse = 43624  #
        # pulse = 43662  #

        # pulse = 43624  #
        # pulse = 43648  #

        # pulse = 43611
        # pulse = 43613
        # pulse = 43614

        # pulse = 43415  # LP and IR data --
        # pulse = 43412  # LP and IR data --

        pulse = 43805  # Strike point sweep to T5 - good data for IR and LP
        # pulse = 43823  # Strike point very split on T2 at t=0.4-0.5 s
        # pulse = 43835  # Strike point split
        # pulse = 43852
        # pulse = 43854  # Rapid strike point sweep to T5
        # pulse = 43836

        # pulse = 43937
        # pulse = 43839

        # pulse = 43859
        # pulse = 43916
        # pulse = 43917
        # pulse = 43922

        # pulse = 43952  # Strike point sweep to T5
        # pulse = 43955  # Evidence of T4 ripple and T5 compensation
        # pulse = 43987  # V broad strike point
        # pulse = 43513  # Clean up ref shot - no IR data

        # pulse = 43995  # Super-X
        # pulse = 43996  # Super-X
        # pulse = 43998  # Super-X
        # pulse = 43999  # Super-X
        # pulse = 44000  # Super-X, detached
        # pulse = 44003  # LM
        # pulse = 44004  # LM
        # pulse = 43835  # Lidia strike point splitting request - good data

        # pulse = 44006  # beams

        # pulse = 44021  # LM
        # pulse = 44022  # LM
        # pulse = 44023  # LM
        # pulse = 44024  # LM
        # pulse = 44025  # LM

        # pulse = 43992  # virtual circuit keep SP on T2
        # pulse = 43998  # Super-X
        # pulse = 43400  # virtual circuit keep SP on T5

        # pulse = 44158  # virtual circuit keep SP on T5
        # pulse = 44092  # virtual circuit keep SP on T5
        # pulse = 44459  # Locked mode
        # pulse = 44461  # Locked mode

        # pulse = 44359  # RT18
        # pulse = 44394  # RT18
        # pulse = 44396  # RT18

        # pulse = 44463  # first irircam automatic aquisition
        # pulse = 44717

        # pulse = 44677  #  swept SXD (attached?)

        # pulse = 44758  #  compass scan
        pulse = 44776  #  compass scan
        # pulse = 44777  #  compass scan
        # pulse = 44778  #  compass scan
        # pulse = 44779  #  compass scan
        # pulse = 44780  #  compass scan
        # pulse = 44781  #  compass scan
        # pulse = 44782  #  compass scan
        # pulse = 44783  #  compass scan
        # pulse = 44784  #  compass scan
        # pulse = 44785  #  compass scan

        # pulse = 44852  #  TF test shot with gas

        pulse = 43952  #  RIR calcam calibration shot

        # pulse = 44822  #  Detachment hysteresis - marf
        # pulse = 44223  #  Fuelling ramp hysteresis - marf

        # pulse = 44606  # RT14 lambda_q
        # pulse = 55931  # RT14 lambda_q

        # pulse = 44842  # 600 kA conv, steady fueling without marfe
        # pulse = 44678  # 750 kA conv, steady fueling without marfe
        pulse = 44679  # 750 kA conv, steady fueling with marfe

        pulse = 44677  # Standard pulse JH suggests comparing with all diagnostics - RT18 slack, time to eurofusion

    debug = {'calcam_calib_image': False, 'debug_detector_window': False,
             'movie_intensity_stats': True,
         'movie_data_animation': False, 'movie_data_nuc_animation': False,
             'movie_temperature_animation': False,
             'movie_temperature_animation_gif': True,
             'spatial_coords_raw': False, 'spatial_coords_derived': False,
         'spatial_res': False,
         'movie_data_nuc': False, 'specific_frames': False, 'camera_shake': False, 'temperature_im': False,
         'surfaces': False,
         'analysis_path': True,
         'temperature_vs_R_t-raw': True,
         'temperature_vs_R_t-robust': False,
         'heat_flux_vs_R_t-robust': True, 'heat_flux_vs_R_t-raw': True,
             'heat_flux_vs_R_t-raw-save': True,
             'heat_flux_vs_R_t-robust-save': True,
         'timings': True, 'strike_point_loc': False,
         # 'heat_flux_path_1d': True,
         'power_to_target': True,
         }
    debug = {k: True for k in debug}
    if show is False:
        debug = {k: False for k in debug}

    review_analysed_shot_pickle(pulse=pulse, diag_tag_raw=camera, machine=machine, debug_figures=debug,
                                recompute=recompute)
    pass

def review_shot_list(shots=None, camera='rit', recompute_pickle=False, copy_recent_shots=False, show=True):
    from fire.interfaces.uda_utils import latest_uda_shot_number
    from ir_tools.data_formats.organise_movie_files_from_diag_pc import copy_raw_files_from_staging_area

    if shots is None:
        # shots = np.arange(44547, 44558)
        # shots = np.arange(44776, 44788)
        # shots = np.arange(44700, 44866)[::-1]
        # shots = [44815,44818,44819, 44820]  # Bob WPTE shots
        shots = [44801, 44805, 44894, 44896, 44903]  # Omkar shots
        # shots = [44786, 44787]
        # shots = [44547, 44548, 44550, 44551, 44554, 44555, 44556, 44558]  # missing 44552, 44553,
        # shots = [44776, 44777, 44778, 44779, 44781, 44782, 44783, 44784, 44785, 44786, 44787][::-1]

    n_shots = len(shots)

    debug = {'calcam_calib_image': False, 'debug_detector_window': False,
             'movie_intensity_stats': True, 'movie_temperature_stats': True,
             'movie_data_animation': False, 'movie_data_nuc_animation': False,
             'movie_temperature_animation': True,
             'movie_temperature_animation_gif': False,
             'spatial_coords': True, 'spatial_res': True,
             'movie_data_nuc': False, 'specific_frames': False, 'camera_shake': False, 'temperature_im': False,
             'surfaces': False, 'analysis_path': True,
             'temperature_vs_R_t': False,
             'heat_flux_vs_R_t-robust': True,
             'heat_flux_vs_R_t-raw': False,
             'heat_flux_vs_R_t-robust-save': True,
             'heat_flux_vs_R_t-raw-save': True,
             'timings': False,
             'strike_point_loc': False,
             'power_to_target': True,
         }
    # debug = {k: False for k in debug}
    # debug = {k: True for k in debug}
    # debug['movie_temperature_animation_gif'] = False
    # debug['movie_temperature_animation_gif'] = False

    logger.info(f'Reviewing shots: {shots}')

    if copy_recent_shots:
        copy_raw_files_from_staging_area(today=True, n_files=np.min([n_shots, 4]), write_ipx=True, overwrite_ipx=False)

    logger.setLevel(logging.WARNING)
    status = {'success': [], 'fail': []}

    for shot in shots:
        try:
            review_analysed_shot_pickle(pulse=shot, diag_tag_raw=camera, debug_figures=debug, recompute=recompute_pickle,
                                        show=show)
        except Exception as e:
            logger.exception(f'Failed to reivew shot {shot}')
            status['fail'].append(shot)
        else:
            status['success'].append(shot)
            print()
    print(f'Finished review of shots {shots}: \n{status}')

def review_latest_shots(n_shots=1, camera='rit', copy_recent_shots=True, n_shots_skip=0, recompute=False, show=True):
    from ir_tools.automation.automation_tools import latest_uda_shot_number
    # from ir_tools.data_formats.organise_movie_files_from_diag_pc import (copy_raw_files_from_staging_area,
    #                                                                      convert_ats_files_archive_to_ipx)
    # from fire.scripts.organise_movie_files_from_diag_pc import (copy_raw_files_from_staging_area, convert_ats_files_archive_to_ipx)
    from fire.plugins.machine_plugins import mast_u

    shot_start = latest_uda_shot_number()
    shots = np.arange(shot_start, shot_start-n_shots, -1) - n_shots_skip  # [::-1]

    debug = {'calcam_calib_image': False, 'debug_detector_window': False,
             'movie_intensity_stats': True, 'movie_temperature_stats': True,
             'movie_data_animation': False, 'movie_data_nuc_animation': False,
             'movie_temperature_animation': False,
             'movie_temperature_animation_gif': True,
             'spatial_coords': True, 'spatial_res': True,
             'movie_data_nuc': False, 'specific_frames': False, 'camera_shake': False, 'temperature_im': False,
             'surfaces': False, 'analysis_path': True,
             'temperature_vs_R_t': False,
             'heat_flux_vs_R_t-robust': True,
             'heat_flux_vs_R_t-raw': True,
             'heat_flux_vs_R_t-robust-save': True,
             'heat_flux_vs_R_t-raw-save': True,
             'timings': True,
             'strike_point_loc': False,
             'power_to_target': True,
         }
    if not show:
        debug = {k: False if ('save' not in k) else debug[k] for k in debug}

    logger.info(f'Reviewing shots: {shots}')

    path_archive = '/home/tfarley/data/movies/diagnostic_pc_transfer/{camera}/{date}/'
    if copy_recent_shots:
        organise_recent_shots(camera=camera, date='today', path_archive=path_archive, n_files=n_shots)

    logger.setLevel(logging.WARNING)
    status = {'success': [], 'fail': []}

    for shot in shots:
        try:
            review_analysed_shot_pickle(pulse=shot, diag_tag_raw=camera, debug_figures=debug, recompute=recompute)
        except Exception as e:
            logger.exception(f'Failed to review shot {shot}')
            status['fail'].append(shot)
            info = mast_u.pulse_meta_data(shot, keys=('exp_date', 'exp_time'))
            if 'exp_date' in info:
                date = info['exp_date']
                organise_recent_shots(camera=camera, date=date, path_archive=path_archive, n_files=n_shots)
        else:
            status['success'].append(shot)
            print()
    print(f'Finished review of shots {shots}: \n{status}')

def organise_recent_shots(camera='rit', date='today', n_files=2, overwrite_ipx=False,
                          path_archive='/home/tfarley/data/movies/diagnostic_pc_transfer/{camera}/{date}/'):
    # from fire.scripts.organise_movie_files_from_diag_pc import copy_raw_files_from_staging_area, convert_ats_files_archive_to_ipx
    from ir_tools.data_formats.organise_movie_files_from_diag_pc import (copy_raw_files_from_staging_area,
                                                                         convert_ats_files_archive_to_ipx)
    if date == 'today':
        date = datetime.datetime.now().strftime("%Y-%m-%d")

    path_archive = path_archive.format(camera=camera, date=date)

    if camera == 'rit':
        copy_raw_files_from_staging_area(date=date, n_files=n_files, path_archive=path_archive, write_ipx=True,
                                         overwrite_ipx=overwrite_ipx)
    elif camera == 'rir':
        path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/rir/{date}/'
        fn_meta = '/home/tfarley/data/movies/mast_u/rir_ats_files/rir_meta.json'
        convert_ats_files_archive_to_ipx(pulses=None, path_in=path_in, copy_ats_file=False, fn_meta=fn_meta, date=date,
                                         n_files=n_files)

if __name__ == '__main__':
    # review_shot(44683, camera='rit', machine='mast_u', recompute=True)  # RT18 slow sweep - raw images as well
    # review_shot(45272, camera='rit', machine='mast_u', recompute=True)  # missing data?
    # review_shot(44910, camera='rit', machine='mast_u', recompute=False)  # Low density locked modes
    # review_shot(45101, camera='rit', machine='mast_u', recompute=False)  # First time datac software working in central (external trigger) mode
    # review_shot(44677, camera='rir', machine='mast_u')
    # review_shot(29541, camera='rir', machine='mast')
    # review_shot(43183, camera='rit', machine='mast_u', show=True)
    # review_shot_list(camera='rit', recompute=False, shots=[44697, 44700, 44702, 44703], show=True)  # P Ryan XD shots

    # shots = [  # James Harrison's end of EXH-01 experiments
        # 45468, 45469, 45470, 45473,
        #      45439, 45443, 45444, 45446, 45450, 45456, 45459, 45461, 45462, 45463, 45464, 45465]
    # shots = [  # Jame's list of PEX KPI shots for re-analysis
    #     43610, 43611, 43612, 43613, 43614, 43643, 43644,
    #     43665, 43666, 43667, 43591, 43589, 43596]
    # shots = [45266, 45270, 45271, 45272, 45282, 45285, 45286, 45340, 45341, 45360, 45387, 45388]  # AKirk H-mode
    # shots = [45060, 45062]  # Yacopo DSF measurements
    # shots = [45112, 45113, 45480, 45481, 45484]  # Vlad snowflake
    # shots = [45419, 45420]  # EXH=06 power ballance
    # shots = [44697, 44699, 44700, 44702, 44797, 44607]  # EXH-16 X-divertor
    # shots = [45419]  # Strike point sweep for EFIT comparison
    # shots = [45470]  # JRH paper
    # shots = [43795, 43804, 45360]  # calcam calibration shots
    # shots = [45360, 45388]  # alpha param tuning shots - CDC H-mode
    shots = [45388]  #
    # shots = [45448]  #
    review_shot_list(camera='rit', recompute_pickle=False, shots=shots, show=True, copy_recent_shots=False)
    # review_latest_shots(camera='rit', n_shots=1, n_shots_skip=3, copy_recent_shots=True, recompute=False, show=True)
