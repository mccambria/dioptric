# -*- coding: utf-8 -*-
"""
Functions, etc to be used mainly by other utils. If you're running into
a circular reference in utils, put the problem code here. 

Created September 10th, 2021

@author: mccambria
"""

import platform
from pathlib import Path
import socket
import json
from importlib import import_module
import sys
import labrad
import numpy as np


def get_config_module(pc_name=None):
    if pc_name is None:
        pc_name = socket.gethostname()
    module_name = f"config.{pc_name}"
    module = import_module(module_name)
    return module


def get_config_dict(pc_name=None):
    module = get_config_module(pc_name)
    return module.config


def get_default_email():
    config = get_config_dict()
    return config["default_email"]


def _get_os_config_val(key):
    os_name = platform.system()  # Windows or Linux
    os_name_lower = os_name.lower()
    config = get_config_dict()
    val = config[f"{os_name_lower}_{key}"]
    return val


def get_nvdata_path():
    """Returns an OS-dependent Path to the nvdata directory"""
    return _get_os_config_val("nvdata_path")


def get_repo_path():
    """Returns an OS-dependent Path to the repo directory"""
    return _get_os_config_val("repo_path")


def get_server(server_name):
    config = get_config_dict()
    dev_name = config["Servers"][server_name]
    return eval(f"cxn.{dev_name}")


# region LabRAD registry utilities - deprecated in favor of config file


def get_registry_entry(cxn, key, directory):
    """Get an entry from the LabRAD registry"""
    p = cxn.registry.packet()
    p.cd("", *directory)
    p.get(key)
    return p.send()["get"]


def _labrad_get_config_dict(cxn=None):
    """Get the whole config from the registry as a dictionary"""
    if cxn is None:
        with labrad.connect() as cxn:
            return _labrad_get_config_dict_sub(cxn)
    else:
        return _labrad_get_config_dict_sub(cxn)


def _labrad_get_config_dict_sub(cxn):
    config_dict = {}
    _labrad_populate_config_dict(cxn, ["", "Config"], config_dict)
    return config_dict


def _labrad_populate_config_dict(cxn, reg_path, dict_to_populate):
    """Populate the config dictionary recursively"""

    # Sub-folders
    cxn.registry.cd(reg_path)
    sub_folders, keys = cxn.registry.dir()
    for el in sub_folders:
        sub_dict = {}
        sub_path = reg_path + [el]
        _labrad_populate_config_dict(cxn, sub_path, sub_dict)
        dict_to_populate[el] = sub_dict

    # Keys
    if len(keys) == 1:
        cxn.registry.cd(reg_path)
        p = cxn.registry.packet()
        key = keys[0]
        p.get(key)
        val = p.send()["get"]
        if type(val) == np.ndarray:
            val = val.tolist()
        dict_to_populate[key] = val

    elif len(keys) > 1:
        cxn.registry.cd(reg_path)
        p = cxn.registry.packet()
        for key in keys:
            p.get(key)
        vals = p.send()["get"]

        for ind in range(len(keys)):
            key = keys[ind]
            val = vals[ind]
            if type(val) == np.ndarray:
                val = val.tolist()
            dict_to_populate[key] = val


# endregion

if __name__ == "__main__":
    print(_labrad_get_config_dict())

{'apd_indices': [0], 'nv_sig_units': "{'coords': 'V', 'exp
ected_count_rate': 'kcps', 'durations': 'ns', 'magnet_angl
e': 'deg', 'resonance': 'GHz', 'rabi': 'ns', 'uwave_power'
: 'dBm'}", 'shared_email': 'kolkowitznvlab@gmail.com', 'wi
ndows_nvdata_path': WindowsPath('E:/Shared drives/Kolkowit
z Lab Group/nvdata'), 'linux_nvdata_path': WindowsPath('C:
/Users/kolkowitz/E/nvdata'), 'windows_repo_path': WindowsP
ath('C:/Users/kolkowitz/Documents/GitHub/dioptric'), 'linu
x_repo_path': WindowsPath('C:/Users/kolkowitz/Documents/Gi
tHub/dioptric'), 'CommonDurations': {'cw_meas_buffer': 500
0, 'pol_to_uwave_wait_dur': 5000, 'scc_ion_readout_buffer'
: 1000, 'uwave_buffer': 1000, 'uwave_to_readout_wait_dur':
 1000}, 'DeviceIDs': {'piezo_stage_616_3cd_model': 'E727',
 'piezo_stage_616_3cd_serial': '0121089079', 'rotation_sta
ge_ell18k_address': 'COM5', 'signal_generator_tsg4104a_vis
a_address': 'TCPIP0::128.104.160.112::5025::SOCKET', 'QM_o
px_ip': '128.104.160.117'}, 'Microwaves': {'sig_gen_TEKT_t
sg4104a': {'delay': 260}, 'iq_comp_amp': 0.5, 'iq_delay':
0, 'sig_gen_HIGH': 'sig_gen_TEKT_tsg4104a', 'sig_gen_LOW':
 'sig_gen_TEKT_tsg4104a'}, 'Optics': {'cobolt_515': {'dela
y': 400, 'feedthrough': False, 'mod_type': <ModTypes.DIGIT
AL: 1>}, 'cobolt_638': {'delay': 300, 'feedthrough': False
, 'mod_type': <ModTypes.DIGITAL: 1>}, 'laserglow_589': {'d
elay': 1750, 'feedthrough': False, 'mod_type': <ModTypes.A
NALOG: 2>}}, 'PhotonCollection': {'qm_opx_max_readout_time
': 5000000}, 'Positioning': {'pos_xy_server': 'pos_xyz_PI_
616_3cd_digital', 'pos_xyz_server': 'pos_xyz_PI_616_3cd_di
gital', 'pos_z_server': 'pos_xyz_PI_616_3cd_digital', 'xy_
control_style': <ControlStyle.STEP: 1>, 'xy_delay': 500000
00, 'xy_dtype': 'float', 'xy_nm_per_unit': 1000, 'xy_optim
ize_range': 0.95, 'xy_server': 'pos_xyz_PI_616_3cd_digital
', 'xy_small_response_delay': 800, 'xy_units': 'um', 'xyz_
positional_accuracy': 0.002, 'xyz_server': 'pos_xyz_PI_616
_3cd_digital', 'xyz_timeout': 1, 'z_control_style': <Contr
olStyle.STEP: 1>, 'z_delay': 50000000, 'z_dtype': 'float',
 'z_nm_per_unit': 1000, 'z_optimize_range': 4, 'z_server':
 'pos_xyz_PI_616_3cd_digital', 'z_small_response_delay': 5
0000000, 'z_units': 'nm'}, 'Servers': {'arb_wave_gen': 'QM
_opx', 'counter': 'QM_opx', 'magnet_rotation': 'rotation_s
tage_THOR_ell18k', 'pos_xy': 'pos_xyz_PI_616_3cd_digital',
 'pos_xyz': 'pos_xyz_PI_616_3cd_digital', 'pos_z': 'pos_xy
z_PI_616_3cd_digital', 'pulse_gen': 'QM_opx', 'sig_gen_HIG
H': 'sig_gen_TEKT_tsg4104a', 'sig_gen_LOW': 'sig_gen_TEKT_
tsg4104a', 'tagger': 'QM_opx'}, 'Wiring': {'PulseGen': {'d
o_apd_0_gate': 1, 'do_apd_1_gate': 0, 'do_integrated_520_d
m': 5, 'do_sample_clock': 0}, 'QmOpx': {'ao_laserglow_589_
am': 5, 'do_cobolt_515_dm': 9}, 'Tagger': {'di_apd_0': 10,
 'di_apd_1': 10, 'di_apd_gate': 10}}}

(dioptric) C:\Users\kolkowitz\Documents\GitHub\dioptric> c
: && cd c:\Users\kolkowitz\Documents\GitHub\dioptric && cm
d /C "C:\Users\kolkowitz\miniconda3\envs\dioptric\python.e
xe c:\Users\kolkowitz\.vscode\extensions\ms-python.python-
2023.12.0\pythonFiles\lib\python\debugpy\adapter/../..\deb
ugpy\launcher 59572 -- C:\Users\kolkowitz\Documents\GitHub
\dioptric\utils\common.py "
<function _labrad_get_config_dict at 0x00000165B69056C0>

(dioptric) C:\Users\kolkowitz\Documents\GitHub\dioptric> c
: && cd c:\Users\kolkowitz\Documents\GitHub\dioptric && cm
d /C "C:\Users\kolkowitz\miniconda3\envs\dioptric\python.e
xe c:\Users\kolkowitz\.vscode\extensions\ms-python.python-
2023.12.0\pythonFiles\lib\python\debugpy\adapter/../..\deb
ugpy\launcher 59584 -- C:\Users\kolkowitz\Documents\GitHub
\dioptric\utils\common.py "
Enter username, or blank for the global user (localhost:76
82):
Enter LabRAD password (localhost:7682):
{'CommonDurations': {'cw_meas_buffer': 5000, 'pol_to_uwave
_wait_dur': 5000, 'scc_ion_readout_buffer': 10000, 'space_
opti_interval_m': 4, 'uwave_buffer': 1000, 'uwave_to_reado
ut_wait_dur': 5000}, 'DeviceIDs': {'arb_wave_gen_visa_addr
ess': 'TCPIP0::128.104.160.119::5025::SOCKET', 'daq0_name'
: 'Dev1', 'filter_slider_ell9k_2_address': 'COM11', 'filte
r_slider_ell9k_3_address': 'COM9', 'filter_slider_ell9k_ad
dress': 'COM5', 'gcs_dll_path': 'C:\\Users\\kolkowitz\\Doc
uments\\GitHub\\kolkowitz-nv-experiment-v1.0\\servers\\out
puts\\GCSTranslator\\PI_GCS2_DLL_x64.dll', 'objective_piez
o_model': 'E709', 'objective_piezo_serial': '0119008970',
'piezo_stage_626_2cd_model': 'E727', 'piezo_stage_626_2cd_
serial': '0116058375', 'pulse_gen_SWAB_82_ip': '128.104.16
0.111', 'rotation_stage_ell18k_address': 'COM6', 'sig_gen_
BERK_bnc835_visa': 'TCPIP::128.104.160.114::inst0::INSTR',
 'sig_gen_STAN_sg394_visa': 'TCPIP::128.104.160.118::inst0
::INSTR', 'sig_gen_TEKT_tsg4104a_visa': 'TCPIP0::128.104.1
60.112::5025::SOCKET', 'tagger_SWAB_20_serial': '1740000JE
H', 'temp_ctrl_tc200': 'COM10', 'z_piezo_kpz101_serial': '
29502179'}, 'Microwaves': {'sig_gen_BERK_bnc835': {'delay'
: 151, 'fm_mod_bandwidth': 100000.0}, 'sig_gen_STAN_sg394'
: {'delay': 104, 'fm_mod_bandwidth': 100000.0}, 'sig_gen_T
EKT_tsg4104a': {'delay': 57}, 'iq_comp_amp': 0.5, 'iq_dela
y': 630, 'uwave_amplifiers': 56}, 'Optics': {'cobolt_515':
 {'am_feedthrough': 'False', 'delay': 120, 'mod_type': 'Mo
dTypes.DIGITAL', 'wavelength': 515}, 'cobolt_638': {'am_fe
edthrough': 'False', 'delay': 80, 'logic_level_shifting_bo
ard': 'True', 'mod_type': 'ModTypes.DIGITAL', 'wavelength'
: 638}, 'collection': {'FilterMapping': {'715_lp': 2, '715
_sp+630_lp': 0, '740_bp': 1, 'no_filter': 3}, 'filter_serv
er': 'filter_slider_ell9k_3'}, 'integrated_520': {'am_feed
through': 'False', 'delay': 250, 'logic_level_shifting_boa
rd': 'True', 'mod_type': 'ModTypes.DIGITAL', 'nd_filter':
1.5, 'wavelength': 520}, 'laser_LGLO_589': {'FilterMapping
': {'nd_0': 0, 'nd_0.5': 1, 'nd_1.0': 2, 'nd_1.5': 3}, 'am
_feedthrough': 'True', 'delay': 2500, 'filter_server': 'fi
lter_slider_ell9k', 'mod_type': 'ModTypes.DIGITAL', 'true_
zero_voltage_daq': 0.0, 'wavelength': 589}, 'laserglow_532
': {'FilterMapping': {'nd_0': 3, 'nd_0.5': 2, 'nd_1.0': 1,
 'nd_2.0': 0}, 'am_feedthrough': 'False', 'delay': 1030, '
filter_server': 'filter_slider_ell9k_2', 'mod_type': 'ModT
ypes.DIGITAL', 'wavelength': 532}}, 'Positioning': {'daq_v
oltage_range_factor': 5.0, 'piezo_stage_scaling_gain': 2.5
, 'piezo_stage_scaling_offset': 250, 'piezo_stage_voltage_
range_factor': 5.0, 'xy_control_style': 'ControlStyle.STRE
AM', 'xy_dtype': 'float', 'xy_incremental_step_size': 0.00
2, 'xy_large_response_delay': 2000000, 'xy_nm_per_unit': 8
0000, 'xy_optimize_range': 0.012, 'xy_positional_accuracy'
: 0.002, 'xy_small_response_delay': 500000, 'xy_timeout':
1, 'xy_units': 'V', 'z_control_style': 'ControlStyle.STREA
M', 'z_delay': 500000, 'z_dtype': 'float', 'z_hysteresis_a
': 0.06, 'z_hysteresis_b': 0.9, 'z_hysteresis_linearity':
0.98, 'z_incremental_step_size': 0.0125, 'z_nm_per_unit':
16000, 'z_optimize_range': 0.2, 'z_units': 'V'}, 'Servers'
: {'arb_wave_gen': 'awg_KEYS_33622A', 'charge_readout_lase
r': 'laser_LGLO_589', 'counter': 'tagger_SWAB_20', 'magnet
_rotation': 'rotation_stage_thor_ell18k', 'pos_xy': 'pos_x
yz_THOR_gvs212_PI_pifoc', 'pos_xyz': 'pos_xyz_THOR_gvs212_
PI_pifoc', 'pos_z': 'pos_xyz_THOR_gvs212_PI_pifoc', 'pulse
_gen': 'pulse_gen_SWAB_82', 'sig_gen_HIGH': 'sig_gen_STAN_
sg394', 'sig_gen_LOW': 'sig_gen_BERK_bnc835', 'sig_gen_omn
i': 'sig_gen_BERK_bnc835', 'sig_gen_single': 'sig_gen_STAN
_sg394', 'tagger': 'tagger_SWAB_20'}, 'Wiring': {'Daq': {'
ai_photodiode': 'Dev1/AI0', 'ai_thermistor_ref': 'dev1/AI1
', 'ao_galvo_x': 'dev1/AO0', 'ao_galvo_y': 'dev1/AO1', 'ao
_laser_LGLO_589_feedthrough': 'dev1/AO3', 'ao_objective_pi
ezo': 'dev1/AO2', 'ao_piezo_stage_626_2cd_x': 'dev1/AO0',
'ao_piezo_stage_626_2cd_y': 'dev1/AO1', 'ao_uwave_sig_gen_
mod': '', 'ao_z_piezo_kpz101': 'dev1/AO2', 'di_clock': 'PF
I12', 'di_laser_LGLO_589_feedthrough': 'PFI0'}, 'Piezo_sta
ge_E727': {'piezo_stage_channel_x': 4, 'piezo_stage_channe
l_y': 5}, 'PulseGen': {'ao_fm_sig_gen_BERK_bnc835': 1, 'ao
_fm_sig_gen_STAN_sg394': 0, 'ao_laser_LGLO_589_am': 1, 'do
_apd_gate': 5, 'do_arb_wave_trigger': 2, 'do_cobolt_638_dm
': 7, 'do_integrated_520_dm': 3, 'do_laser_LGLO_589_am': 6
, 'do_sample_clock': 0, 'do_sig_gen_BERK_bnc835_gate': 1,
'do_sig_gen_STAN_sg394_gate': 4}, 'Tagger': {'di_apd_0': 2
, 'di_apd_1': 4, 'di_apd_gate': 3, 'di_clock': 1}}, 'apd_i
ndices': [1], 'nv_sig_units': "{'coords': 'V', 'expected_c
ount_rate': 'kcps', 'durations': 'ns', 'magnet_angle': 'de
g', 'resonance': 'GHz', 'rabi': 'ns', 'uwave_power': 'dBm'
}"}