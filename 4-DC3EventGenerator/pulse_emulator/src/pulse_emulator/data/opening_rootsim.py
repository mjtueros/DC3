from glob import glob

import numpy as np
import uproot

from pulse_emulator.utils import ns_to_us, sec_to_us

R2D = 180. / np.pi

def _get_all_event_numbers(root_dir):
    all_ev_num = []
    for tshower_file in sorted(glob(f"{root_dir}/shower_*_L0_*.root")):
        with uproot.open(tshower_file) as f:
            all_ev_num += list(f["tshower"]["event_number"].array())
    all_ev_num = np.array(all_ev_num)
    return all_ev_num

def _get_rootfiles_between_bounds(shower_meta_data_files, start, stop):
    n_events_per_file = []
    for met in shower_meta_data_files:
        with uproot.open(met) as f:
            n_events_per_file.append(f['tshower'].num_entries)
    n_events_per_file = np.array(n_events_per_file)
    if stop is None:
        stop = np.sum(n_events_per_file)

    last_indices = np.cumsum(n_events_per_file)
    first_indices = last_indices - n_events_per_file
    files_with_events_after_start = (last_indices > start)
    files_with_events_before_stop = (first_indices < stop)
    overlap = np.where(files_with_events_after_start & files_with_events_before_stop)[0]
    return overlap, first_indices[overlap], last_indices[overlap]

def _verify_xmax_pos(xmax_pos, shower_core_pos, zenith, azimuth, antenna_pos):

    k_p = shower_core_pos - xmax_pos
    k = k_p / np.linalg.norm(k_p, axis=1, keepdims=True)
    th_p = np.arccos(-k[:, 2])
    az_p = np.arctan2(-k[:, 1], -k[:, 0]) % (2 * np.pi)
    assert np.allclose(th_p * 180/np.pi, zenith * 180/np.pi, atol=1e-4, rtol=1e-4), f"{th_p*180/np.pi} vs {zenith*180/np.pi}"
    assert np.allclose(az_p * 180/np.pi, azimuth * 180/np.pi, atol=1e-4, rtol=1e-4), f"{az_p*180/np.pi} vs {azimuth*180/np.pi}"
    assert np.allclose(shower_core_pos[:,2], antenna_pos[:,2].mean(), atol=1e2, rtol=0), f"{shower_core_pos[:,2].mean()} vs {antenna_pos[:,2].mean()}"


def get_shower_properties(root_dir, start=0, stop=None):
    antenna_pos_file = sorted(glob(f'{root_dir}/run_*_L0_*.root'))[0]
    shower_meta_data_files = sorted(glob(f'{root_dir}/shower_*_L0_*.root'))
    efield_files = sorted(glob(f'{root_dir}/efield_*_L0_*.root'))

    with uproot.open(antenna_pos_file) as f:
        antenna_pos = f['trun']['du_xyz'].array().to_numpy()[0]

    overlap, first_indices, last_indices = _get_rootfiles_between_bounds(shower_meta_data_files, start, stop)
    zenith, azimuth, energy_primary, energy_em, xmax_grams, ptypes, event_numbers, efield_event_number = [np.array([]) for _ in range(8)]
    shower_core_pos, xmax_pos = [np.array([[]]).reshape(0, 3) for _ in range(2)]

    efield_du_id, efield_du_ns, efield_du_s, efield_du_pos, file_names = [], [], [], [], []

    def _append_array(arr, shower_meta_data, quant_to_append, start, stop):
        batch = shower_meta_data[quant_to_append].array(
            entry_start=start, entry_stop=stop).to_numpy()
        return np.concatenate((arr, batch))
    for index_overlap, first_idx, last_idx in zip(overlap, first_indices, last_indices):
        shower_meta_data_file = shower_meta_data_files[index_overlap]
        efield_file = efield_files[index_overlap]

        start_index = max(start, first_idx) - first_idx
        stop_index = min(stop, last_idx) - first_idx
        with uproot.open(shower_meta_data_file) as f:
            shower_meta_data = f['tshower']
            
            shower_core_pos = _append_array(shower_core_pos, shower_meta_data, 'shower_core_pos', start_index, stop_index)
            zenith = _append_array(zenith, shower_meta_data, 'zenith', start_index, stop_index)
            azimuth = _append_array(azimuth, shower_meta_data, 'azimuth', start_index, stop_index)
            energy_primary = _append_array(energy_primary, shower_meta_data, 'energy_primary', start_index, stop_index)
            energy_em = _append_array(energy_em, shower_meta_data, 'energy_em', start_index, stop_index)
            xmax_grams = _append_array(xmax_grams, shower_meta_data, 'xmax_grams', start_index, stop_index)
            xmax_pos = _append_array(xmax_pos, shower_meta_data, 'xmax_pos_shc', start_index, stop_index)
            ptypes = _append_array(ptypes, shower_meta_data, 'primary_type', start_index, stop_index)
            event_numbers = _append_array(event_numbers, shower_meta_data, 'event_number', start_index, stop_index)


        with uproot.open(efield_file) as f:
            batch_du_ns = f['tefield']['du_nanoseconds'].array(
                entry_start=start_index, entry_stop=stop_index)
            batch_du_s = f['tefield']['du_seconds'].array(
                entry_start=start_index, entry_stop=stop_index)
            batch_du_id = f['tefield']['du_id'].array(
                entry_start=start_index, entry_stop=stop_index)
            
            efield_du_ns += [du_ns.to_numpy() for du_ns in batch_du_ns]
            efield_du_s += [du_s.to_numpy()-du_s.to_numpy().min() for du_s in batch_du_s]
            efield_du_id += [du_id.to_numpy() for du_id in batch_du_id]
            efield_du_pos += [antenna_pos[du_id.to_numpy()] for du_id in batch_du_id]
        file_names += [root_dir.split('/')[-1]] * (stop_index - start_index)

    ptypes_int = np.ones_like(ptypes, dtype=int) 
    ptypes_int[ptypes == 'Fe^56'] = 56
    zenith = zenith * np.pi / 180
    azimuth = azimuth * np.pi / 180
    # xmax_pos = xmax_pos + shower_core_pos - np.array([[0, 0, altitude]])
    xmax_pos = xmax_pos + shower_core_pos
    _verify_xmax_pos(xmax_pos, shower_core_pos, zenith, azimuth, antenna_pos)
    meta_data = {
        "file_name": file_names,
        "event_numbers": event_numbers,
        'core_pos': shower_core_pos,
        'zenith': zenith,
        'azimuth': azimuth,
        'energy_primary': energy_primary,
        'energy_em': energy_em,
        'xmax_grams': xmax_grams,
        'xmax_pos': xmax_pos,
        'p_type': ptypes_int
    }
    efield_data = {
        'du_s': efield_du_s,
        'du_ns': efield_du_ns,
        'du_id': efield_du_id,
        'du_pos': efield_du_pos,
    }
    properties = {**meta_data, **efield_data}
    return antenna_pos, properties

def get_traces(root_dir, start, stop=None, voltage=False):
    efield_files = sorted(glob(f'{root_dir}/efield_*_L0_*.root'))
    adc_files = sorted(glob(f'{root_dir}/adc*_L1_*.root'))
    shower_meta_data_files = sorted(glob(f'{root_dir}/shower_*_L0_*.root'))
    overlap, first_indices, last_indices = _get_rootfiles_between_bounds(shower_meta_data_files, start, stop)
    all_traces, du_ids, file_names = [], [], []
    all_traces_adc = []
    theta, azimuth = np.array([]), np.array([])
    event_number = np.array([])
    for index_overlap, first_idx, last_idx in zip(overlap, first_indices, last_indices):
        efield_file = efield_files[index_overlap]
        adc_file = adc_files[index_overlap]
        shower_meta_data_file = shower_meta_data_files[index_overlap]

        start_index = max(start, first_idx) - first_idx
        stop_index = min(stop, last_idx) - first_idx

        with uproot.open(shower_meta_data_file) as f:
            shower_meta_data = f['tshower']
            batch_theta = shower_meta_data['zenith'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy() * np.pi / 180
            batch_azimuth = shower_meta_data['azimuth'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy() * np.pi / 180
            theta = np.concatenate((theta, batch_theta))
            azimuth = np.concatenate((azimuth, batch_azimuth))
        
        with uproot.open(efield_file) as f:
            efield_tree = f['tefield']
            batch_traces = efield_tree['trace'].array(
                entry_start=start_index, entry_stop=stop_index)
            batch_event_number = efield_tree['event_number'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy()
            batch_du_id = efield_tree['du_id'].array(
                entry_start=start_index, entry_stop=stop_index)
            
            event_number = np.concatenate([event_number, batch_event_number])
            du_ids += [du_id.to_numpy() for du_id in batch_du_id]
            all_traces += [trace.to_numpy() for trace in batch_traces]
        if voltage:
            with uproot.open(adc_file) as f:
                adc_tree = f['tadc']
                batch_traces = adc_tree['trace_ch'].array(
                    entry_start=start_index, entry_stop=stop_index)
                all_traces_adc += [trace.to_numpy() for trace in batch_traces]
                
        file_names += [root_dir.split('/')[-1]] * (stop_index - start_index)

    if voltage:
        return all_traces, all_traces_adc, du_ids, event_number, file_names, theta, azimuth
    return all_traces, du_ids, event_number, file_names, theta, azimuth


def get_traces_single_event(root_dir, event_number, all_event_numbers=None, voltage=False):
    if all_event_numbers is None:
        all_event_numbers = _get_all_event_numbers( root_dir )
    index = np.where(all_event_numbers == event_number)[0][0]
    if voltage:
        all_traces, all_trace_adc, du_ids, event_number_n, _, theta, azimuth= get_traces(root_dir, index, index+1, voltage=True)
        assert event_number_n[0] == event_number, "Event number mismatch."
        return all_traces[0], all_trace_adc[0], du_ids[0], theta, azimuth
    else:
        all_traces, du_ids, event_number_n, _, theta, azimuth= get_traces(root_dir, index, index+1)
        assert event_number_n[0] == event_number, "Event number mismatch."
        return all_traces[0], du_ids[0], theta, azimuth

def get_properties_single_event(root_dir, event_number, all_event_numbers=None):
    if all_event_numbers is None:
        all_event_numbers = _get_all_event_numbers( root_dir )
    index = np.where(all_event_numbers == event_number)[0][0]
    all_antenna_pos, properties = get_shower_properties(root_dir, index, index+1)
    properties = {key: val[0] for key, val in properties.items()}
    properties['du_times'] = sec_to_us(properties['du_s'] - properties['du_s'].min()) + ns_to_us(properties['du_ns'])
    properties['du_times'] = properties['du_times'] - properties['du_times'].min()
    antenna_pos = all_antenna_pos[properties['du_id']]
    return antenna_pos, properties
    
def get_traces_single_du(root_dir, event_number, du_id, all_event_numbers=None, voltage=False):
    event_traces, du_ids, theta, azimuth = get_traces_single_event(root_dir, event_number, all_event_numbers=all_event_numbers, voltage=voltage)
    index = np.where(du_ids == du_id)[0][0]
    trace_3d = event_traces[index]

    return trace_3d

def get_properties_single_du(root_dir, event_number, du_id):
    all_event_numbers = _get_all_event_numbers( root_dir )
    index = np.where(all_event_numbers == event_number)[0][0]
    _, properties = get_shower_properties(root_dir, index, index+1)

    du_index = np.where(properties['du_id'][0] == du_id)[0][0]
    properties['du_s'] = [properties['du_s'][0][du_index]]
    properties['du_ns'] = [properties['du_ns'][0][du_index]]
    properties['du_id'] = [properties['du_id'][0][du_index]]
    properties['du_pos'] = [properties['du_pos'][0][du_index]]
    properties = {key: val[0] for key, val in properties.items()}
    return properties

def open_event_root(directory_to_roots, start=0, stop=None, L1_or_L0='0'):
    """
    Open the ROOT file containing the event data.

    Parameters
    ----------
    directory_to_roots : str
        The path to the directory containing the ROOT files.
    start : int, optional
        The starting index for reading entries. Default is 0.
    stop : int, optional
        The stopping index for reading entries. Default is None.
    L1_or_L0 : str, optional
        Specify whether to use L1 or L0 data. Default is '0'.

    Returns
    -------
    tuple
        - antenna_pos : ndarray
            The positions of the antennas.
        - meta_data : dict
            Metadata about the shower, including core position, zenith, azimuth, etc.
        - efield_data : dict
            The electric field time traces and associated data.
    """
    antenna_pos, properties = get_shower_properties(directory_to_roots, start=start, stop=stop)

    
    antenna_pos_file = sorted(glob(f'{directory_to_roots}/run_*_L0_*.root'))[0]
    shower_meta_data_files = sorted(glob(f'{directory_to_roots}/shower_*_L0_*.root'))
    efield_files = sorted(glob(f'{directory_to_roots}/efield_*_L{L1_or_L0}_*.root'))

    with uproot.open(antenna_pos_file) as f:
        antenna_pos = f['trun']['du_xyz'].array().to_numpy()[0]
    
    n_events = []
    for met in shower_meta_data_files:
        with uproot.open(met) as f:
            n_events.append(f['tshower'].num_entries)
    n_events = np.array(n_events)
    if stop is None:
        stop = np.sum(n_events)

    last_indices = np.cumsum(n_events)
    first_indices = last_indices - n_events
    mask_above_start = (last_indices > start)
    mask_below_stop = (first_indices < stop)
    overlap = np.where(mask_above_start & mask_below_stop)[0]
    zenith, azimuth, energy_primary, energy_em, xmax_grams, ptypes, event_numbers, efield_event_number = [np.array([]) for _ in range(8)]
    shower_core_pos, xmax_pos = [np.array([[]]).reshape(0, 3) for _ in range(2)]

    efield_trace, efield_du_ns, efield_du_s, efield_du_id, file_names = [], [], [], [], []
    for index_overlap in overlap:
        shower_meta_data_file = shower_meta_data_files[index_overlap]
        efield_file = efield_files[index_overlap]
        start_index = max(start, first_indices[index_overlap]) - first_indices[index_overlap]
        stop_index = min(stop, last_indices[index_overlap]) - first_indices[index_overlap]
        with uproot.open(shower_meta_data_file) as f:
            shower_meta_data = f['tshower']
            shower_core_pos = np.concatenate((shower_core_pos, shower_meta_data['shower_core_pos'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy()))
            zenith = np.concatenate((zenith, shower_meta_data['zenith'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy() * np.pi / 180))
            azimuth = np.concatenate((azimuth, shower_meta_data['azimuth'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy() * np.pi / 180))
            energy_primary = np.concatenate((energy_primary, shower_meta_data['energy_primary'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy()))
            energy_em = np.concatenate((energy_em, shower_meta_data['energy_em'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy()))
            xmax_grams = np.concatenate((xmax_grams, shower_meta_data['xmax_grams'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy()))
            xmax_pos = np.concatenate((xmax_pos, shower_meta_data['xmax_pos_shc'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy()))
            ptypes = np.concatenate((ptypes, shower_meta_data['primary_type'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy()))
            event_numbers = np.concatenate((event_numbers, shower_meta_data['event_number'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy()))

        with uproot.open(efield_file) as f:
            efield_trace += [traces.to_numpy() for traces in f['tefield']['trace'].array(
                entry_start=start_index, entry_stop=stop_index)]
            efield_du_ns += [du_ns.to_numpy() for du_ns in f['tefield']['du_nanoseconds'].array(
                entry_start=start_index, entry_stop=stop_index)]
            efield_du_s += [du_s.to_numpy() for du_s in f['tefield']['du_seconds'].array(
                entry_start=start_index, entry_stop=stop_index)]
            efield_du_id += [du_id.to_numpy() for du_id in f['tefield']['du_id'].array(
                entry_start=start_index, entry_stop=stop_index)]
            efield_event_number = np.concatenate((efield_event_number, f['tefield']['event_number'].array(
                entry_start=start_index, entry_stop=stop_index).to_numpy()))
        file_names += [efield_file] * (stop_index - start_index)

    # xmax_pos = xmax_pos + shower_core_pos - np.array([[0, 0, altitude]])
    ptypes_int = np.ones_like(ptypes, dtype=int) 
    ptypes_int[ptypes == 'Fe^56'] = 56
    xmax_pos = xmax_pos + shower_core_pos
    k_p = shower_core_pos - xmax_pos
    k = k_p / np.linalg.norm(k_p, axis=1, keepdims=True)
    th_p = np.arccos(-k[:, 2])
    assert np.allclose(th_p * R2D, zenith * R2D, atol=1e-4, rtol=1e-4), f"{th_p*R2D} vs {zenith*R2D}"
    assert np.allclose(shower_core_pos[:,2], antenna_pos[:,2].mean(), atol=1e2, rtol=0), f"{shower_core_pos[:,2].mean()} vs {antenna_pos[:,2].mean()}"
    meta_data = {
        "event_numbers": event_numbers,
        'core_pos': shower_core_pos,
        'zenith': zenith,
        'azimuth': azimuth,
        'energy_primary': energy_primary,
        'energy_em': energy_em,
        'xmax_grams': xmax_grams,
        'xmax_pos': xmax_pos,
        'p_type': ptypes_int
    }
    efield_data = {
        'traces': efield_trace,
        'du_s': efield_du_s,
        'du_ns': efield_du_ns,
        'du_id': efield_du_id,
        'event_number': efield_event_number
    }
    return antenna_pos, meta_data, efield_data
