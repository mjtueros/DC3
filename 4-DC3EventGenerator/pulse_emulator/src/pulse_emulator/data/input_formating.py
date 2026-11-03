import numpy as np
import torch

from pulse_emulator.utils import C_M_PER_US, Bvec, altitude, sph2cart


def compute_kxB_kxkxB(k, Bvec=Bvec):
    """
    Compute kxB and kxkxB vectors.
    Parameters:
    k (<class 'ndarray'>): Direction vector. Shape (3,) or (N, 3). Could be torch.tensor.
    Bvec (<class 'ndarray'>): Magnetic field vector. Shape (3,). Could be torch.tensor.
    Returns:
    kxB (<class 'ndarray'>): Cross product of k and Bvec, normalized. Shape (N, 3) or (3,).
    kxkxB (<class 'ndarray'>): Cross product of k and kxB, normalized. Shape (N, 3) or (3,).
    """
    # Ensure k and Bvec are good shapes
    assert k.shape[-1] == 3, "k must have shape (3,) or (N, 3)."
    assert Bvec.shape == (3,), "Bvec must have shape (3,)."
    if type(k) is np.ndarray:
        met = np
    elif type(k) is torch.Tensor:
        met = torch
        Bvec = torch.tensor(Bvec, dtype=k.dtype, device=k.device)
    else:
        raise TypeError("k must be either a numpy array or a torch tensor.")

    kxB = met.cross(k, Bvec)
    kxB = kxB / met.linalg.norm(kxB, axis=-1, keepdims=True)
    kxkxB = met.cross(k, kxB)
    kxkxB = kxkxB / met.linalg.norm(kxkxB, axis=-1, keepdims=True)
    return kxB, kxkxB


def compute_l_omega_eta(Xa, Xs, k, kxB, kxkxB):
    """
    Compute omega and eta values based on input parameters.

    Parameters:
    Xa (<class 'ndarray'>): Coordinates of the antenna. Shape (N_ants, 3) or (3,). Could be torch.tensor.
    Xs (<class 'ndarray'>): Coordinates of the source. Shape (3,). Could be torch.tensor.
    k (<class 'ndarray'>): Direction vector. Shape (3,). Could be torch.tensor.
    Returns:
    omega (<class 'ndarray'>): Computed omega values. Shape (N_ants,).
    eta (<class 'ndarray'>): Computed eta values. Shape (N_ants,).
    l (<class 'ndarray'>): Computed l values. Shape (N_ants,).
    """
    # Ensure Xa and Xs are good shapes
    if Xs.shape == (1,3):
        Xs = Xs.squeeze(0)
    assert Xa.shape[-1] == 3, "Xa must have shape (N_ants, 3) or (3,)."
    assert Xs.shape == (3,) or Xs.shape == Xa.shape, "Xs must have shape (3,) or (N_ants, 3)."
    assert k.shape == (3,) or k.shape == Xa.shape, "k must have shape (3,) or (N_ants, 3)."
    assert type(Xa) == type(Xs), "Xa and Xs must be of the same type."
    assert type(Xa) == type(kxB), "Xa and kxB must be of the same type."
    assert type(Xa) == type(kxkxB), "Xa and kxkxB must be of the same type."
    assert type(k) == type(Xa), "k and Xa must be of the same type."
    if type(Xa) is np.ndarray:
        met = np
    elif type(Xa) is torch.Tensor:
        met = torch
    else:
        raise TypeError("Xa and Xs must be either numpy arrays or torch tensors.")

    D = Xa - Xs
    l = met.sqrt((D**2).sum(axis=-1))
    ka = D / l[:, None]
    omega = met.arccos(
        np.clip((k * ka).sum(axis=-1), -1.0, 1.0) 
        )
    kxB_component = (kxB * ka).sum(axis=-1)
    kxkxB_component = (kxkxB * ka).sum(axis=-1)
    eta = met.arctan2(kxkxB_component, kxB_component)
    
    return omega, eta, l

def swf_time(Xs, Xa, n_eff=1.0003):
    """
    Compute the shower wavefront time delay.
    Parameters:
    Xs (<class 'ndarray'> or <class 'torch.Tensor'>): Coordinates of the source. Shape (3,). Could be torch.tensor.
    Xa (<class 'ndarray'> or <class 'torch.Tensor'>): Coordinates of the antenna. Shape (N_ants, 3) or (3,). Could be torch.tensor.
    n_eff (float, ndarray-like): Effective refractive index. Default is 1.0003.
    Xcore (<class 'ndarray'> or <class 'torch.Tensor'>): Coordinates of the core. Shape (3,). Could be torch.tensor.
    Returns:
    time_delay (<class 'ndarray'> or <class 'torch.Tensor'>): Time delay of the shower wavefront at the antenna positions. Shape (N_ants,).
    """
    D = Xa - Xs
    if type(Xa) is np.ndarray:
        met = np
    elif type(Xa) is torch.Tensor:
        met = torch
    else:
        raise TypeError("Xa and Xs must be either numpy arrays or torch tensors.")


    # l = met.linalg.norm(D, axis=-1) # linalg.norm is too slow
    l = met.sqrt((D**2).sum(axis=-1))
    time_delay = n_eff * l / C_M_PER_US
    return time_delay

def event_swf_time(Xs, X_ants, n_effs=None):
    """
    Compute the shower wavefront time delay for each antenna in the event.
    
    Xs: shape (3,) - shower maximum position
    X_ants: shape (n_antennas, 3) - antenna positions
    """
    if n_effs is None:
        n_effs = ZHSEffectiveRefractionIndexvect(Xs, X_ants)
    swf_times = swf_time(Xs, X_ants, n_effs)
    return swf_times



def make_input_array(values, config_inputs):
    """
    Fast version of make_input_array that works with a dict of numpy arrays
    instead of a DataFrame.
    
    Eliminates pandas overhead entirely.
    """
    # Direction vector
    zenith = values['zenith']
    azimuth = values['azimuth']
    
    if np.all(zenith == zenith[0]):
        k = -sph2cart(zenith[0], azimuth[0])
    else:
        k = -sph2cart(zenith, azimuth)
    kxB, kxkxB = compute_kxB_kxkxB(k)
    
    # Compute geometric quantities only if needed
    need_geometry = {'omega', 'eta', 'l', 'cos_eta', 'sin_eta'}.intersection(config_inputs)
    if need_geometry:
        Xa = np.column_stack([values['du_pos_x'], values['du_pos_y'], values['du_pos_z']])
        Xs = np.column_stack([values['xmax_pos_x'], values['xmax_pos_y'], values['xmax_pos_z']])
        omega, eta, l = compute_l_omega_eta(Xa, Xs, k, kxB, kxkxB)
    
    X_list = []
    n_eff_arr = None
    
    for feature in config_inputs:
        if feature in values:
            X_list.append(values[feature])
        elif feature == 'sin_eta':
            X_list.append(np.sin(eta))
        elif feature == 'cos_eta':
            X_list.append(np.cos(eta))
        elif feature == 'sin_azimuth':
            X_list.append(np.sin(values['azimuth']))
        elif feature == 'cos_azimuth':
            X_list.append(np.cos(values['azimuth']))
        elif feature == 'omega':
            X_list.append(omega)
        elif feature == 'eta':
            X_list.append(eta)
        elif feature == 'l':
            X_list.append(l)
        elif feature == 'El':
            X_list.append(values['energy_primary'] * l)
        elif feature == 'n_eff':
            Xa = np.column_stack([values['du_pos_x'], values['du_pos_y'], values['du_pos_z']])
            Xs_row = np.array([values['xmax_pos_x'][0], values['xmax_pos_y'][0], values['xmax_pos_z'][0]])
            n_eff_arr = ZHSEffectiveRefractionIndexvect(Xs_row, Xa) - 1
            X_list.append(n_eff_arr)
        elif feature == 'omega_cr':
            n_effs = n_eff_arr + 1 if n_eff_arr is not None else X_list[config_inputs.index('n_eff')] + 1
            X_list.append(np.arccos(1.0 / n_effs))
        else:
            raise ValueError(f"Unsupported input feature: {feature}")
    input_arr = np.column_stack(X_list)
    return input_arr, (k, kxB, kxkxB)


def ZHSEffectiveRefractionIndex(X0,Xa):
    R_earth = 6371007.0
    ns = 325
    kr = -0.1218

    R02 = X0[0]**2 + X0[1]**2
    
    # Altitude of emission in km
    h0 = (np.sqrt( (X0[2]+R_earth)**2 + R02 ) - R_earth)/1e3
    # print('Altitude of emission in km = ',h0)
    # print(h0)
    
    # Refractivity at emission 
    rh0 = ns*np.exp(kr*h0)

    modr = np.sqrt(R02)
    # print(modr)

    if (modr > 1e3):

        # Vector between antenna and emission point
        U = Xa-X0
        # Divide into pieces shorter than 10km
        #nint = np.int(modr/2e4)+1
        nint = int(modr/2e4)+1
        K = U/nint

        # Current point coordinates and altitude
        Curr  = X0
        currh = h0
        s = 0.

        for i in np.arange(nint):
            Next = Curr + K # Next point
            nextR2 = Next[0]*Next[0] + Next[1]*Next[1]
            nexth  = (np.sqrt( (Next[2]+R_earth)**2 + nextR2 ) - R_earth)/1e3
            if (np.abs(nexth-currh) > 1e-10):
                s += (np.exp(kr*nexth)-np.exp(kr*currh))/(kr*(nexth-currh))
            else:
                s += np.exp(kr*currh)

            Curr = Next
            currh = nexth
            # print (currh)

        avn = ns*s/nint
        # print(avn)
        n_eff = 1. + 1e-6*avn # Effective (average) index

    else:
        print("No integration needed")
        # without numerical integration
        hd = Xa[2]/1e3 # Antenna altitude
        #if (np.abs(hd-h0) > 1e-10):
        avn = (ns/(kr*(hd-h0)))*(np.exp(kr*hd)-np.exp(kr*h0))
        #else:
        #    avn = ns*np.exp(kr*h0)

        n_eff = 1. + 1e-6*avn # Effective (average) index

    return (n_eff)


def ZHSEffectiveRefractionIndexvect(X0, Xa):
    R_earth = 6371007.0
    ns = 325
    kr = -0.1218

    R02 = X0[0]**2 + X0[1]**2

    # Altitude of emission in km
    h0 = (np.sqrt( (X0[2]+R_earth)**2 + R02 ) - R_earth)/1e3
    # print('Altitude of emission in km = ',h0)
    # print(h0)

    # Refractivity at emission 
    rh0 = ns*np.exp(kr*h0)

    modr = np.sqrt(R02)
    # print(modr)

    if (modr > 1e3):

        # Vector between antenna and emission point
        U = Xa-X0
        # Divide into pieces shorter than 10km
        #nint = np.int(modr/2e4)+1
        nint = int(modr/2e4)+1
        K = U/nint
        # Current point coordinates and altitude
        Curr  = np.repeat(X0[None, :], Xa.shape[0], axis=0)
        currh = np.array([h0]*Xa.shape[0])
        s = np.zeros(Xa.shape[0])

        for i in np.arange(nint):
            Next = Curr + K # Next point
            nextR2 = Next[:,0]*Next[:,0] + Next[:,1]*Next[:,1]
            nexth  = (np.sqrt( (Next[:,2]+R_earth)**2 + nextR2 ) - R_earth)/1e3

            mask = np.abs(nexth-currh) > 1e-10
            s[mask] += (np.exp(kr*nexth[mask])-np.exp(kr*currh[mask]))/(kr*(nexth[mask]-currh[mask]))
            s[~mask] += np.exp(kr*currh[~mask])
            
            Curr = Next
            currh = nexth

        avn = ns*s/nint
        # print(avn)
        n_eff = 1. + 1e-6*avn # Effective (average) index

    else:

        # without numerical integration
        hd = Xa[:,2]/1e3 # Antenna altitude
        #if (np.abs(hd-h0) > 1e-10):
        avn = (ns/(kr*(hd-h0)))*(np.exp(kr*hd)-np.exp(kr*h0))
        #else:
        #    avn = ns*np.exp(kr*h0)

        n_eff = 1. + 1e-6*avn # Effective (average) index

    return (n_eff)
    

def av_ref_index_flat_slow(xmax_pos_z, altitude=altitude):
    """
    Calculate the average refractive index from shower maximum to the antennas.
    Xmax_pos_z: Height of Xmax in the antenna coordinate system !! Not altitude !!
    """
    C = 0.1218 # in km^-1
    k = 3.25e-4 # no unit
    z_ground = altitude / 1e3 # in km
    average_refract = 1 + k/(C*xmax_pos_z) * np.exp(-C * z_ground) * (1 - np.exp(-C * xmax_pos_z)) # average refractive index at altitude xmax_pos_z
    return average_refract


def cherenkov_flat_slow(xmax_pos_z, altitude=altitude):
    """Calculate the Cherenkov angle from shower maximum to the antennas.
    """
    n_ref = av_ref_index_flat_slow(xmax_pos_z, altitude=altitude)
    cherenkov_angle = np.arccos(1/n_ref)
    return cherenkov_angle