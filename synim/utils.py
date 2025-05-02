import os
import re
import yaml
import datetime
import numpy as np

from scipy.special import eval_legendre
from scipy.ndimage import rotate

import specula
specula.init(device_idx=-1, precision=1)

from specula.calib_manager import CalibManager
from specula.lib.make_mask import make_mask
from specula.data_objects.ifunc import IFunc
from specula.data_objects.m2c import M2C
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.subap_data import SubapData

def rebin(array, new_shape, method='average'):
    """
    The rebin function resizes a vector or array to dimensions given by the parameters new_shape.
    In case of a 3D array the third dimension is not affected.

    Parameters:
    - array: numpy 2D or 3D array
    - new_shape: 2 elements tuple
    - method: 'sum' or 'average', used in the compression case

    Returns:
    - rebinned_array: numpy 2D or 3D array
    """
        
    if array.ndim == 1:
        array = array.reshape((-1, 1))  # Convert 1D array to 2D
        
    shape = array.shape
    m, n = shape[0:2]
    M, N = new_shape      

    if M > m or N > n:
        if M % m != 0 or N % n != 0:
            raise ValueError("New shape must be multiples of the input dimensions.")

        m_factor, n_factor = M // m, N // n

        # Replicate the array in both dimensions
        if len(shape) == 3:
            rebinned_array = np.tile(array, (m_factor, n_factor, 1))
        else:
            rebinned_array = np.tile(array, (m_factor, n_factor))
    else:    
        if method == 'sum':
            if len(shape) == 3:
                rebinned_array = array[:M*(m//M), :N*(n//N), :].reshape((M, m//M, N, n//N, shape[2])).sum(axis=(1, 3))
            else:
                rebinned_array = array[:M*(m//M), :N*(n//N)].reshape((M, m//M, N, n//N)).sum(axis=(1, 3))
        elif method == 'average':
            if len(shape) == 3:
                rebinned_array = array[:M*(m//M), :N*(n//N), :].reshape((M, m//M, N, n//N, shape[2])).mean(axis=(1, 3))
            else:
                rebinned_array = array[:M*(m//M), :N*(n//N)].reshape((M, m//M, N, n//N)).mean(axis=(1, 3))
        else:
            raise ValueError("Unsupported rebin method. Use 'sum' or 'average'.")  
        
    return rebinned_array

def polar_to_xy(r,theta):
    # conversion polar to rectangular coordinates
    # theta is in rad
    return np.array(( r * np.cos(theta),r * np.sin(theta) ))

def xy_to_polar(x, y):
    # conversion rectangular to polar coordinates
    # theta is in rad
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def apply_mask(array,mask,norm=False):
    # multiply a 2D or 3D by a 2D mask
    # if norm is True 1/mask is used.
    norm_array = np.copy(mask)
    if norm:
        idx2D = np.array(np.where(norm_array == 0))
        norm_array[idx2D[0,:],idx2D[1,:]] = 1
        norm_array = norm_array**(-1)
    if len(array.shape) == 3:
        norm_array_1d = norm_array.flatten()
        array_2d = array.reshape((-1,array.shape[2]))
        array_2d = array_2d*norm_array_1d[:, np.newaxis]
        new_array = array_2d.reshape(array.shape)
        if norm:
            new_array[idx2D[0,:],idx2D[1,:],:] = 0
    else:
        new_array = array * norm_array
        if norm:
            new_array[idx2D[0,:],idx2D[1,:]] = 0
    return new_array

def make_xy(sampling, ratio, is_polar=False, is_double=False, is_vector=False,
            use_zero=False, quarter=False, fft=False):
    """
    This function generates zero-centered domains in cartesian plane or axis, tipically for pupil sampling
    and FFT usage. Converted from Armando Riccardi IDL make_xy procedure of IdlTools/oaa_lib/utilities library.

    Parameters:
    - sampling: number of points on the side ot he output arrays
    - ratio: maximum value on the output arrays
    - ...

    Returns:
    - x: numpy 2D array
    - y: numpy 2D array
    """    
    
    if sampling <= 1:
        raise ValueError("make_xy -- sampling must be larger than 1")

    if quarter:
        if sampling % 2 == 0:
            size = sampling // 2
            x0 = 0.0 if use_zero else -0.5
        else:
            size = (sampling + 1) // 2
            x0 = 0.0
    else:
        size = sampling
        x0 = (sampling - 1) / 2.0 if is_double else (sampling - 1) / 2

        if sampling % 2 == 0 and use_zero:
            x0 += 0.5

    ss = float(sampling)

    x = (np.arange(size) - x0) / (ss / 2) * ratio

    if not quarter:
        if sampling % 2 == 0 and fft:
            x = np.roll(x, -sampling // 2)
        elif sampling % 2 != 0 and fft:
            x = np.roll(x, -(sampling - 1) // 2)

    if not is_vector or is_polar:
        y = rebin(x, (size, size), method='average')
        x = np.transpose(y)
        if is_polar:
            r, theta = xy_to_polar(x, y)
            return r, theta

    if is_vector:
        y = x

    return x, y

def make_mask(npoints, obsratio=None, diaratio=1.0, xc=0.0, yc=0.0, square=False, inverse=False, centeronpixel=False):
    """
    This function generates nn array representing a mask.
    Converted from Lorenzo Busoni IDL make_mask function of IdlTools/oaa_lib/ao_lib library.

    Parameters:
    - npoints: number of points on the side ot he output arrays
    - obsratio: relative size of obscuration
    - diaratio: relative size of diameter
    - ...

    Returns:
    - mask: numpy 2D array
    """
    
    x, y = np.meshgrid(np.linspace(-1, 1, npoints), np.linspace(-1, 1, npoints))

    if xc is None:
        xc = 0.0
    if yc is None:
        yc = 0.0
    if obsratio is None:
        obsratio = 0.0
    ir = obsratio

    if centeronpixel:
        idx = np.argmin(np.abs(xc - x[0, :]))
        idxneigh = np.argmin(np.abs(xc - x[0, idx - 1:idx + 2]))
        k = -0.5 if idxneigh == 0 else 0.5
        xc = x[0, idx] + k * (x[0, 1] - x[0, 0])

        idx = np.argmin(np.abs(yc - y[:, 0]))
        idxneigh = np.argmin(np.abs(yc - y[idx - 1:idx + 2, 0]))
        k = -0.5 if idxneigh == 0 else 0.5
        yc = y[idx, 0] + k * (y[1, 0] - y[0, 0])

    if square:
        mask = ((np.abs(x - xc) <= diaratio) & (np.abs(y - yc) <= diaratio) & 
                ((np.abs(x - xc) >= diaratio * ir) | (np.abs(y - yc) >= diaratio * ir))).astype(np.uint8)
    else:
        mask = (((x - xc)**2 + (y - yc)**2 < diaratio**2) & 
                ((x - xc)**2 + (y - yc)**2 >= (diaratio * ir)**2)).astype(np.uint8)

    if inverse:
        mask = 1 - mask

    return mask

def make_orto_modes(array):
    # return an othogonal 2D array
    
    size_array = np.shape(array)

    if len(size_array) != 2:
        raise ValueError('Error in input data, the input array must have two dimensions.')

    if size_array[1] > size_array[0]:
        Q, R = np.linalg.qr(array.T)
        Q = Q.T
    else:
        Q, R = np.linalg.qr(array)
    
    return Q

def zern_degree(j):
    """
    From Armando Riccardi IDL function.
    """
    # return the zernike degree
    n = np.floor(0.5 * (np.sqrt(8 * j - 7) - 3)) + 1
    cn = n * (n + 1) / 2 + 1

    if np.isscalar(n):
        if n % 2 == 0:
            m = np.floor((j - cn + 1) / 2) * 2
        else:
            m = np.floor((j - cn) / 2) * 2 + 1
    else:
        # new code for j vector
        idx_even = np.where(n % 2 == 0)[0]
        idx_odd = np.where((n + 1) % 2 == 0)[0]
        m = n * 0
        temp = j - cn
        m[idx_even] = np.floor((temp[idx_even] + 1) / 2) * 2
        m[idx_odd] = np.floor(temp[idx_odd] / 2) * 2 + 1

    return n, m.astype(int)

def zern_jradial(n, m, r):
    """
    Compute radial Zernike polynomial. 
    From Armando Riccardi IDL function.

    Parameters:
    - n: int, radial order
    - m: int, azimuthal order (0 <= m <= n, n-m even)
    - r: numpy array, radial coordinate

    Returns:
    - jpol: numpy array, radial Zernike polynomial
    """

    if n < 0 or m < 0 or m > n or (n - m) % 2 != 0:
        raise ValueError("Invalid values for n and m.")

    nmm2 = (n - m) // 2

    if m == 0:
        return eval_legendre(nmm2, 2.0 * r ** 2 - 1.0)
    else:
        prefactor = np.sqrt(2.0 / (1.0 + 2.0 * n))
        return prefactor * r ** m * eval_legendre(nmm2, 2.0 * r ** 2 - 1.0)

def zern_jpolar(j, rho, theta):
    """
    Compute polar Zernike polynomial.
    From Armando Riccardi IDL function.

    Parameters:
    - j: int, index of the polynomial, j >= 1
    - rho: numpy array, point to evaluate (polar coord.)
    - theta: numpy array, 

    Returns:
    - jpol: numpy 2D array, j-th Zernike polynomial in the point of polar coordinates r, theta
    """
    
    if j < 1:
        print("zern_jpolar -- must have j >= 1")
        return 0.0

    n, m = zern_degree(j)

    result = np.sqrt(n + 1 + rho ** 2) * zern_jradial(n, m, rho)

    if m == 0:
        return result
    elif j % 2 == 0:
        return np.sqrt(2) * result * np.cos(m * theta)
    else:
        return np.sqrt(2) * result * np.sin(m * theta)

def zern(j, x, y):
    """
    Compute the value of J-th Zernike polynomial in the points of coordinates x,y.
    From Armando Riccardi IDL function.

    Parameters:
    - j: int, index of the polynomial, j >= 1
    - rho: numpy 2D array, X coordinates
    - theta: numpy 2D array, X coordinates

    Returns:
    - jzern: numpy 2D array, j-th Zernike polynomial in the point of coordinates x ,y
    """

    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return zern_jpolar(j, rho, theta)

def zern2phi(dim, maxZernNumber, mask=None, no_round_mask=False, xsign=1, ysign=1, rot_angle=0, verbose=False):
    """
    Computes the Zernike phase cube and orthonormalizes it on a desired pupil (mask).
    From Guido Agapito IDL function.

    Parameters:
    - dim: int, number of point on the side of the output array
    - maxZernNumber: int, number of zernike modes excluding piston
    - mask: optional, numpy 2D array, mask
    - xsign: optional, sign of the x axis (for Zernike computation with zern function)
    - ysign: optional, sign of the y axis (for Zernike computation with zern function)
    - rot_angle: optional, float, rotation in deg
    - verbose, optional

    Returns:
    - z2phi: numpy 3D array, set of maxZernNumber zernike modes
    """
    
    if not no_round_mask:
        round_mask = np.array(make_mask(dim))
    else:
        round_mask = np.ones((dim, dim), dtype=float)

    if verbose:
        print('Computing Zernike cube...')

    xx, yy = make_xy(dim, 1, is_polar=False, is_double=False, is_vector=False,
            use_zero=False, quarter=False, fft=False)
    
    z2phi = np.zeros((dim, dim, maxZernNumber + 1), dtype=float)

    for i in range(maxZernNumber + 1):
        zern_shape = zern(i + 1, xsign * xx, ysign * yy)
        if rot_angle != 0:
            zern_shape = rotate(zern_shape, rot_angle, axes=(1, 0), reshape=False)
        z2phi[:, :, i] = zern_shape * round_mask

    if verbose:
        print('... Zernike cube computed')

    if mask is not None:
        if verbose:
            print('Orthogonalizing Zernike cube...')

        idx1D = np.where(mask.flatten())
        idx2D = np.where(mask)

        z2phi_temp = z2phi.reshape(-1, maxZernNumber + 1)
        z2phi_on_pupil = z2phi_temp[idx1D,:]
        z2phi_on_pupil = z2phi_on_pupil.reshape(-1,maxZernNumber + 1)
        
        z2phi_matrix_ortho = make_orto_modes(z2phi_on_pupil)

        #z2phi = np.zeros((dim, dim, maxZernNumber + 1), dtype=float)
        z2phi = np.full((dim,dim, maxZernNumber + 1),np.nan)
        
        for i in range(maxZernNumber + 1):
            temp = np.zeros((dim, dim), dtype=float)
            temp[idx2D[0],idx2D[1]] = z2phi_matrix_ortho[:, i] * 1/np.std(z2phi_matrix_ortho[:, i])
            temp[idx2D[0],idx2D[1]] = temp[idx2D[0],idx2D[1]] - np.mean(temp[idx2D[0],idx2D[1]])
            z2phi[:, :, i] = temp

        if verbose:
            print('Zernike cube orthogonalized!')

    z2phi = z2phi[:, :, 1:]

    return z2phi

def is_simple_config(config):
    """
    Detect if this is a simple SCAO config or a complex MCAO config.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        bool: True for simple SCAO config, False for complex MCAO config
    """
    # Check for multiple DMs
    dm_count = sum(1 for key in config if key.startswith('dm') and key != 'dm')
    
    # Check for multiple WFSs
    wfs_count = sum(1 for key in config if 
                   (key.startswith('sh_') or key.startswith('pyramid')) and key != 'pyramid')
    
    return dm_count == 0 and wfs_count == 0

def wfs_fov_from_config(wfs_params):
    """
    Extract field of view value from WFS parameters.
    
    Args:
        wfs_params (dict): Dictionary with WFS parameters
        
    Returns:
        float: Field of view in arcseconds
    """
    if wfs_params.get('sensor_fov') is not None:
        wfs_fov_arcsec = wfs_params['sensor_fov']
    elif wfs_params.get('fov') is not None:
        wfs_fov_arcsec = wfs_params['fov']
    elif wfs_params.get('subap_wanted_fov') is not None:
        wfs_fov_arcsec = wfs_params['subap_wanted_fov']
    else:
        wfs_fov_arcsec = 0
    return wfs_fov_arcsec

def determine_source_type(wfs_name):
    """
    Determine the source type from a WFS name.
    
    Args:
        wfs_name (str): Name of the WFS
        
    Returns:
        str: Source type ('lgs', 'ngs', or 'ref')
    """
    if 'lgs' in wfs_name:
        return 'lgs'
    elif 'ngs' in wfs_name:
        return 'ngs'
    elif 'ref' in wfs_name:
        return 'ref'
    return 'ngs'  # default

def extract_source_coordinates(config, wfs_key):
    """
    Extract polar coordinates for a given source.
    
    Args:
        config (dict): Configuration dictionary
        wfs_key (str): Key of the WFS in the config
        
    Returns:
        list: [distance, angle] polar coordinates
    """
    # First check if coordinates are in WFS parameters
    if wfs_key in config and 'gs_pol_coo' in config[wfs_key]:
        return config[wfs_key]['gs_pol_coo']
    
    # Try to find source corresponding to this WFS
    source_match = re.search(r'((?:lgs|ngs|ref)\d+)', wfs_key)
    if source_match:
        source_key = f'source_{source_match.group(1)}'
        if source_key in config:
            if 'polar_coordinates' in config[source_key]:
                return config[source_key]['polar_coordinates']
            elif 'polar_coordinate' in config[source_key]:
                return config[source_key]['polar_coordinate']
    
    # Try on_axis_source for simple configs
    if 'on_axis_source' in config:
        if 'polar_coordinates' in config['on_axis_source']:
            return config['on_axis_source']['polar_coordinates']
        elif 'polar_coordinate' in config['on_axis_source']:
            return config['on_axis_source']['polar_coordinate']
    
    # Default to on-axis
    return [0.0, 0.0]

def load_pupilstop(cm, pupilstop_params, pixel_pupil, pixel_pitch, verbose=False):
    """
    Load or create a pupilstop.
    
    Args:
        cm (CalibManager): SPECULA calibration manager
        pupilstop_params (dict): Parameters for the pupilstop
        pixel_pupil (int): Number of pixels across pupil
        pixel_pitch (float): Pixel pitch in meters
        verbose (bool): Whether to print details
        
    Returns:
        numpy.ndarray: Pupil mask array
    """
    if 'tag' in pupilstop_params or 'pupil_mask_tag' in pupilstop_params:
        if 'pupil_mask_tag' in pupilstop_params:
            pupilstop_tag = pupilstop_params['pupil_mask_tag']
            if verbose:
                print(f"     Loading pupilstop from file, tag: {pupilstop_tag}")
        else:
            pupilstop_tag = pupilstop_params['tag']
            if verbose:
                print(f"     Loading pupilstop from file, tag: {pupilstop_tag}")
        pupilstop_path = cm.filename('pupilstop', pupilstop_tag)
        pupilstop = Pupilstop.restore(pupilstop_path)
        return pupilstop.A
    else:
        # Create pupilstop from parameters
        mask_diam = pupilstop_params.get('mask_diam', 1.0)
        obs_diam = pupilstop_params.get('obs_diam', 0.0)
        
        # Create a new Pupilstop instance with the given parameters
        pupilstop = Pupilstop(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch,
            mask_diam=mask_diam,
            obs_diam=obs_diam,
            target_device_idx=-1,
            precision=0
        )
        return pupilstop.A

def load_influence_functions(cm, dm_params, pixel_pupil, verbose=False):
    """
    Load or generate DM influence functions.
    
    Args:
        cm (CalibManager): SPECULA calibration manager
        dm_params (dict): DM parameters
        pixel_pupil (int): Number of pixels across pupil
        verbose (bool): Whether to print details
        
    Returns:
        tuple: (dm_array, dm_mask) - 3D array of DM influence functions and mask
    """
    if 'ifunc_object' in dm_params or 'ifunc_tag' in dm_params:        
        if 'ifunc_tag' in dm_params:
            ifunc_tag = dm_params['ifunc_tag']
            if verbose:
                print(f"     Loading influence function from file, tag: {ifunc_tag}")
        else:
            ifunc_tag = dm_params['ifunc_object']
            if verbose:
                print(f"     Loading influence function from file, tag: {ifunc_tag}")
        ifunc_path = cm.filename('ifunc', ifunc_tag)
        ifunc = IFunc.restore(ifunc_path)
        
        m2c_tag = None
        if 'm2c_tag' in dm_params:
            m2c_tag = dm_params['m2c_tag']
            if verbose:
                print(f"     Loading M2C from file, tag: {m2c_tag}")
        if 'm2c_object' in dm_params:
            m2c_tag = dm_params['m2c_object']
            if verbose:
                print(f"     Loading M2C from file, tag: {m2c_tag}")
        if m2c_tag is not None:
            m2c_path = cm.filename('m2c', m2c_tag)
            m2c = M2C.restore(m2c_path)
            # multiply the influence function by the M2C
            ifunc.influence_function = ifunc.influence_function @ m2c.m2c

        # Convert influence function from 2D to 3D
        if ifunc.mask_inf_func is not None:           
            # Create empty 3D array (height, width, n_modes)
            dm_array = dm2d_to_3d(ifunc.influence_function, ifunc.mask_inf_func)
            if verbose:
                print(f"     DM array shape: {dm_array.shape}")
            # Create the DM mask
            dm_mask = ifunc.mask_inf_func.copy()
            if verbose:
                print(f"     DM mask shape: {dm_mask.shape}")
                print(f"     DM mask sum: {np.sum(dm_mask)}")
            return dm_array, dm_mask
        else:
            # If we don't have a mask, assume the influence function is already properly organized
            raise ValueError("IFunc without mask_inf_func is not supported. Mask is required to reconstruct the 3D array.")

    elif 'type_str' in dm_params:
        if verbose:
            print(f"     Loading influence function from type_str: {dm_params['type_str']}")
        # Create influence functions directly using Zernike modes
        from specula.lib.compute_zern_ifunc import compute_zern_ifunc
        nmodes = dm_params.get('nmodes', 100)
        obsratio = dm_params.get('obsratio', 0.0)
        npixels = dm_params.get('npixels', pixel_pupil)

        if 'mask_object' in dm_params:
            mask_tag = dm_params['mask_object']
            mask_path = cm.filename('pupilstop', mask_tag)
            print(f"     Loading mask from file, tag: {mask_tag}")
            pupilstop = Pupilstop.restore(mask_path)
            mask = pupilstop.A
        else:
            mask = None
            print("     No mask provided. Using default mask.")
        
        # Compute Zernike influence functions
        z_ifunc, z_mask = compute_zern_ifunc(npixels, nmodes, xp=np, dtype=float,
                                             obsratio=obsratio, diaratio=1.0,
                                             start_mode=0, mask=mask)

        # Create empty 3D array (height, width, n_modes)
        dm_array = dm2d_to_3d(z_ifunc, z_mask)
        if verbose:
            print(f"     DM array shape: {dm_array.shape}")
        dm_mask = z_mask
        if verbose:
            print(f"     DM mask shape: {dm_mask.shape}")
            print(f"     DM mask sum: {np.sum(dm_mask)}")
        return dm_array, dm_mask
    else:
        raise ValueError("No valid influence function configuration found. Need either 'ifunc_tag', 'ifunc_object', or 'type_str'.")

def find_subapdata(cm, wfs_params, wfs_key, params, verbose=False):
    """
    Find and load SubapData for valid subapertures.
    
    Args:
        cm (CalibManager): SPECULA calibration manager
        wfs_params (dict): WFS parameters
        wfs_key (str): WFS key in configuration
        params (dict): Full configuration parameters
        verbose (bool): Whether to print details
        
    Returns:
        numpy.ndarray: Array of valid subaperture indices or None
    """
    subap_path = None
    subap_tag = None

    # First check - Try to get subapdata from WFS params
    if 'subapdata_object' in wfs_params or 'subapdata_tag' in wfs_params:
        if 'subapdata_tag' in wfs_params:
            subap_tag = wfs_params['subapdata_tag']
            if verbose:
                print("     Loading subapdata from file, tag:", subap_tag)
            subap_path = cm.filename('subap_data', subap_tag)
        else:
            subap_tag = wfs_params['subapdata_object']
            if verbose:
                print("     Loading subapdata from file, tag:", subap_tag)
            subap_path = cm.filename('subapdata', subap_tag)

    # Second check - Try to find corresponding slopec section based on WFS name
    elif wfs_key is not None:
        # Determine potential slopec key based on WFS key (e.g., sh_lgs1 -> slopec_lgs1)
        slopec_key = None
        if wfs_key.startswith('sh_'):
            potential_slopec = 'slopec_' + wfs_key[3:]
            if potential_slopec in params:
                slopec_key = potential_slopec
        elif wfs_key.startswith('pyramid_'):
            potential_slopec = 'slopec_' + wfs_key[8:]
            if potential_slopec in params:
                slopec_key = potential_slopec
        # Handle numeric indices (e.g., sh1 -> slopec1)
        elif any(char.isdigit() for char in wfs_key):
            # Extract numeric portion
            numeric_part = ''.join(char for char in wfs_key if char.isdigit())
            if numeric_part:
                potential_slopec = f'slopec{numeric_part}'
                if potential_slopec in params:
                    slopec_key = potential_slopec
        
        # Check standard slopec key
        if slopec_key is None and 'slopec' in params:
            slopec_key = 'slopec'
        
        if slopec_key:
            slopec_params = params[slopec_key]
            if 'subapdata_tag' in slopec_params:
                if verbose:
                    print(f"     Loading subapdata from {slopec_key}, tag:", slopec_params['subapdata_tag'])
                subap_tag = slopec_params['subapdata_tag']
                subap_path = cm.filename('subap_data', subap_tag)
            elif 'subapdata_object' in slopec_params:
                if verbose:
                    print(f"     Loading subapdata from {slopec_key}, tag:", slopec_params['subapdata_object'])
                subap_tag = slopec_params['subapdata_object']
                subap_path = cm.filename('subapdata', subap_tag)

    # Third check - Try generic slopec section
    elif 'slopec' in params:
        slopec_params = params['slopec']
        if 'subapdata_tag' in slopec_params:
            if verbose:
                print("     Loading subapdata from slopec, tag:", slopec_params['subapdata_tag'])
            subap_tag = slopec_params['subapdata_tag']
            subap_path = cm.filename('subap_data', subap_tag)
        elif 'subapdata_object' in slopec_params:
            if verbose:
                print("     Loading subapdata from slopec, tag:", slopec_params['subapdata_object'])
            subap_tag = slopec_params['subapdata_object']
            subap_path = cm.filename('subapdata', subap_tag)
            
    if subap_path is None:
        if verbose:
            print("     No subapdata file found. Using default.")
        return None
    else:
        if verbose:
            print("     Subapdata file found:", subap_path)

    # Try to load the subapdata if a path was found
    if subap_path and os.path.exists(subap_path):
        if verbose:
            print("     Loading subapdata from file:", subap_path)
        subap_data = SubapData.restore(subap_path)
        return np.transpose(np.asarray(np.where(subap_data.single_mask())))
    
    return None

def insert_interaction_matrix_part(im_full, intmat_obj, mode_idx, slope_idx_start, slope_idx_end, verbose=False):
    """
    Insert part of an interaction matrix into a combined matrix.
    
    Args:
        im_full (numpy.ndarray): Target combined interaction matrix
        intmat_obj (Intmat): Source interaction matrix object
        mode_idx (list): Indices of modes to extract
        slope_idx_start (int): Start index for slopes in target matrix
        slope_idx_end (int): End index for slopes in target matrix
        verbose (bool): Whether to print details
        
    Returns:
        bool: True if insertion was successful
    """
    # Make sure we don't exceed matrix dimensions
    if not mode_idx or slope_idx_end > im_full.shape[1]:
        if verbose:
            print(f"  Warning: Invalid indices for matrix insertion")
        return False
        
    # Calculate how many modes we can actually use from this IM
    available_dm_modes = intmat_obj._intmat.shape[0]
    actual_mode_indices = [idx for idx in mode_idx if idx < available_dm_modes]
    
    if not actual_mode_indices:
        if verbose:
            print(f"  Warning: No valid mode indices. "
                  f"Available modes: {available_dm_modes}, requested: {mode_idx}")
        return False
        
    # Insert the valid modes into our combined matrix
    n_slopes = slope_idx_end - slope_idx_start
    im_full[mode_idx, slope_idx_start:slope_idx_end] = intmat_obj._intmat[actual_mode_indices, :n_slopes]
    
    if verbose:
        print(f"  Inserted {len(actual_mode_indices)} modes at indices {actual_mode_indices}, "
              f"slopes {slope_idx_start}:{slope_idx_end}")
    
    return True

def build_source_filename_part(source_config):
    """
    Build filename part for source parameters.
    
    Args:
        source_config (dict): Source configuration
        
    Returns:
        list: Filename parts for source
    """
    parts = []
    
    if 'polar_coordinates' in source_config:
        dist, angle = source_config['polar_coordinates']
        parts.append(f"pd{dist:.1f}a{angle:.0f}")
    elif 'polar_coordinate' in source_config:
        dist, angle = source_config['polar_coordinate']
        parts.append(f"pd{dist:.1f}a{angle:.0f}")
    elif 'pol_coords' in source_config:
        dist, angle = source_config['pol_coords']
        parts.append(f"pd{dist:.1f}a{angle:.0f}")
    
    if 'height' in source_config:
        parts.append(f"h{source_config['height']:.0f}")
    
    return parts

def build_pupil_filename_part(pupil_params):
    """
    Build filename part for pupil parameters.
    
    Args:
        pupil_params (dict): Pupil configuration
        
    Returns:
        list: Filename parts for pupil
    """
    parts = []
    
    if pupil_params:
        ps = pupil_params.get('pixel_pupil', 0)
        pp = pupil_params.get('pixel_pitch', 0)
        parts.append(f"ps{ps}p{pp:.4f}")
        
        if 'obsratio' in pupil_params and pupil_params['obsratio'] > 0:
            parts.append(f"o{pupil_params['obsratio']:.3f}")
    
    return parts

def build_wfs_filename_part(wfs_config, wfs_type):
    """
    Build filename part for WFS parameters.
    
    Args:
        wfs_config (dict): WFS configuration
        wfs_type (str): Type of WFS ('sh' or 'pyr')
        
    Returns:
        list: Filename parts for WFS
    """
    parts = []
    
    if wfs_type == 'sh':
        nsubaps = wfs_config.get('subap_on_diameter', 0)
        wl = wfs_config.get('wavelengthInNm', 0)
        fov = wfs_fov_from_config(wfs_config)
        npx = wfs_config.get('subap_npx', 0)
        parts.append(f"sh{nsubaps}x{nsubaps}_wl{wl}_fv{fov:.1f}_np{npx}")
    
    elif wfs_type == 'pyr':
        pup_diam = wfs_config.get('pup_diam', 0)
        wl = wfs_config.get('wavelengthInNm', 0)
        fov = wfs_fov_from_config(wfs_config)
        mod_amp = wfs_config.get('mod_amp', 0)
        parts.append(f"pyr{pup_diam:.1f}_wl{wl}_fv{fov:.1f}_ma{mod_amp:.1f}")
    
    return parts

def build_dm_filename_part(dm_config, config=None):
    """
    Build filename part for DM parameters.
    
    Args:
        dm_config (dict): DM configuration
        config (dict, optional): Full configuration (for DM reference in simple config)
        
    Returns:
        list: Filename parts for DM
    """
    parts = []
    
    height = dm_config.get('height', 0)
    parts.append(f"dmH{height}")
    
    # Check for custom influence functions - use config for simple configs
    target_config = config['dm'] if config and 'dm' in config else dm_config
    
    if 'ifunc_tag' in target_config:
        parts.append(f"ifunc_{target_config['ifunc_tag']}")
        if 'm2c_tag' in target_config:
            parts.append(f"m2c_{target_config['m2c_tag']}")
    elif 'ifunc_object' in target_config:
        parts.append(f"ifunc_{target_config['ifunc_object']}")
        if 'm2c_object' in target_config:
            parts.append(f"m2c_{target_config['m2c_object']}")
    elif 'type_str' in target_config:
        nmodes = dm_config.get('nmodes', 0)
        parts.append(f"nm{nmodes}_{target_config['type_str']}")
    else:
        # Default case
        nmodes = dm_config.get('nmodes', 0)
        parts.append(f"nm{nmodes}")
    
    return parts

def dm3d_to_2d(dm_array, mask):
    """
    Convert a 3D DM influence function to a 2D array using a mask.
    
    Args:
        dm_array (numpy.ndarray): 3D DM influence function array.
        mask (numpy.ndarray): 2D mask array.

    Returns:
        numpy.ndarray: 2D DM influence function array.
    """
    # Check if the mask is 2D
    if mask.ndim != 2:
        raise ValueError("The mask must be a 2D array.")
    # Check if the dm_array is 3D
    if dm_array.ndim != 3:
        raise ValueError("The dm_array must be a 3D array.")
    nmodes = dm_array.shape[2]
    idx = np.where(mask > 0)
    dm_array_2d = dm_array[idx[0], idx[1], :].transpose()
    for i in range(nmodes):
        dm_array_2d[i,:] /= np.sqrt(np.mean(dm_array_2d[i,:]**2))
        dm_array_2d[i,:] -= np.mean(dm_array_2d[i,:])

    return dm_array_2d

def dm2d_to_3d(dm_array, mask, normalize=True):
    """
    Convert a 2D DM influence function to a 3D array using a mask.
    
    Args:
        dm_array (numpy.ndarray): 2D DM influence function array.
        mask (numpy.ndarray): 2D mask array.

    Returns:
        numpy.ndarray: 3D DM influence function array.
    """
    # Check if the mask is 2D
    if mask.ndim != 2:
        raise ValueError("The mask must be a 2D array.")
    # Check if the dm_array is 2D
    if dm_array.ndim != 2:
        raise ValueError("The dm_array must be a 2D array.")
    npixels = mask.shape[0]
    nmodes = dm_array.shape[0]
    dm_array_3d = np.zeros((npixels, npixels, nmodes), dtype=float)
    for i in range(nmodes):
        idx = np.where(mask > 0)
        dm_i = dm_array[i]
        # normalize by the RMS
        if normalize:
            dm_i /= np.sqrt(np.mean(dm_i**2))
        dm_i_3d = np.zeros(mask.shape, dtype=float)
        dm_i_3d[idx] = dm_i
        dm_array_3d[:, :, i] = dm_i_3d

    return dm_array_3d

def parse_pro_file(pro_file_path):
    """
    Parse a .pro file and extract its structure into a Python dictionary.

    Args:
        pro_file_path (str): Path to the .pro file.

    Returns:
        dict: Parsed data as a dictionary.
    """
    data = {}
    current_section = None

    with open(pro_file_path, 'r') as file:
        for line in file:
            # Remove comments and whitespace
            line = line.split(';')[0].strip()
            if not line:
                continue

            # Recognize the start of a new section (e.g., {main, {DM, etc.)
            section_match = re.match(r'^\{(\w+),', line)
            if section_match:
                current_section = section_match.group(1).lower()
                data[current_section] = {}
                continue

            # Recognize the end of a section
            if line == '}':
                current_section = None
                continue

            # If we're in a section, process key-value pairs
            if current_section:
                key_value_match = re.match(r'(\w+)\s*[:=]\s*(.+)', line)
                if key_value_match:
                    key = key_value_match.group(1).strip()
                    value = key_value_match.group(2).strip()

                    # Remove any trailing commas
                    if value.endswith(','):
                        value = value[:-1].strip()

                    # Remove single quotes around strings
                    if value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]

                    # Interpret value types
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    elif re.match(r'^-?\d+(\.\d+)?$', value):  # Integer or float
                        value = float(value) if '.' in value else int(value)
                    elif re.match(r'^\[.*\]$', value):  # List
                        # Special handling for 'replicate'
                        def replicate_replacer(match):
                            val = match.group(1)
                            num = int(match.group(2))
                            return str([float(val)] * num)
                        # Replace all occurrences of replicate(x, n)
                        value = re.sub(r'replicate\(([^,]+),\s*(\d+)\)', replicate_replacer, value)
                        value = eval(value)
                    elif re.match(r'^[\d\.]+/[^\s]+$', value):  # Mathematical expression (e.g., 8.118/160)
                        try:
                            value = eval(value)
                        except Exception:
                            pass
                    elif value.lower() == '!values.f_infinity':  # Special case for infinity
                        value = float('inf')

                    data[current_section][key] = value

    return data

def parse_params_file(file_path):
    """
    Parse a parameters file (YAML or .pro) and return its contents as a dictionary.

    Args:
        file_path (str): Path to the parameters file.

    Returns:
        dict: Parsed data as a dictionary.
    """
    if file_path.endswith('.yml') or file_path.endswith('.yaml'):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    elif file_path.endswith('.pro'):
        return parse_pro_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def prepare_interaction_matrix_params(params, wfs_type=None, wfs_index=None, dm_index=None):
    """
    Prepares parameters for synim.interaction_matrix from a SPECULA YAML configuration file.
    
    Args:
        params (dict): Dictionary with configuration parameters.
        wfs_type (str, optional): Type of WFS ('sh', 'pyr') or source type ('lgs', 'ngs', 'ref')
        wfs_index (int, optional): Index of the WFS to use (1-based)
        dm_index (int, optional): Index of the DM to use (1-based)
        
    Returns:
        dict: Parameters ready to be passed to synim.interaction_matrix
    """
    
    # Prepare the CalibManager
    main_params = params['main']
    cm = CalibManager(main_params['root_dir'])
    
    # Extract general parameters
    pixel_pupil = main_params['pixel_pupil']
    pixel_pitch = main_params['pixel_pitch']
    pup_diam_m = pixel_pupil * pixel_pitch
    
    # Determine if this is a simple or complex configuration
    simple_config = is_simple_config(params)
    
    # Extract all WFS and DM configurations
    wfs_list = extract_wfs_list(params)
    dm_list = extract_dm_list(params)
    
    # Load pupilstop and create pupil mask
    pup_mask = None
    if 'pupilstop' in params:
        pupilstop_params = params['pupilstop']
        pup_mask = load_pupilstop(cm, pupilstop_params, pixel_pupil, pixel_pitch, verbose=True)
    
    # If no pupilstop defined, create a default circular pupil
    if pup_mask is None:
        pupilstop = Pupilstop(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch,
            mask_diam=1.0,
            obs_diam=0.0,
            target_device_idx=-1,
            precision=0
        )
        pup_mask = pupilstop.A
    
    # Find the appropriate DM based on dm_index
    selected_dm = None
    
    if dm_index is not None:
        # Try to find DM with specified index
        for dm in dm_list:
            if dm['index'] == str(dm_index):
                selected_dm = dm
                print(f"DM -- Using specified DM: {dm['name']}")
                break
    
    # If no DM found with specified index or no index specified, use first DM
    if selected_dm is None and dm_list:
        selected_dm = dm_list[0]
        print(f"DM -- Using first available DM: {selected_dm['name']}")
    
    if selected_dm is None:
        raise ValueError("No DM configuration found in the YAML file.")
    
    dm_key = selected_dm['name']
    dm_params = selected_dm['config']
    
    # Extract DM parameters
    dm_height = dm_params.get('height', 0)
    dm_rotation = dm_params.get('rotation', 0.0)
    
    # Load influence functions
    dm_array, dm_mask = load_influence_functions(cm, dm_params, pixel_pupil, verbose=True)

    if 'nmodes' in dm_params:
        nmodes = dm_params['nmodes']
        if dm_array.shape[2] > nmodes:
            print(f"     Trimming DM array to first {nmodes} modes")
            dm_array = dm_array[:, :, :nmodes]

    # WFS selection logic
    selected_wfs = None
    source_type = None
    
    print("WFS -- Looking for WFS parameters...")
    
    # Simple SCAO configuration case
    if simple_config:
        print("     Simple SCAO configuration detected")
        if len(wfs_list) > 0:
            selected_wfs = wfs_list[0]
            source_type = 'ngs'  # Default for simple configs
            print(f"     Using WFS: {selected_wfs['name']} of type {selected_wfs['type']}")
        else:
            raise ValueError("No WFS configuration found in the YAML file.")
    else:
        # Complex MCAO configuration
        print("     Complex MCAO configuration detected")
        
        # Case 1: wfs_type specifies the sensor type ('sh', 'pyr')
        if wfs_type in ['sh', 'pyr']:
            print(f"     Looking for WFS of type: {wfs_type}")
            matching_wfs = [wfs for wfs in wfs_list if wfs['type'] == wfs_type]
            
            if wfs_index is not None:
                # Try to find specific index
                for wfs in matching_wfs:
                    if wfs['index'] == str(wfs_index):
                        selected_wfs = wfs
                        print(f"     Found WFS with specified index: {wfs['name']}")
                        break
            
            # If no specific index found, use the first one
            if selected_wfs is None and matching_wfs:
                selected_wfs = matching_wfs[0]
                print(f"     Using first WFS of type {wfs_type}: {selected_wfs['name']}")
        
        # Case 2: wfs_type specifies the source type ('lgs', 'ngs', 'ref')
        elif wfs_type in ['lgs', 'ngs', 'ref']:
            source_type = wfs_type
            print(f"     Looking for WFS associated with {wfs_type} source")
            
            # Pattern for WFS names corresponding to the source type
            pattern = f"sh_{source_type}"
            matching_wfs = [wfs for wfs in wfs_list if pattern in wfs['name']]
            
            if wfs_index is not None:
                # Try to find specific index within the source type
                target_name = f"{pattern}{wfs_index}"
                for wfs in matching_wfs:
                    if wfs['name'] == target_name:
                        selected_wfs = wfs
                        print(f"     Found WFS with specified index: {wfs['name']}")
                        break
            
            # If no specific index found, use the first one
            if selected_wfs is None and matching_wfs:
                selected_wfs = matching_wfs[0]
                print(f"     Using first WFS for {wfs_type}: {selected_wfs['name']}")
        
        # Case 3: Only wfs_index is specified (no wfs_type)
        elif wfs_index is not None:
            print(f"     Looking for WFS with index: {wfs_index}")
            for wfs in wfs_list:
                if wfs['index'] == str(wfs_index):
                    selected_wfs = wfs
                    print(f"     Found WFS with specified index: {wfs['name']}")
                    break
        
        # Case 4: No specific criteria, use the first available WFS
        if selected_wfs is None and wfs_list:
            selected_wfs = wfs_list[0]
            print(f"     Using first available WFS: {selected_wfs['name']}")
    
    # If no WFS found, raise error
    if selected_wfs is None:
        raise ValueError("No matching WFS configuration found in the YAML file.")
    
    wfs_key = selected_wfs['name']
    wfs_params = selected_wfs['config']
    wfs_type_detected = selected_wfs['type']
    
    # Determine source type from WFS name if not already set
    if source_type is None:
        source_type = determine_source_type(wfs_key)
    
    print(f"     Source type determined: {source_type}")
    
    # Extract WFS parameters
    wfs_rotation = wfs_params.get('rotation', 0.0)
    wfs_translation = wfs_params.get('translation', [0.0, 0.0])
    wfs_magnification = wfs_params.get('magnification', 1.0)
    wfs_fov_arcsec = wfs_fov_from_config(wfs_params)
    
    if wfs_type_detected == 'sh':
        # Shack-Hartmann specific parameters
        wfs_nsubaps = wfs_params.get('subap_on_diameter', 0)
    else:
        # Pyramid specific parameters
        wfs_nsubaps = wfs_params.get('pup_diam', 0)
    
    # Load SubapData for valid subapertures
    idx_valid_sa = find_subapdata(cm, wfs_params, wfs_key, params, verbose=True)
    
    # Guide star parameters
    if source_type == 'lgs':
        # LGS is at finite height
        # Try to get height from source or use typical LGS height
        gs_height = None
        
        # Check if there's a specific source for this WFS and try to get height
        source_match = re.search(r'(lgs\d+)', wfs_key)
        if source_match:
            source_key = f'source_{source_match.group(1)}'
            if source_key in params:
                gs_height = params[source_key].get('height', None)
        
        # If still no height, use default
        if gs_height is None:
            gs_height = 90000.0  # Default LGS height in meters
    else:
        # NGS and REF are at infinite distance
        gs_height = float('inf')
    
    # Get source polar coordinates
    gs_pol_coo = extract_source_coordinates(params, wfs_key)

    # Return the prepared parameters
    return {
        'pup_diam_m': pup_diam_m,
        'pup_mask': pup_mask,
        'dm_array': dm_array,
        'dm_mask': dm_mask,
        'dm_height': dm_height,
        'dm_rotation': dm_rotation,
        'wfs_key': wfs_key,
        'wfs_type': wfs_type_detected,
        'wfs_nsubaps': wfs_nsubaps,
        'wfs_rotation': wfs_rotation,
        'wfs_translation': wfs_translation,
        'wfs_magnification': wfs_magnification,
        'wfs_fov_arcsec': wfs_fov_arcsec,
        'gs_pol_coo': gs_pol_coo,
        'gs_height': gs_height,
        'idx_valid_sa': idx_valid_sa,
        'dm_key': dm_key,
        'source_type': source_type
    }

def extract_wfs_list(config):
    """Extract all WFS configurations from config"""
    wfs_list = []
    
    # Find Pyramid WFSs
    for key in config:
        if key == 'pyramid' or (isinstance(key, str) and (key.startswith('pyramid') or key == 'pyr')):
            wfs_list.append({
                'name': key,
                'type': 'pyr',
                'index': re.findall(r'\d+', key)[0] if re.findall(r'\d+', key) else '1',
                'config': config[key]
            })
    
    # Find Shack-Hartmann WFSs
    for key in config:
        if key == 'sh' or key.startswith('sh_') or key.startswith('sh'):
            wfs_list.append({
                'name': key,
                'type': 'sh', 
                'index': re.findall(r'\d+', key)[0] if re.findall(r'\d+', key) else '1',
                'config': config[key]
            })
    
    return wfs_list

def extract_dm_list(config):
    """Extract all DM configurations from config"""
    dm_list = []
    
    for key in config:
        if key == 'dm' or (isinstance(key, str) and key.startswith('dm')):
            dm_list.append({
                'name': key,
                'index': re.findall(r'\d+', key)[0] if re.findall(r'\d+', key) else '1',
                'config': config[key]
            })
    
    # If no DMs found, create a default one
    if len(dm_list) == 0 and 'dm' in config:
        dm_list.append({
            'name': 'dm',
            'index': '1',
            'config': config['dm']
        })
    
    return dm_list

def extract_source_info(config, wfs_name):
    """Extract source information related to a specific WFS"""
    source_info = {}
    
    # Check both formats: direct reference or connection through prop
    if 'inputs' in config[wfs_name] and 'in_ef' in config[wfs_name]['inputs']:
        source_ref = config[wfs_name]['inputs']['in_ef']
        
        # Format might be 'prop.out_source_lgs1_ef' or direct 'out_source_lgs1_ef'
        match = re.search(r'out_(\w+)_ef', source_ref)
        if match:
            source_name = match.group(1)
            
            # Find the source in the config
            if source_name in config:
                source = config[source_name]
                source_info['type'] = 'lgs' if 'lgs' in source_name else 'ngs'
                source_info['name'] = source_name
                if 'polar_coordinates' in source:
                    source_info['pol_coords'] = source['polar_coordinates']
                if 'height' in source:
                    source_info['height'] = source['height']
                if 'wavelengthInNm' in source:
                    source_info['wavelength'] = source['wavelengthInNm']
    
    # Direct reference within source objects (for simple YAML)
    if not source_info and 'on_axis_source' in config:
        source_info['type'] = 'ngs'
        source_info['name'] = 'on_axis_source'
        if config['on_axis_source'].get('polar_coordinates'):
            source_info['pol_coords'] = config['on_axis_source']['polar_coordinates']
        elif config['on_axis_source'].get('polar_coordinate'):
            source_info['pol_coords'] = config['on_axis_source']['polar_coordinate']
        else:
            source_info['pol_coords'] = [0, 0]
        source_info['wavelength'] = config['on_axis_source'].get('wavelengthInNm', 750)
    
    return source_info

def is_simple_config(config):
    """Detect if this is a simple SCAO config or a complex MCAO config"""
    # Check for multiple DMs
    dm_count = sum(1 for key in config if key.startswith('dm') and key != 'dm')
    
    # Check for multiple WFSs
    wfs_count = sum(1 for key in config if key.startswith('sh_') or key.startswith('pyramid') and key != 'pyramid')
    
    return dm_count == 0 and wfs_count == 0

def generate_im_filename(config_file, wfs_type=None, wfs_index=None, dm_index=None, timestamp=False, verbose=False):
    """
    Generate a specific interaction matrix filename based on WFS and DM indices.
    
    Args:
        config_file (str): Path to YAML or PRO configuration file
        wfs_type (str, optional): Type of WFS ('sh', 'pyr') or source type ('lgs', 'ngs', 'ref')
        wfs_index (int, optional): Index of the WFS to use (1-based)
        dm_index (int, optional): Index of the DM to use (1-based)
        timestamp (bool, optional): Whether to include timestamp in the filename
        verbose (bool, optional): Whether to print verbose output
        
    Returns:
        str: Filename for the interaction matrix with the specified parameters
    """
    # Load configuration
    if isinstance(config_file, str):
        config = parse_params_file(config_file)
    else:
        # Assume it's already a parsed config dictionary
        config = config_file
    
    # Convert indices to strings for comparison
    wfs_index_str = str(wfs_index) if wfs_index is not None else None
    dm_index_str = str(dm_index) if dm_index is not None else None
    
    # Determine if this is a simple or complex configuration
    simple_config = is_simple_config(config)
    
    # For simple configuration, there's typically only one WFS and DM
    if simple_config:
        if verbose:
            print("Simple SCAO configuration detected")
        # Just generate the single filename that would be created
        filenames = generate_im_filenames(config)
        
        # Simple configs typically use NGS
        if 'ngs' in filenames and filenames['ngs']:
            return filenames['ngs'][0]
        # Fall back to any available filename
        for source_type in ['lgs', 'ref']:
            if source_type in filenames and filenames[source_type]:
                return filenames[source_type][0]
        
        # No valid filename found
        return None
    
    # For complex configuration, we need to find the matching WFS and DM
    if verbose:
        print("Complex MCAO configuration detected")
    
    # Get lists of all WFSs and DMs
    wfs_list = extract_wfs_list(config)
    dm_list = extract_dm_list(config)
    
    # Filter WFS list based on wfs_type and wfs_index
    filtered_wfs = wfs_list
    if wfs_type:
        # Check if wfs_type is a sensor type ('sh', 'pyr')
        if wfs_type in ['sh', 'pyr']:
            filtered_wfs = [wfs for wfs in filtered_wfs if wfs['type'] == wfs_type]
        # Check if wfs_type is a source type ('lgs', 'ngs', 'ref')
        elif wfs_type in ['lgs', 'ngs', 'ref']:
            filtered_wfs = [wfs for wfs in filtered_wfs if wfs_type in wfs['name']]
    
    if wfs_index_str:
        filtered_wfs = [wfs for wfs in filtered_wfs if wfs['index'] == wfs_index_str]
    
    # Filter DM list based on dm_index
    filtered_dm = dm_list
    if dm_index_str:
        filtered_dm = [dm for dm in filtered_dm if dm['index'] == dm_index_str]
    
    # If we couldn't find matching WFS or DM, return None
    if not filtered_wfs or not filtered_dm:
        if verbose:
            print("No matching WFS or DM found with the specified parameters")
        return None
    
    # Select the first WFS and DM from the filtered lists
    selected_wfs = filtered_wfs[0]
    selected_dm = filtered_dm[0]
    
    if verbose:
        print(f"Selected WFS: {selected_wfs['name']} (type: {selected_wfs['type']}, index: {selected_wfs['index']})")
        print(f"Selected DM: {selected_dm['name']} (index: {selected_dm['index']})")
    
    # Determine the source type from the WFS name
    source_type = determine_source_type(selected_wfs['name'])
    
    # Extract source information
    source_coords = extract_source_coordinates(config, selected_wfs['name'])
    
    # Extract DM height
    dm_height = selected_dm['config'].get('height', 0)
    
    # Generate filename parts
    base_name = "IM_syn"
    parts = [base_name]
    
    # Source info
    if source_coords is not None:
        dist, angle = source_coords
        parts.append(f"pd{dist:.1f}a{angle:.0f}")
    
    # WFS info
    wfs_config = selected_wfs['config']
    parts.extend(build_wfs_filename_part(wfs_config, selected_wfs['type']))
    
    # DM info
    parts.append(f"dmH{dm_height}")
    
    # Add DM-specific parts
    parts.extend(build_dm_filename_part(selected_dm['config']))
    
    # Add timestamp if requested
    if timestamp:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(ts)
    
    # Join all parts with underscores and add extension
    filename = "_".join(parts) + ".fits"
    return filename

def generate_im_filenames(config_file, timestamp=False):
    """
    Generate interaction matrix filenames for all WFS-DM combinations, grouped by star type.
    
    Args:
        config_file (str or dict): Path to YAML/PRO file or config dictionary
        timestamp (bool, optional): Whether to include timestamp in filenames
        
    Returns:
        dict: Dictionary with star types as keys and list of filenames as values
    """
    # Load YAML or PRO configuration
    if isinstance(config_file, str):
        config = parse_params_file(config_file)
    else:
        config = config_file
    
    # Detect if simple or complex configuration
    simple_config = is_simple_config(config)
    
    # Basic system info
    base_name = 'IM_syn'
    
    # Pupil parameters
    pupil_params = {}
    if 'main' in config:
        pupil_params['pixel_pupil'] = config['main'].get('pixel_pupil', 0)
        pupil_params['pixel_pitch'] = config['main'].get('pixel_pitch', 0)
    
    if 'pupilstop' in config:
        pupstop = config['pupilstop']
        if isinstance(pupstop, dict):
            pupil_params['obsratio'] = pupstop.get('obsratio', 0.0)
            pupil_params['tag'] = pupstop.get('tag', '')
    
    # Output dictionary: key=star type, value=list of filenames
    filenames_by_type = {
        'lgs': [],
        'ngs': [],
        'ref': []
    }
    
    # Extract all DM configurations
    dm_list = extract_dm_list(config)

    # For simple configurations with on-axis source
    if simple_config:
        # Simple SCAO configuration
        wfs_type = None
        wfs_params = {}
        
        if 'pyramid' in config:
            wfs_type = 'pyr'
            wfs_params = config['pyramid']
        elif 'sh' in config:
            wfs_type = 'sh'
            wfs_params = config['sh']
            
        # Source info
        source_config = config.get('on_axis_source', {})
        
        # Build filename parts
        parts = [base_name]
        
        # Add source parts
        parts.extend(build_source_filename_part(source_config))
        
        # Add pupil parts
        parts.extend(build_pupil_filename_part(pupil_params))
        
        # Add WFS parts
        if wfs_type:
            parts.extend(build_wfs_filename_part(wfs_params, wfs_type))
        
        # Add DM parts - use config for simple configs
        for dm in dm_list:
            dm_parts = build_dm_filename_part(dm['config'], config)
            parts.extend(dm_parts)
            
            # Add timestamp if requested
            if timestamp:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                parts.append(ts)

            # Join all parts with underscores and add extension
            filename = "_".join(parts) + ".fits"
            filenames_by_type['ngs'].append(filename)  # Default to NGS for simple config
            break  # Only one DM in simple config
    else:
        # Complex MCAO configuration
        wfs_list = extract_wfs_list(config)
        
        # Generate filenames for all WFS-DM combinations
        for wfs in wfs_list:
            wfs_type = wfs['type']
            wfs_params = wfs['config']
            
            # Determine source type from WFS name
            source_type = determine_source_type(wfs['name'])
            
            # Get source information
            source_config = {}
            source_match = re.search(r'((?:lgs|ngs|ref)\d+)', wfs['name'])
            if source_match:
                source_key = f'source_{source_match.group(1)}'
                if source_key in config:
                    source_config = config[source_key]
            
            # For each DM, generate a filename
            for dm in dm_list:
                dm_params = dm['config']
                
                # Build filename parts
                parts = [base_name]
                
                # Add source parts
                parts.extend(build_source_filename_part(source_config))
                
                # Add pupil parts
                parts.extend(build_pupil_filename_part(pupil_params))
                
                # Add WFS parts
                parts.extend(build_wfs_filename_part(wfs_params, wfs_type))
                
                # Add DM parts
                parts.extend(build_dm_filename_part(dm_params))
                
                # Add timestamp if requested
                if timestamp:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    parts.append(ts)
                
                # Join all parts with underscores and add extension
                filename = "_".join(parts) + ".fits"
                filenames_by_type[source_type].append(filename)
    
    return filenames_by_type

def compute_mmse_reconstructor(interaction_matrix, C_atm, noise_variance=None, C_noise=None, 
                              cinverse=False, verbose=False):
    """
    Compute the Minimum Mean Square Error (MMSE) reconstructor.
    
    Args:
        interaction_matrix (numpy.ndarray): Interaction matrix A relating modes to slopes
        C_atm (numpy.ndarray): Covariance matrix of atmospheric modes (Cx)
        noise_variance (list, optional): List of noise variances per WFS. 
                                        Used to build C_noise if C_noise is None.
        C_noise (numpy.ndarray, optional): Covariance matrix of measurement noise (Cz).
                                         If None, it's built from noise_variance.
        cinverse (bool, optional): If True, C_atm and C_noise are already inverted.
        verbose (bool, optional): Whether to print detailed information during computation.
        
    Returns:
        numpy.ndarray: MMSE reconstructor matrix
    """
    if verbose:
        print("Starting MMSE reconstructor computation")
    
    # Setup matrices
    A = interaction_matrix
    
    # Handle noise covariance matrix
    if C_noise is None and noise_variance is not None:
        n_slopes_total = A.shape[1]
        n_wfs = len(noise_variance)
        n_slopes_per_wfs = n_slopes_total // n_wfs
        
        if verbose:
            print(f"Building noise covariance matrix for {n_wfs} WFSs with {n_slopes_per_wfs} slopes each")
        
        C_noise = np.zeros((n_slopes_total, n_slopes_total))
        for i in range(n_wfs):
            # Set the diagonal elements for this WFS
            start_idx = i * n_slopes_per_wfs
            end_idx = (i + 1) * n_slopes_per_wfs
            C_noise[start_idx:end_idx, start_idx:end_idx] = noise_variance[i] * np.eye(n_slopes_per_wfs)
    
    # Check dimensions
    if A.shape[1] != C_atm.shape[0]:
        raise ValueError(f"A ({A.shape}) and C_atm ({C_atm.shape}) must have compatible dimensions")
    
    if C_noise is not None and A.shape[0] != C_noise.shape[0]:
        raise ValueError(f"A ({A.shape}) and C_noise ({C_noise.shape}) must have compatible dimensions")
    
    # Compute inverses if needed
    if not cinverse:
        # Check if matrices are diagonal
        if C_noise is not None:
            is_diag_noise = np.all(np.abs(np.diag(np.diag(C_noise)) - C_noise) < 1e-10)
            
            if is_diag_noise:
                if verbose:
                    print("C_noise is diagonal, using optimized inversion")
                C_noise_inv = np.diag(1.0 / np.diag(C_noise))
            else:
                if verbose:
                    print("Inverting C_noise matrix")
                try:
                    C_noise_inv = np.linalg.inv(C_noise)
                except np.linalg.LinAlgError:
                    if verbose:
                        print("Warning: C_noise inversion failed, using pseudo-inverse")
                    C_noise_inv = np.linalg.pinv(C_noise)
        else:
            # Default: identity matrix (no noise)
            if verbose:
                print("No C_noise provided, using identity matrix")
            C_noise_inv = np.eye(A.shape[1])
            
        is_diag_atm = np.all(np.abs(np.diag(np.diag(C_atm)) - C_atm) < 1e-10)
        
        if is_diag_atm:
            if verbose:
                print("C_atm is diagonal, using optimized inversion")
            C_atm_inv = np.diag(1.0 / np.diag(C_atm))
        else:
            if verbose:
                print("Inverting C_atm matrix")
            try:
                C_atm_inv = np.linalg.inv(C_atm)
            except np.linalg.LinAlgError:
                if verbose:
                    print("Warning: C_atm inversion failed, using pseudo-inverse")
                C_atm_inv = np.linalg.pinv(C_atm)
    else:
        # Matrices are already inverted
        C_atm_inv = C_atm
        C_noise_inv = C_noise if C_noise is not None else np.eye(A.shape[1])
    
    # Compute H = A' Cz^(-1) A + Cx^(-1)
    if verbose:
        print("Computing H = A' Cz^(-1) A + Cx^(-1)")
    
    # Check if C_noise_inv is scalar
    if isinstance(C_noise_inv, (int, float)) or (hasattr(C_noise_inv, 'size') and C_noise_inv.size == 1):
        H = C_noise_inv * np.dot(A.T, A) + C_atm_inv
    else:
        H = np.dot(A.T, np.dot(C_noise_inv, A)) + C_atm_inv
    
    # Compute H^(-1)
    if verbose:
        print("Inverting H")
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        if verbose:
            print("Warning: H inversion failed, using pseudo-inverse")
        H_inv = np.linalg.pinv(H)
    
    # Compute W = H^(-1) A' Cz^(-1)
    if verbose:
        print("Computing W = H^(-1) A' Cz^(-1)")
    
    # Check if C_noise_inv is scalar
    if isinstance(C_noise_inv, (int, float)) or (hasattr(C_noise_inv, 'size') and C_noise_inv.size == 1):
        W_mmse = C_noise_inv * np.dot(H_inv, A.T)
    else:
        W_mmse = np.dot(H_inv, np.dot(A.T, C_noise_inv))
    
    if verbose:
        print("MMSE reconstruction matrix computed")
        print(f"Matrix shape: {W_mmse.shape}")
    
    return W_mmse