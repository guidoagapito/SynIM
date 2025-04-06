import os
import yaml
import numpy as np
import synim
from specula.calib_manager import CalibManager
from specula.data_objects.ifunc import IFunc
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.subap_data import SubapData

def prepare_interaction_matrix_params(yaml_file, wfs_type=None, wfs_index=None, dm_index=None):
    """
    Prepares parameters for synim.interaction_matrix from a SPECULA YAML configuration file.
    
    Args:
        yaml_file (str): Path to the YAML configuration file
        wfs_type (str, optional): Type of WFS ('lgs', 'ngs', or 'ref')
        wfs_index (int, optional): Index of the WFS to use (1-based)
        dm_index (int, optional): Index of the DM to use (1-based)
        
    Returns:
        dict: Parameters ready to be passed to synim.interaction_matrix
    """
    # Load the YAML file
    with open(yaml_file, 'r') as stream:
        params = yaml.safe_load(stream)
    
    # Prepare the CalibManager
    main_params = params['main']
    cm = CalibManager(main_params['root_dir'])
    
    # Extract general parameters
    pixel_pupil = main_params['pixel_pupil']
    pixel_pitch = main_params['pixel_pitch']
    pup_diam_m = pixel_pupil * pixel_pitch
    
    # Find DM parameters based on specified index or use the first available one
    if dm_index is not None:
        dm_key = f'dm{dm_index}'
        if dm_key not in params:
            # Fallback: if specified DM doesn't exist, try 'dm' without number
            if 'dm' in params:
                dm_key = 'dm'
            else:
                raise ValueError(f"DM with index {dm_index} not found in YAML file.")
        dm_params = params[dm_key]
    else:
        # Original behavior: find the first available DM
        dm_keys = [key for key in params if key.startswith('dm')]
        if dm_keys:
            dm_params = params[dm_keys[0]]
        else:
            raise ValueError("No DM configuration found in the YAML file.")
    
    # Extract DM parameters
    dm_height = dm_params.get('height', 0)
    dm_rotation = dm_params.get('rotation', 0.0)
    
    # Load influence functions
    dm_array = None
    if 'ifunc_object' in dm_params:
        ifunc_tag = dm_params['ifunc_object']
        ifunc_path = cm.filename('ifunc', ifunc_tag)
        ifunc = IFunc.restore(ifunc_path)
        
        # Convert influence function from 2D to 3D
        if ifunc.mask_inf_func is not None:
            # If we have a mask, reconstruct the complete 3D array
            mask_shape = ifunc.mask_inf_func.shape
            n_modes = ifunc.influence_function.shape[0] if ifunc.influence_function.ndim > 1 else 1
            
            # Create empty 3D array (height, width, n_modes)
            dm_array = np.zeros((mask_shape[0], mask_shape[1], n_modes), dtype=float)
            
            # Fill the 3D array using the mask
            if n_modes > 1:
                # For multiple modes
                for i in range(n_modes):
                    temp = np.zeros(mask_shape, dtype=float)
                    idx = np.where(ifunc.mask_inf_func > 0)
                    temp[idx] = ifunc.influence_function[i, :]
                    dm_array[:, :, i] = temp
            else:
                # For a single mode
                temp = np.zeros(mask_shape, dtype=float)
                idx = np.where(ifunc.mask_inf_func > 0)
                temp[idx] = ifunc.influence_function
                dm_array[:, :, 0] = temp
                
            # Create the DM mask
            dm_mask = ifunc.mask_inf_func.copy()
        else:
            # If we don't have a mask, assume the influence function is already properly organized
            raise ValueError("IFunc without mask_inf_func is not supported. Mask is required to reconstruct the 3D array.")
            
    elif 'type_str' in dm_params:
        # Create influence functions directly
        from specula.lib.compute_zern_ifunc import compute_zern_ifunc
        nmodes = dm_params.get('nmodes', 100)
        obsratio = dm_params.get('obsratio', 0.0)
        npixels = dm_params.get('npixels', pixel_pupil)
        
        # Rest of the function...
        
    # Find WFS parameters based on the specified type and index
    # Load pupilstop
    # Load SubapData for idx_valid_sa
    # If we don't have idx_valid_sa from the file, calculate an estimate

    # Rest of the function...
    
    return {
        'pup_diam_m': pup_diam_m,
        'pup_mask': pup_mask,
        'dm_array': dm_array,
        'dm_mask': dm_mask,
        'dm_height': dm_height,
        'dm_rotation': dm_rotation,
        # Rest of the parameters...
    }

def compute_interaction_matrix(yaml_file, wfs_type=None, wfs_index=None, dm_index=None, verbose=False, display=False):
    """
    Calculates the interaction matrix for SynIM directly from a SPECULA YAML configuration file.
    
    Args:
        yaml_file (str): Path to the YAML configuration file
        wfs_type (str, optional): Type of WFS ('lgs', 'ngs', or 'ref')
        wfs_index (int, optional): Index of the WFS to use (1-based)
        dm_index (int, optional): Index of the DM to use (1-based)
        verbose (bool): Flag to enable verbose output
        display (bool): Flag to enable visualization
        
    Returns:
        tuple: (interaction matrix, parameters used)
    """
    params = prepare_interaction_matrix_params(
        yaml_file, 
        wfs_type=wfs_type, 
        wfs_index=wfs_index, 
        dm_index=dm_index
    )
    
    if verbose:
        print(f"Calculating IM for WFS: {params['wfs_key']} with DM: {params['dm_key']}")
        print(f"WFS type: {params['wfs_type']}, subapertures: {params['wfs_nsubaps']}")
        print(f"DM height: {params['dm_height']}")
        print(f"Guide star: {params['gs_pol_coo']} at height {params['gs_height']} m")
    
    im = synim.interaction_matrix(
        pup_diam_m=params['pup_diam_m'],
        pup_mask=params['pup_mask'],
        dm_array=params['dm_array'],
        dm_mask=params['dm_mask'],
        dm_height=params['dm_height'],
        dm_rotation=params['dm_rotation'],
        wfs_nsubaps=params['wfs_nsubaps'],
        wfs_rotation=params['wfs_rotation'],
        wfs_translation=params['wfs_translation'],
        wfs_magnification=params['wfs_magnification'],
        gs_pol_coo=params['gs_pol_coo'],
        gs_height=params['gs_height'],
        idx_valid_sa=params['idx_valid_sa'],
        verbose=verbose,
        display=display
    )
    
    return im, params