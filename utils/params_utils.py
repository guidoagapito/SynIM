import os
import re
import yaml
import numpy as np
import synim

import specula
specula.init(device_idx=-1, precision=1)

from specula.calib_manager import CalibManager
from specula.lib.make_mask import make_mask
from specula.data_objects.ifunc import IFunc
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.subap_data import SubapData

def prepare_interaction_matrix_params(params_file, wfs_type=None, wfs_index=None, dm_index=None):
    """
    Prepares parameters for synim.interaction_matrix from a SPECULA YAML configuration file.
    
    Args:
        params_file (str): Path to the YAML or PRO configuration file
        wfs_type (str, optional): Type of WFS ('lgs', 'ngs', or 'ref')
        wfs_index (int, optional): Index of the WFS to use (1-based)
        dm_index (int, optional): Index of the DM to use (1-based)
        
    Returns:
        dict: Parameters ready to be passed to synim.interaction_matrix
    """
    # Load the YAML or PRO file
    
    params = parse_params_file(params_file)
    
    # Prepare the CalibManager
    main_params = params['main']
    cm = CalibManager(main_params['root_dir'])
    
    # Extract general parameters
    pixel_pupil = main_params['pixel_pupil']
    pixel_pitch = main_params['pixel_pitch']
    pup_diam_m = pixel_pupil * pixel_pitch
    
    # Load pupilstop and create pupil mask
    pup_mask = None
    if 'pupilstop' in params:
        pupilstop_params = params['pupilstop']
        if 'object' in pupilstop_params or 'pupil_mask_tag' in pupilstop_params:
            # Load pupilstop from file
            if 'pupil_mask_tag' in pupilstop_params:
                pupilstop_tag = pupilstop_params['pupil_mask_tag']
            else:
                pupilstop_tag = pupilstop_params['object']
            pupilstop_path = cm.filename('pupilstop', pupilstop_tag)
            pupilstop = Pupilstop.restore(pupilstop_path)
            pup_mask = pupilstop.A  # Use the amplitude attribute of Pupilstop
        else:
            # Create pupilstop from parameters
            mask_diam = pupilstop_params.get('mask_diam', 1.0)
            obs_diam = pupilstop_params.get('obs_diam', None)
            
            # Create a new Pupilstop instance with the given parameters
            pupilstop = Pupilstop(
                pixel_pupil=pixel_pupil,
                pixel_pitch=pixel_pitch,
                mask_diam=mask_diam,
                obs_diam=obs_diam,
                target_device_idx=-1,
                precision=0
            )
            pup_mask = pupilstop.A  # Use the amplitude attribute
    
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
    
    # Find DM parameters based on specified index or use the first available one
    print("DM -- Looking for DM parameters...")
    if dm_index is not None:
        print(f"     Using specified DM index: {dm_index}")
        dm_key = f'dm{dm_index}'
        if dm_key not in params:
            # Fallback: if specified DM doesn't exist, try 'dm' without number
            if 'dm' in params:
                dm_key = 'dm'
                print(f"     DM with index {dm_index} not found. Using 'dm' section instead.")
            else:
                raise ValueError(f"DM with index {dm_index} not found in YAML file.")
        else:
            print(f"     Using specified DM: {dm_key}")
        dm_params = params[dm_key]
    else:
        # Original behavior: find the first available DM
        dm_keys = [key for key in params if key.startswith('dm')]
        if dm_keys:
            dm_key = dm_keys[0]
            dm_params = params[dm_key]
            print(f"     Using first available DM: {dm_key}")
        else:
            raise ValueError("No DM configuration found in the YAML file.")
    
    # Extract DM parameters
    dm_height = dm_params.get('height', 0)
    dm_rotation = dm_params.get('rotation', 0.0)
    
    # Load influence functions
    dm_array = None
    dm_mask = None
    if 'ifunc_object' in dm_params or 'ifunc_tag' in dm_params:
        if 'ifunc_tag' in dm_params:
            print("     Loading influence function from file, tag:", dm_params['ifunc_tag'])
            ifunc_tag = dm_params['ifunc_tag']
        else:
            print("     Loading influence function from file, tag:", dm_params['ifunc_object'])
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
        print("     Loading influence function from type_str:", dm_params['type_str'])
        # Create influence functions directly using Zernike modes
        from specula.lib.compute_zern_ifunc import compute_zern_ifunc
        nmodes = dm_params.get('nmodes', 100)
        obsratio = dm_params.get('obsratio', 0.0)
        npixels = dm_params.get('npixels', pixel_pupil)
        
        # Compute Zernike influence functions
        #mask = make_mask(npixels, obsratio, 1.0, get_idx=False, xp=np)
        z_ifunc, z_mask = compute_zern_ifunc(npixels, nmodes, xp=np, dtype=float, obsratio=obsratio, diaratio=1.0, start_mode=0, mask=None)
        
        # Convert to the right format for SynIM
        dm_array = np.zeros((npixels**2, nmodes), dtype=float)
        for i in range(nmodes):
            dm_array[np.where(z_mask.flat),i] = z_ifunc[i]
        dm_array = dm_array.reshape((npixels, npixels, nmodes))
        print("     DM array shape:", dm_array.shape)
        dm_mask = z_mask
        print("     DM mask shape:", dm_mask.shape)
        print("     DM mask sum:", np.sum(dm_mask))

    else:
        # If no influence function is specified, raise an error
        raise ValueError("No influence function specified in the YAML file. Please provide a valid influence function.")    

    if dm_array is None:
        raise ValueError("No influence function data found for the specified DM.")

    print("WFS -- Looking for WFS parameters...")
    # Find WFS parameters based on the specified type and index
    wfs_found = False
    wfs_key = None
    wfs_params = None
    
    # Try to find the specified WFS - ora cerca anche 'sh' o 'pyramid'
    # Expanded WFS search to include standard SPECULA sections
    wfs_keys = []
    
    # First, look for sections starting with 'wfs'
    wfs_sections = [key for key in params if key.startswith('wfs')]
    wfs_keys.extend(wfs_sections)
    
    # Look for 'sh' and 'pyramid' sections
    if 'sh' in params:
        print("      Found 'sh' section in YAML file.")
        wfs_keys.append('sh')
    if 'pyramid' in params:
        print("      Found 'pyramid' section in YAML file.")
        wfs_keys.append('pyramid')
    
    # Look for sections starting with sh_ or pyramid_
    for key in params:
        if key.startswith('sh_') or key.startswith('pyramid_'):
            print(f"      Found '{key}' section in YAML file.")
            wfs_keys.append(key)
    
    # Process specific wfs_type if provided
    if wfs_type is not None:
        # Filter keys by type
        if wfs_type == 'sh':
            potential_keys = [k for k in wfs_keys if k == 'sh' or k.startswith('sh_')]
        elif wfs_type == 'pyr':
            potential_keys = [k for k in wfs_keys if k == 'pyramid' or k.startswith('pyramid_')]
        else:
            potential_keys = [k for k in wfs_keys if k.endswith(f"_{wfs_type}")]
        
        # Further filter by index if provided
        if wfs_index is not None and potential_keys:
            for key in potential_keys:
                if f"_{wfs_index}" in key or (wfs_index == 1 and key in ['sh', 'pyramid']):
                    wfs_key = key
                    wfs_params = params[key]
                    wfs_found = True
                    break
        elif potential_keys:
            # Take the first one matching the type
            wfs_key = potential_keys[0]
            wfs_params = params[wfs_key]
            wfs_found = True
    
    # If not found by type or no type specified, try by index
    elif wfs_index is not None:
        # Try specific keys with index
        test_keys = [f'wfs{wfs_index}', f'sh_{wfs_index}', f'pyramid_{wfs_index}']
        for key in test_keys:
            if key in params:
                wfs_key = key
                wfs_params = params[key]
                wfs_found = True
                break
        
        # Special case for index 1, might be just 'sh' or 'pyramid'
        if not wfs_found and wfs_index == 1:
            if 'sh' in params:
                wfs_key = 'sh'
                wfs_params = params[wfs_key]
                wfs_found = True
            elif 'pyramid' in params:
                wfs_key = 'pyramid'
                wfs_params = params[wfs_key]
                wfs_found = True
    
    # If no specific search criteria, use the first available WFS
    if not wfs_found and wfs_keys:
        wfs_key = wfs_keys[0]
        wfs_params = params[wfs_key]
        wfs_found = True
    
    if not wfs_found:
        raise ValueError("No matching WFS configuration found in the YAML file. Available keys: " + 
                         ", ".join(list(params.keys())))
    
    # Setup WFS parameters
    # Determine WFS type based on the key
    if wfs_key == 'pyramid' or wfs_key.startswith('pyramid'):
        wfs_type = 'pyr'
    elif wfs_key == 'sh' or wfs_key.startswith('sh'):
        wfs_type = 'sh'
    else:
        wfs_type = wfs_params.get('type', 'sh')  # Default to Shack-Hartmann
    
    # Define WFS parameters with appropriate defaults
    if wfs_type == 'sh':
        # Shack-Hartmann parameters
        wfs_nsubaps = wfs_params.get('nsubaps', wfs_params.get('subap_on_diameter', 1))
        wfs_wavelength = wfs_params.get('wavelengthInNm', 750)
    else:
        # Pyramid parameters
        wfs_nsubaps = wfs_params.get('nsubaps', wfs_params.get('pup_diam', 1))
        wfs_wavelength = wfs_params.get('wavelengthInNm', 750)
    
    # Common WFS parameters
    wfs_rotation = wfs_params.get('rotation', 0.0)
    wfs_translation = wfs_params.get('translation', [0.0, 0.0])
    wfs_magnification = wfs_params.get('magnification', 1.0)
    if np.isnan(wfs_magnification):
        wfs_magnification = 1.0
    if np.size(wfs_magnification) == 1:
        wfs_magnification = [wfs_magnification, wfs_magnification]
    
    # Load SubapData for valid subapertures if available
    idx_valid_sa = None
    if 'subap_object' in wfs_params or 'subapdata_tag' in wfs_params:
        if 'subapdata_tag' in wfs_params:
            print("     Loading subapdata from file, tag:", wfs_params['subapdata_tag'])
            subap_tag = wfs_params['subapdata_tag']
        else:
            print("     Loading subapdata from file, tag:", wfs_params['subap_object'])
            subap_tag = wfs_params['subap_object']
        subap_path = cm.filename('subap_data', subap_tag)
        if os.path.exists(subap_path):
            subap_data = SubapData.restore(subap_path)
            idx_valid_sa = subap_data.idx_valid_sa
    
    # If we don't have idx_valid_sa from the file, calculate an estimate
    if idx_valid_sa is None:
        # Calculate valid subapertures based on the pupil mask
        from scipy.ndimage import zoom
        
        # Resize pupil mask to match the WFS subaperture grid
        zoom_factor = wfs_nsubaps / pup_mask.shape[0]
        binned_mask = zoom(pup_mask, zoom_factor, order=0)
        
        # Consider a subaperture valid if it's at least half illuminated
        valid_threshold = 0.5
        subap_illumination = np.zeros((wfs_nsubaps, wfs_nsubaps))
        
        # Calculate illumination for each subaperture
        for i in range(wfs_nsubaps):
            for j in range(wfs_nsubaps):
                subap_illumination[i, j] = np.mean(binned_mask[i, j])
        
        # Get indices of valid subapertures
        valid_indices = np.where(subap_illumination >= valid_threshold)
        idx_valid_sa = np.column_stack((valid_indices[0], valid_indices[1]))
    
    # Guide star parameters
    gs_pol_coo = wfs_params.get('gs_pol_coo', [0.0, 0.0])  # [theta, rho] in radians and arcsec
    
    # L'altezza dovrebbe essere infinita per NGS (stelle naturali) e specifica per LGS (stelle laser)
    if wfs_type == 'sh' or wfs_key.startswith('sh'):
        # Per Shack-Hartmann, verifica se è esplicitamente specificata un'altezza
        gs_height = wfs_params.get('gs_height', float('inf'))  # Default a infinito per NGS
        
        # Se non c'è altezza ma c'è 'type' che specifica 'lgs', usa un'altezza tipica delle LGS
        if gs_height == float('inf') and wfs_params.get('type') == 'lgs':
            gs_height = wfs_params.get('lgs_height', 90000.0)  # Altezza tipica delle LGS (90km)
    else:
        # Per altre tipologie di WFS, default a infinito
        gs_height = wfs_params.get('gs_height', float('inf'))
    
    # Try to get source info from on_axis_source if no guide star parameters found
    if 'on_axis_source' in params and gs_pol_coo == [0.0, 0.0]:
        source = params['on_axis_source']
        if 'polar_coordinates' in source:
            gs_pol_coo = source['polar_coordinates']
        
        # Se stiamo usando on_axis_source, è sicuramente una NGS (infinito)
        if gs_height == 0.0:
            gs_height = float('inf')

    # Try to get source info from on_axis_source if no guide star parameters found
    if 'on_axis_source' in params and gs_pol_coo == [0.0, 0.0]:
        source = params['on_axis_source']
        if 'polar_coordinates' in source:
            gs_pol_coo = source['polar_coordinates']

    return {
        'pup_diam_m': pup_diam_m,
        'pup_mask': pup_mask,
        'dm_array': dm_array,
        'dm_mask': dm_mask,
        'dm_height': dm_height,
        'dm_rotation': dm_rotation,
        'wfs_key': wfs_key,
        'wfs_type': wfs_type,
        'wfs_nsubaps': wfs_nsubaps,
        'wfs_rotation': wfs_rotation,
        'wfs_translation': wfs_translation,
        'wfs_magnification': wfs_magnification,
        'gs_pol_coo': gs_pol_coo,
        'gs_height': gs_height,
        'idx_valid_sa': idx_valid_sa,
        'dm_key': dm_key
    }

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
            # Rimuovi commenti e spazi bianchi
            line = line.split(';')[0].strip()
            if not line:
                continue

            # Riconosci l'inizio di una nuova sezione (e.g., {main, {DM, etc.)
            section_match = re.match(r'^\{(\w+),', line)
            if section_match:
                current_section = section_match.group(1).lower()
                data[current_section] = {}
                continue

            # Riconosci la fine di una sezione
            if line == '}':
                current_section = None
                continue

            # Se siamo in una sezione, processa le coppie chiave-valore
            if current_section:
                key_value_match = re.match(r'(\w+)\s*[:=]\s*(.+)', line)
                if key_value_match:
                    key = key_value_match.group(1).strip()
                    value = key_value_match.group(2).strip()

                    # Rimuovi eventuali virgole finali
                    if value.endswith(','):
                        value = value[:-1].strip()

                    # Rimuovi apici singoli attorno alle stringhe
                    if value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]

                    # Interpreta i tipi di valore
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    elif re.match(r'^-?\d+(\.\d+)?$', value):  # Intero o float
                        value = float(value) if '.' in value else int(value)
                    elif re.match(r'^\[.*\]$', value):  # Lista
                        value = eval(value)  # Usa eval per interpretare la lista
                    elif re.match(r'^[\d\.]+/[^\s]+$', value):  # Espressione matematica (e.g., 8.118/160)
                        try:
                            value = eval(value)
                        except Exception:
                            pass
                    elif value.lower() == '!values.f_infinity':  # Caso speciale per infinito
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
        print("Interaction Matrix Parameters:")
        print(f"      YAML file: {yaml_file}")
        print(f"      Calculating IM for WFS: {params['wfs_key']} with DM: {params['dm_key']}")
        print(f"      WFS type: {params['wfs_type']}, subapertures: {params['wfs_nsubaps']}")
        print(f"      Valid subapertures shape: {params['idx_valid_sa'].shape}")
        print(f"      WFS rotation: {params['wfs_rotation']}")
        print(f"      WFS translation: {params['wfs_translation']}")
        print(f"      WFS magnification: {params['wfs_magnification']}")
        print(f"      DM height: {params['dm_height']}")
        print(f"      DM rotation: {params['dm_rotation']}")
        print(f"      DM array shape: {params['dm_array'].shape}")
        print(f"      DM mask shape: {params['dm_mask'].shape}")
        print(f"      Guide star: {params['gs_pol_coo']} at height {params['gs_height']} m")
        print(f"      source polar coordinates: {params['gs_pol_coo']}")
        print(f"      Pupil diameter: {params['pup_diam_m']} m")
        print(f"      Pupil mask shape: {params['pup_mask'].shape}")
    
    if display:
        print("Displaying parameters...")
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        plt.imshow(params['pup_mask'], cmap='gray')
        plt.colorbar()
        plt.title("Pupil Mask")
        plt.figure(figsize=(10, 8))
        plt.imshow(params['dm_mask'], cmap='gray')
        plt.colorbar()
        plt.title("DM Mask")
        plt.figure(figsize=(10, 8))
        plt.imshow(params['dm_array'][:, :, 0], cmap='gray')
        plt.colorbar()
        plt.title("DM Influence Function (First Mode)")
        plt.figure(figsize=(10, 8))
        sa_mask = np.zeros((params['wfs_nsubaps'], params['wfs_nsubaps']))
        sa_mask[params['idx_valid_sa'][:, 0], params['idx_valid_sa'][:, 1]] = 1
        plt.imshow(sa_mask, cmap='gray')
        plt.colorbar()
        plt.title("Valid Subapertures")
        plt.show()

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