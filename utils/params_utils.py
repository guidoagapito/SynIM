import os
import re
import yaml
import datetime
import re
import numpy as np
import synim

import specula
specula.init(device_idx=-1, precision=1)

from specula.calib_manager import CalibManager
from specula.lib.make_mask import make_mask
from specula.data_objects.ifunc import IFunc
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.subap_data import SubapData

def wfs_fov_from_config(wfs_params):
    if wfs_params.get('sensor_fov') is not None:
        wfs_fov_arcsec = wfs_params['sensor_fov']
    elif wfs_params.get('fov') is not None:
        wfs_fov_arcsec = wfs_params['fov']
    elif wfs_params.get('subap_wanted_fov') is not None:
        wfs_fov_arcsec = wfs_params['subap_wanted_fov']
    else:
        wfs_fov_arcsec = 0
    return wfs_fov_arcsec

def dm2d_to_3d(dm_array, mask):
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
        dm_i /= np.sqrt(np.mean(dm_i**2))
        dm_i_3d = np.zeros(mask.shape, dtype=float)
        dm_i_3d[idx] = dm_i
        dm_array_3d[:, :, i] = dm_i_3d

    return dm_array_3d

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
    # Importiamo le funzioni necessarie
    from utils.filename_generator import extract_wfs_list, extract_dm_list, is_simple_config, wfs_fov_from_config
    import re
    import numpy as np
    import os
    
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
    dm_array = None
    dm_mask = None
    
    if 'ifunc_object' in dm_params or 'ifunc_tag' in dm_params:
        if 'ifunc_tag' in dm_params:
            print(f"     Loading influence function from file, tag: {dm_params['ifunc_tag']}")
            ifunc_tag = dm_params['ifunc_tag']
        else:
            print(f"     Loading influence function from file, tag: {dm_params['ifunc_object']}")
            ifunc_tag = dm_params['ifunc_object']
        ifunc_path = cm.filename('ifunc', ifunc_tag)
        ifunc = IFunc.restore(ifunc_path)
        
        # Convert influence function from 2D to 3D
        if ifunc.mask_inf_func is not None:           
            # Create empty 3D array (height, width, n_modes)
            dm_array = dm2d_to_3d(ifunc.influence_function, ifunc.mask_inf_func)
            print(f"     DM array shape: {dm_array.shape}")
            # Create the DM mask
            dm_mask = ifunc.mask_inf_func.copy()
            print(f"     DM mask shape: {dm_mask.shape}")
            print(f"     DM mask sum: {np.sum(dm_mask)}")
        else:
            # If we don't have a mask, assume the influence function is already properly organized
            raise ValueError("IFunc without mask_inf_func is not supported. Mask is required to reconstruct the 3D array.")
            
    elif 'type_str' in dm_params:
        print(f"     Loading influence function from type_str: {dm_params['type_str']}")
        # Create influence functions directly using Zernike modes
        from specula.lib.compute_zern_ifunc import compute_zern_ifunc
        nmodes = dm_params.get('nmodes', 100)
        obsratio = dm_params.get('obsratio', 0.0)
        npixels = dm_params.get('npixels', pixel_pupil)
        
        # Compute Zernike influence functions
        z_ifunc, z_mask = compute_zern_ifunc(npixels, nmodes, xp=np, dtype=float, obsratio=obsratio, diaratio=1.0, start_mode=0, mask=None)
        
        # Create empty 3D array (height, width, n_modes)
        dm_array = dm2d_to_3d(z_ifunc, z_mask)
        print(f"     DM array shape: {dm_array.shape}")
        dm_mask = z_mask
        print(f"     DM mask shape: {dm_mask.shape}")
        print(f"     DM mask sum: {np.sum(dm_mask)}")
    else:
        raise ValueError("No valid influence function configuration found. Need either 'ifunc_tag', 'ifunc_object', or 'type_str'.")

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
        if 'lgs' in wfs_key:
            source_type = 'lgs'
        elif 'ngs' in wfs_key:
            source_type = 'ngs'
        elif 'ref' in wfs_key:
            source_type = 'ref'
        else:
            source_type = 'ngs'  # Default
    
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
    
    # Codice mancante per il caricamento di SubapData
    if 'slopec' in params:
        slopec_params = params['slopec']

    # Load SubapData for valid subapertures if available
    idx_valid_sa = None
    subap_path = None
    subap_tag = None

    # First check - Try to get subapdata from WFS params
    if 'subapdata_object' in wfs_params or 'subapdata_tag' in wfs_params:
        if 'subapdata_tag' in wfs_params:
            print("     Loading subapdata from file, tag:", wfs_params['subapdata_tag'])
            subap_tag = wfs_params['subapdata_tag']
            subap_path = cm.filename('subap_data', subap_tag)
        else:
            print("     Loading subapdata from file, tag:", wfs_params['subapdata_object'])
            subap_tag = wfs_params['subapdata_object']
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
                print(f"     Loading subapdata from {slopec_key}, tag:", slopec_params['subapdata_tag'])
                subap_tag = slopec_params['subapdata_tag']
                subap_path = cm.filename('subap_data', subap_tag)
            elif 'subapdata_object' in slopec_params:
                print(f"     Loading subapdata from {slopec_key}, tag:", slopec_params['subapdata_object'])
                subap_tag = slopec_params['subapdata_object']
                subap_path = cm.filename('subapdata', subap_tag)

    # Third check - Try generic slopec section
    elif 'slopec' in params:
        slopec_params = params['slopec']
        if 'subapdata_tag' in slopec_params:
            print("     Loading subapdata from slopec, tag:", slopec_params['subapdata_tag'])
            subap_tag = slopec_params['subapdata_tag']
            subap_path = cm.filename('subap_data', subap_tag)
        elif 'subapdata_object' in slopec_params:
            print("     Loading subapdata from slopec, tag:", slopec_params['subapdata_object'])
            subap_tag = slopec_params['subapdata_object']
            subap_path = cm.filename('subapdata', subap_tag)
            
    if subap_path is None:
        print("     No subapdata file found. Using default.")
    else:
        print("     Subapdata file found:", subap_path)

    # Try to load the subapdata if a path was found
    if subap_path and os.path.exists(subap_path):
        print("     Loading subapdata from file:", subap_path)
        subap_data = SubapData.restore(subap_path)
        idx_valid_sa = np.transpose(np.asarray(np.where(subap_data.single_mask())))
    
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
    
    # Try to get source polar coordinates
    gs_pol_coo = [0.0, 0.0]  # Default is on-axis
    
    # First check if coordinates are in WFS parameters
    if 'gs_pol_coo' in wfs_params:
        gs_pol_coo = wfs_params['gs_pol_coo']
    else:
        # Try to find source corresponding to this WFS
        source_match = re.search(r'((?:lgs|ngs|ref)\d+)', wfs_key)
        if source_match:
            source_key = f'source_{source_match.group(1)}'
            if source_key in params and 'polar_coordinates' in params[source_key]:
                gs_pol_coo = params[source_key]['polar_coordinates']
        # Try on_axis_source for simple configs
        elif 'on_axis_source' in params and 'polar_coordinates' in params['on_axis_source']:
            gs_pol_coo = params['on_axis_source']['polar_coordinates']
    
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

def compute_interaction_matrix(params_file, wfs_type=None, wfs_index=None, dm_index=None, verbose=False, display=False):
    """
    Calculates the interaction matrix for SynIM directly from a SPECULA YAML configuration file.
    
    Args:
        params_file (str or dict): Path to YAML/PRO configuration file or dictionary with configuration
        wfs_type (str, optional): Type of WFS ('lgs', 'ngs', or 'ref')
        wfs_index (int, optional): Index of the WFS to use (1-based)
        dm_index (int, optional): Index of the DM to use (1-based)
        verbose (bool): Flag to enable verbose output
        display (bool): Flag to enable visualization
        save (bool): Flag to save the interaction matrix to file
        
    Returns:
        tuple: (interaction matrix, parameters used, filename if save=True)
    """
    # Load configuration
    if isinstance(params_file, str):
        params_parsed = parse_params_file(params_file)
        # Prepare parameters for the interaction matrix
        params = prepare_interaction_matrix_params(params_parsed, wfs_type, wfs_index, dm_index)
    else:
        params = params_file
    
    if verbose:
        print("Interaction Matrix Parameters:")
        print(f"      Calculating IM for WFS: {params['wfs_key']} with DM: {params['dm_key']}")
        print(f"      WFS type: {params['wfs_type']}, subapertures: {params['wfs_nsubaps']}")
        if params['idx_valid_sa'] is not None:
            print(f"      Valid subapertures shape: {params['idx_valid_sa'].shape}")
        print(f"      WFS rotation: {params['wfs_rotation']}")
        print(f"      WFS translation: {params['wfs_translation']}")
        print(f"      WFS magnification: {params['wfs_magnification']}")
        print(f"      WFS field of view: {params['wfs_fov_arcsec']} arcsec")
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
    
    # Calculate the interaction matrix
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
        wfs_fov_arcsec=params['wfs_fov_arcsec'],
        gs_pol_coo=params['gs_pol_coo'],
        gs_height=params['gs_height'],
        idx_valid_sa=params['idx_valid_sa'],
        verbose=verbose,
        display=display
    )
    
    return im

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
        source_info['pol_coords'] = config['on_axis_source'].get('polar_coordinates', [0, 0])
        source_info['wavelength'] = config['on_axis_source'].get('wavelengthInNm', 750)
    
    return source_info

def is_simple_config(config):
    """Detect if this is a simple SCAO config or a complex MCAO config"""
    # Check for multiple DMs
    dm_count = sum(1 for key in config if key.startswith('dm') and key != 'dm')
    
    # Check for multiple WFSs
    wfs_count = sum(1 for key in config if key.startswith('sh_') or key.startswith('pyramid') and key != 'pyramid')
    
    return dm_count == 0 and wfs_count == 0
   
def generate_im_filename(config_file, wfs_type=None, wfs_index=None, dm_index=None, timestamp=False):
    """
    Generate a specific interaction matrix filename based on WFS and DM indices.
    
    Args:
        config_file (str): Path to YAML or PRO configuration file
        wfs_type (str, optional): Type of WFS ('sh', 'pyr') or source type ('lgs', 'ngs', 'ref')
        wfs_index (int, optional): Index of the WFS to use (1-based)
        dm_index (int, optional): Index of the DM to use (1-based)
        timestamp (bool, optional): Whether to include timestamp in the filename
        
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
        print("No matching WFS or DM found with the specified parameters")
        return None
    
    # Select the first WFS and DM from the filtered lists
    selected_wfs = filtered_wfs[0]
    selected_dm = filtered_dm[0]
    
    print(f"Selected WFS: {selected_wfs['name']} (type: {selected_wfs['type']}, index: {selected_wfs['index']})")
    print(f"Selected DM: {selected_dm['name']} (index: {selected_dm['index']})")
    
    # Determine the source type from the WFS name
    source_type = 'ngs'  # Default
    if 'lgs' in selected_wfs['name']:
        source_type = 'lgs'
    elif 'ngs' in selected_wfs['name']:
        source_type = 'ngs'
    elif 'ref' in selected_wfs['name']:
        source_type = 'ref'
    
    # Generate filenames for all combinations
    all_filenames = generate_im_filenames(config_file, timestamp=timestamp)
    
    # Look for a filename matching our WFS and DM in the correct source type group
    if source_type in all_filenames:
        filenames = all_filenames[source_type]
        
        # Try to find a filename containing identifiers for both our WFS and DM
        wfs_identifier = selected_wfs['name']
        dm_identifier = selected_dm['name']
        
        # First try to find an exact match
        for filename in filenames:
            # This is an approximation - we should parse the filename more precisely
            if wfs_identifier in filename and dm_identifier in filename:
                return filename
        
        # If no exact match, return the first filename for this source type
        if filenames:
            return filenames[0]
    
    # If all else fails, generate a custom filename
    # This is a simplified version - ideally, we'd follow the same pattern as in generate_im_filenames
    base_name = "IM_syn"
    wfs_part = f"wfs{selected_wfs['index']}_{selected_wfs['type']}"
    dm_part = f"dm{selected_dm['index']}"
    timestamp_part = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp else ""
    
    parts = [base_name, wfs_part, dm_part]
    if timestamp_part:
        parts.append(timestamp_part)
    
    return "_".join(parts) + ".fits"

def generate_im_filenames(config_file, timestamp=False):
    """Generate interaction matrix filenames for all WFS-DM combinations, grouped by star type"""
    
    # Load YAML or PRO configuration
    config = parse_params_file(config_file)
    
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
            wfs_fov_arcsec = wfs_fov_from_config(config['pyramid'])
            wfs_params = {
                'pup_diam': config['pyramid'].get('pup_diam', 0),
                'mod_amp': config['pyramid'].get('mod_amp', 0),
                'wavelength': config['pyramid'].get('wavelengthInNm', 0),
                'fov': wfs_fov_arcsec
            }
        elif 'sh' in config:
            wfs_type = 'sh'
            wfs_fov_arcsec = wfs_fov_from_config(config['sh'])
            wfs_params = {
                'nsubaps': config['sh'].get('subap_on_diameter', 0),
                'wavelength': config['sh'].get('wavelengthInNm', 0),
                'fov': wfs_fov_arcsec,
                'npx': config['sh'].get('subap_npx', 0)
            }
        # Source info
        source_info = {}
        if 'on_axis_source' in config:
            source_info = {
                'type': 'ngs',
                'pol_coords': config['on_axis_source'].get('polar_coordinates', [0, 0]),
                'wavelength': config['on_axis_source'].get('wavelengthInNm', 0)
            }
        
        # DM info
        dm_params = {}
        if 'dm' in config:
            dm_params = {
                'height': config['dm'].get('height', 0),
                'nmodes': config['dm'].get('nmodes', 0),
                'type': config['dm'].get('type_str', 'zernike')
            }
        
        # Build filename parts
        parts = []
        parts.append(base_name)
        
        if source_info:
            if 'pol_coords' in source_info:
                dist, angle = source_info['pol_coords']
                parts.append(f"pd{dist:.1f}a{angle:.0f}")
        
        if pupil_params:
            ps = pupil_params.get('pixel_pupil', 0)
            pp = pupil_params.get('pixel_pitch', 0)
            parts.append(f"ps{ps}p{pp:.4f}")
            
            if 'obsratio' in pupil_params and pupil_params['obsratio'] > 0:
                parts.append(f"o{pupil_params['obsratio']:.3f}")
        
        if wfs_type == 'sh':
            nsubaps = wfs_params.get('nsubaps', 0)
            wl = wfs_params.get('wavelength', 0)
            wfs_fov_arcsec = wfs_fov_from_config(wfs_params)           
            npx = wfs_params.get('npx', 0)
            parts.append(f"sh{nsubaps}x{nsubaps}_wl{wl}_fv{wfs_fov_arcsec:.1f}_np{npx}")
        
        elif wfs_type == 'pyr':
            pup_diam = wfs_params.get('pup_diam', 0)
            wl = wfs_params.get('wavelength', 0)
            mod_amp = wfs_params.get('mod_amp', 0)
            if wfs_params.get('sensor_fov') is not None:
                wfs_fov_arcsec = wfs_params['sensor_fov']
            elif wfs_params.get('fov') is not None:
                wfs_fov_arcsec = wfs_params['fov']
            elif wfs_params.get('subap_wanted_fov') is not None:
                wfs_fov_arcsec = wfs_params['subap_wanted_fov']
            else:
                wfs_fov_arcsec = 0   
            parts.append(f"pyr{pup_diam:.1f}_wl{wl}_fv{wfs_fov_arcsec:.1f}_ma{mod_amp:.1f}")

        if dm_params:
            height = dm_params.get('height', 0)
            parts.append(f"dmH{height}")
            
            # Check for custom influence functions
            if 'ifunc_tag' in config['dm']:
                parts.append(f"ifunc_{config['dm']['ifunc_tag']}")
            elif 'ifunc_object' in config['dm']:
                parts.append(f"ifunc_{config['dm']['ifunc_object']}")
            elif 'type_str' in config['dm']:
                nmodes = dm_params.get('nmodes', 0)
                parts.append(f"nm{nmodes}_{dm_params['type']}")
            else:
                # Default case
                nmodes = dm_params.get('nmodes', 0)
                parts.append(f"nm{nmodes}")

        # Add timestamp if requested
        if timestamp:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            parts.append(ts)
        
        # Join all parts with underscores and add extension
        filename = "_".join(parts) + ".fits"
        filenames_by_type['ngs'].append(filename)  # Default to NGS for simple config
    else:
        # Complex MCAO configuration: find all sources and related WFSs
        
        # Find all LGS sources and WFSs
        lgs_sources = [k for k in config.keys() if k.startswith('source_lgs')]
        for source_key in lgs_sources:
            source_idx = re.search(r'(\d+)$', source_key)
            if source_idx:
                idx = source_idx.group(1)
                wfs_key = f'sh_lgs{idx}'
                
                if wfs_key in config:
                    # Process each DM with this WFS
                    for dm in dm_list:
                        parts = []
                        parts.append(base_name)
                        
                        # Source parameters
                        source = config[source_key]
                        if 'polar_coordinates' in source:
                            dist, angle = source['polar_coordinates']
                            parts.append(f"pd{dist:.1f}a{angle:.0f}")
                        
                        if 'height' in source:
                            parts.append(f"h{source['height']:.0f}")
                        
                        # Pupil parameters
                        if pupil_params:
                            ps = pupil_params.get('pixel_pupil', 0)
                            pp = pupil_params.get('pixel_pitch', 0)
                            parts.append(f"ps{ps}p{pp:.4f}")
                            
                            if 'obsratio' in pupil_params and pupil_params['obsratio'] > 0:
                                parts.append(f"o{pupil_params['obsratio']:.3f}")
                        
                        # WFS parameters
                        wfs_config = config[wfs_key]
                        nsubaps = wfs_config.get('subap_on_diameter', 0)
                        wl = wfs_config.get('wavelengthInNm', 0)
                        fov = wfs_fov_from_config(wfs_config)
                        npx = wfs_config.get('subap_npx', 0)
                        parts.append(f"sh{nsubaps}x{nsubaps}_wl{wl}_fv{fov:.1f}_np{npx}")

                        # DM parameters
                        dm_config = dm['config']
                        height = dm_config.get('height', 0)
                        parts.append(f"dmH{height}")

                        # Check for custom influence functions
                        if 'ifunc_tag' in dm_config:
                            parts.append(f"ifunc_{dm_config['ifunc_tag']}")
                        elif 'ifunc_object' in dm_config:
                            parts.append(f"ifunc_{dm_config['ifunc_object']}")
                        elif 'type_str' in dm_config:
                            nmodes = dm_config.get('nmodes', 0)
                            parts.append(f"nm{nmodes}_{dm_config['type_str']}")
                        else:
                            # Default case
                            nmodes = dm_config.get('nmodes', 0)
                            parts.append(f"nm{nmodes}")

                        # Add WFS and DM indices to make filename unique
                        #parts.append(f"lgs{idx}_dm{dm['index']}")
                        
                        # Add timestamp if requested
                        if timestamp:
                            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            parts.append(ts)
                        
                        # Join all parts with underscores and add extension
                        filename = "_".join(parts) + ".fits"
                        filenames_by_type['lgs'].append(filename)
        
        # Find all NGS sources and WFSs
        ngs_sources = [k for k in config.keys() if k.startswith('source_ngs')]
        for source_key in ngs_sources:
            source_idx = re.search(r'(\d+)$', source_key)
            if source_idx:
                idx = source_idx.group(1)
                wfs_key = f'sh_ngs{idx}'
                
                if wfs_key in config:
                    # Process each DM with this WFS
                    for dm in dm_list:
                        parts = []
                        parts.append(base_name)
                        
                        # Source parameters
                        source = config[source_key]
                        if 'polar_coordinates' in source:
                            dist, angle = source['polar_coordinates']
                            parts.append(f"pd{dist:.1f}a{angle:.0f}")
                        
                        # Pupil parameters
                        if pupil_params:
                            ps = pupil_params.get('pixel_pupil', 0)
                            pp = pupil_params.get('pixel_pitch', 0)
                            parts.append(f"ps{ps}p{pp:.4f}")
                            
                            if 'obsratio' in pupil_params and pupil_params['obsratio'] > 0:
                                parts.append(f"o{pupil_params['obsratio']:.3f}")
                        
                        # WFS parameters
                        wfs_config = config[wfs_key]
                        nsubaps = wfs_config.get('subap_on_diameter', 0)
                        wl = wfs_config.get('wavelengthInNm', 0)
                        fov = wfs_fov_from_config(wfs_config)
                        npx = wfs_config.get('subap_npx', 0)
                        parts.append(f"sh{nsubaps}x{nsubaps}_wl{wl}_fv{fov:.1f}_np{npx}")
                        
                        # DM parameters
                        dm_config = dm['config']
                        height = dm_config.get('height', 0)
                        nmodes = dm_config.get('nmodes', 0)
                        parts.append(f"dmH{height}_nm{nmodes}")
                        
                        # Add WFS and DM indices to make filename unique
                        #parts.append(f"ngs{idx}_dm{dm['index']}")
                        
                        # Add timestamp if requested
                        if timestamp:
                            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            parts.append(ts)
                        
                        # Join all parts with underscores and add extension
                        filename = "_".join(parts) + ".fits"
                        filenames_by_type['ngs'].append(filename)
        
        # Find all REF sources and WFSs
        ref_sources = [k for k in config.keys() if k.startswith('source_ref')]
        for source_key in ref_sources:
            source_idx = re.search(r'(\d+)$', source_key)
            if source_idx:
                idx = source_idx.group(1)
                wfs_key = f'sh_ref{idx}'
                
                if wfs_key in config:
                    # Process each DM with this WFS
                    for dm in dm_list:
                        parts = []
                        parts.append(base_name)
                        
                        # Source parameters
                        source = config[source_key]
                        if 'polar_coordinates' in source:
                            dist, angle = source['polar_coordinates']
                            parts.append(f"pd{dist:.1f}a{angle:.0f}")
                        
                        # Pupil parameters
                        if pupil_params:
                            ps = pupil_params.get('pixel_pupil', 0)
                            pp = pupil_params.get('pixel_pitch', 0)
                            parts.append(f"ps{ps}p{pp:.4f}")
                            
                            if 'obsratio' in pupil_params and pupil_params['obsratio'] > 0:
                                parts.append(f"o{pupil_params['obsratio']:.3f}")
                        
                        # WFS parameters
                        wfs_config = config[wfs_key]
                        nsubaps = wfs_config.get('subap_on_diameter', 0)
                        wl = wfs_config.get('wavelengthInNm', 0)
                        fov = wfs_fov_from_config(wfs_config)
                        npx = wfs_config.get('subap_npx', 0)
                        parts.append(f"sh{nsubaps}x{nsubaps}_wl{wl}_fv{fov:.1f}_np{npx}")
                        
                        # DM parameters
                        dm_config = dm['config']
                        height = dm_config.get('height', 0)
                        nmodes = dm_config.get('nmodes', 0)
                        parts.append(f"dmH{height}_nm{nmodes}")
                        
                        # Add WFS and DM indices to make filename unique
                        #parts.append(f"ref{idx}_dm{dm['index']}")
                        
                        # Add timestamp if requested
                        if timestamp:
                            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            parts.append(ts)
                        
                        # Join all parts with underscores and add extension
                        filename = "_".join(parts) + ".fits"
                        filenames_by_type['ref'].append(filename)
    
    return filenames_by_type