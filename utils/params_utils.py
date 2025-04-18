import os
import re
import yaml
import datetime
import numpy as np
import matplotlib.pyplot as plt
import synim


# Import all utility functions from params_common_utils
from utils.params_common_utils import *

import specula
specula.init(device_idx=-1, precision=1)

from specula.calib_manager import CalibManager
from specula.data_objects.intmat import Intmat

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
        
    Returns:
        numpy.ndarray: Computed interaction matrix
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

def compute_interaction_matrices(yaml_file, root_dir=None, output_im_dir=None, output_rec_dir=None,
                                 wfs_type=None, overwrite=False, verbose=False, display=False):
    """
    Computes and saves interaction matrices for all combinations of WFSs and DMs
    based on a SPECULA YAML configuration file.
    
    Args:
        yaml_file (str): Path to the YAML configuration file.
        root_dir (str, optional): Root directory to set in params['main']['root_dir']. 
                                 If None, uses SPECULA repo path.
        output_im_dir (str, optional): Output directory for saved matrices. 
                                      If None, uses calib/MCAO/im in the SPECULA repo.
        output_rec_dir (str, optional): Output directory for reconstruction matrices.
                                      If None, uses calib/MCAO/rec in the SPECULA repo.
        wfs_type (str, optional): Type of WFS ('ngs', 'lgs', 'ref') to use. If None, uses all types.
        overwrite (bool, optional): Whether to overwrite existing files.
        verbose (bool, optional): Whether to print detailed information.
        display (bool, optional): Whether to display plots of interaction matrices.
        
    Returns:
        dict: Dictionary mapping WFS-DM pairs to saved interaction matrix paths.
    """
    # Load the YAML or PRO file
    paramsAll = parse_params_file(yaml_file)
    
    # Find the SPECULA repository path
    specula_init_path = specula.__file__
    specula_package_dir = os.path.dirname(specula_init_path)
    specula_repo_path = os.path.dirname(specula_package_dir)
    
    # Set up directories
    if root_dir is None:
        root_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "MCAO")
    
    if output_im_dir is None:
        output_im_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "MCAO", "im")
    
    if output_rec_dir is None:
        output_rec_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "MCAO", "rec")
    
    # Update root_dir in params
    if 'main' in paramsAll:
        paramsAll['main']['root_dir'] = root_dir
        if verbose:
            print(f"Root directory set to: {paramsAll['main']['root_dir']}")
    
    # Make sure the output directories exist
    os.makedirs(output_im_dir, exist_ok=True)
    os.makedirs(output_rec_dir, exist_ok=True)
    
    # Get WFS and DM lists
    wfs_list = extract_wfs_list(paramsAll)
    dm_list = extract_dm_list(paramsAll)
    
    # Filter by wfs_type if specified
    if wfs_type is not None:
        filtered_wfs_list = []
        for wfs in wfs_list:
            if wfs_type in wfs['name']:
                filtered_wfs_list.append(wfs)
        wfs_list = filtered_wfs_list
    
    if verbose:
        print(f"Found {len(wfs_list)} WFS(s) and {len(dm_list)} DM(s)")
        for wfs in wfs_list:
            print(f"  WFS: {wfs['name']} (type: {wfs['type']}, index: {wfs['index']})")
        for dm in dm_list:
            print(f"  DM: {dm['name']} (index: {dm['index']})")
    
    # Dictionary to store saved matrix paths
    saved_matrices = {}
    
    # Process each WFS-DM combination
    for wfs in wfs_list:
        wfs_idx = int(wfs['index'])
        wfs_name = wfs['name']
        
        for dm in dm_list:
            dm_idx = int(dm['index'])
            dm_name = dm['name']
            
            if verbose:
                print(f"\nProcessing WFS {wfs_name} (index {wfs_idx}) and DM {dm_name} (index {dm_idx})")
            
            # Determine source type from WFS name
            source_type = determine_source_type(wfs_name)
            
            # Generate filename for this combination
            im_filename = generate_im_filename(yaml_file, wfs_type=source_type, 
                                              wfs_index=wfs_idx, dm_index=dm_idx)
            
            # Full path for the file
            im_path = os.path.join(output_im_dir, im_filename)
            
            # Check if the file already exists
            if os.path.exists(im_path) and not overwrite:
                if verbose:
                    print(f"  File {im_filename} already exists. Skipping computation.")
                saved_matrices[f"{wfs_name}_{dm_name}"] = im_path
                continue
            
            # Prepare parameters for interaction matrix computation
            params = prepare_interaction_matrix_params(paramsAll, wfs_type=source_type, 
                                                     wfs_index=wfs_idx, dm_index=dm_idx)
            
            # Calculate the interaction matrix
            im = compute_interaction_matrix(params, verbose=verbose, display=display)
            
            # Transpose to be coherent with the specula convention
            im = im.transpose() * 2 * np.pi
            
            if verbose:
                print(f"  Interaction matrix shape: {im.shape}")
                print(f'  First few rows of IM: {im[0:5,:]}')
            
            # Display the matrix if requested
            if display:
                plt.figure(figsize=(10, 8))
                plt.imshow(im, cmap='viridis')
                plt.colorbar()
                plt.title(f"Interaction Matrix: {wfs_name} - {dm_name}")
                plt.tight_layout()
                plt.show()
            
            # Create the Intmat object
            wfs_info = f"{params['wfs_type']}_{params['wfs_nsubaps']}"
            pupdata_tag = f"{os.path.basename(yaml_file).split('.')[0]}_{wfs_info}"
            
            # Create Intmat object and save it
            intmat_obj = Intmat(
                im, 
                pupdata_tag=pupdata_tag,
                norm_factor=1.0,  # Default value
                target_device_idx=None,  # Use default device
                precision=None    # Use default precision
            )
            
            # Save the interaction matrix
            intmat_obj.save(im_path)
            if verbose:
                print(f"  Interaction matrix saved as: {im_path}")
            
            saved_matrices[f"{wfs_name}_{dm_name}"] = im_path
    
    return saved_matrices

def combine_interaction_matrices(yaml_file, output_im_dir=None, wfs_type='ngs', n_modes=None, 
                                dm_indices=None, verbose=False, display=False):
    """
    Loads and combines individual interaction matrices into a full system matrix.
    
    Args:
        yaml_file (str): Path to the YAML configuration file.
        output_im_dir (str, optional): Directory where interaction matrices are stored.
        wfs_type (str, optional): Type of WFS ('ngs', 'lgs', 'ref') to use.
        n_modes (int, list, or dict, optional): 
            - If int: Total number of modes for the combined matrix
            - If list: Number of modes per DM by index position (e.g., [2, 0, 3] means 2 modes from DM1, 0 from DM2, 3 from DM3)
            - If dict: Number of modes per DM with keys as DM indices (e.g., {1: 2, 3: 3} means 2 modes from DM1, 3 from DM3)
        dm_indices (list, optional): List of DM indices to include. If None, uses all.
        verbose (bool, optional): Whether to print detailed information.
        display (bool, optional): Whether to display plots of the combined matrix.
        
    Returns:
        numpy.ndarray: The combined interaction matrix.
    """
    # Find the SPECULA repository path
    specula_init_path = specula.__file__
    specula_package_dir = os.path.dirname(specula_init_path)
    specula_repo_path = os.path.dirname(specula_package_dir)
    
    # Set up output directory
    if output_im_dir is None:
        output_im_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "MCAO", "im")
    
    # Load the YAML file
    paramsAll = parse_params_file(yaml_file)
    
    # Get WFS and DM lists
    wfs_list = extract_wfs_list(paramsAll)
    dm_list = extract_dm_list(paramsAll)
    
    # Filter WFSs by type
    filtered_wfs = []
    for wfs in wfs_list:
        if wfs_type in wfs['name']:
            filtered_wfs.append(wfs)
    
    # Filter DMs by indices if specified
    if dm_indices is not None:
        filtered_dms = []
        for dm in dm_list:
            if int(dm['index']) in dm_indices:
                filtered_dms.append(dm)
        dm_list = filtered_dms
    
    # First, determine n_slopes by loading the first available matrix
    n_slopes = None
    
    # Try to load the first available matrix to determine n_slopes
    for wfs in filtered_wfs:
        wfs_idx = int(wfs['index'])
        
        for dm in dm_list:
            dm_idx = int(dm['index'])
            
            im_filename = generate_im_filename(yaml_file, wfs_type=wfs_type, 
                                             wfs_index=wfs_idx, dm_index=dm_idx)
            im_path = os.path.join(output_im_dir, im_filename)
            
            if os.path.exists(im_path):
                if verbose:
                    print(f"Loading {im_filename} to determine matrix dimensions")
                
                intmat_obj = Intmat.restore(im_path)
                n_slopes = intmat_obj._intmat.shape[1]
                
                if verbose:
                    print(f"Determined n_slopes = {n_slopes} from matrix shape {intmat_obj._intmat.shape}")
                
                break
        
        if n_slopes is not None:
            break
    
    if n_slopes is None:
        raise ValueError("Could not determine n_slopes. No interaction matrices found.")
    
    # Process n_modes parameter
    mode_map = {}
    total_modes = 0
    
    if n_modes is None:
        # Default: 2 modes for DM1 and 3 modes for DM3
        mode_map = {1: list(range(2)), 3: list(range(2, 5))}
        total_modes = 5
    elif isinstance(n_modes, int):
        # Backward compatibility: distribute modes with default logic
        current_idx = 0
        for dm in dm_list:
            dm_idx = int(dm['index'])
            
            # Configure mode indices based on DM index
            if dm_idx == 1:  # First 2 modes for DM1
                mode_map[dm_idx] = list(range(2))
                current_idx = 2
            elif dm_idx == 2:  # Skip DM2 (typically tip-tilt mirror)
                continue
            elif dm_idx == 3:  # Next 3 modes for DM3
                mode_map[dm_idx] = list(range(current_idx, current_idx + 3))
                current_idx += 3
            else:
                # Default allocation
                nm = min(2, n_modes - current_idx)  # Default 2 modes per DM
                if nm <= 0:
                    break  # No more modes to allocate
                mode_map[dm_idx] = list(range(current_idx, current_idx + nm))
                current_idx += nm
        
        total_modes = n_modes
    elif isinstance(n_modes, list):
        # List format: position corresponds to DM index - 1
        current_idx = 0
        for i, nm in enumerate(n_modes):
            dm_idx = i + 1  # Convert to 1-based index
            
            # Skip if modes count is 0
            if nm <= 0:
                continue
                
            # Check if this DM is in our filtered list
            dm_in_filtered = any(int(dm['index']) == dm_idx for dm in dm_list)
            if not dm_in_filtered:
                if verbose:
                    print(f"  DM{dm_idx} is specified in n_modes but not in filtered DM list. Skipping.")
                continue
                
            mode_map[dm_idx] = list(range(current_idx, current_idx + nm))
            current_idx += nm
            
        total_modes = current_idx
    elif isinstance(n_modes, dict):
        # Dict format: keys are DM indices
        current_idx = 0
        
        # Sort DM indices for consistent mode ordering
        for dm_idx in sorted(n_modes.keys()):
            nm = n_modes[dm_idx]
            
            # Skip if modes count is 0
            if nm <= 0:
                continue
                
            # Check if this DM is in our filtered list
            dm_in_filtered = any(int(dm['index']) == dm_idx for dm in dm_list)
            if not dm_in_filtered:
                if verbose:
                    print(f"  DM{dm_idx} is specified in n_modes but not in filtered DM list. Skipping.")
                continue
                
            mode_map[dm_idx] = list(range(current_idx, current_idx + nm))
            current_idx += nm
            
        total_modes = current_idx
    else:
        raise ValueError("n_modes must be an integer, list, or dictionary")
        
    if total_modes == 0:
        raise ValueError("No valid modes to combine. Check n_modes parameter and DM filtering.")
    
    # Calculate dimensions for the combined matrix
    N = total_modes
    M = len(filtered_wfs) * n_slopes
    im_full = np.zeros((N, M))
    
    if verbose:
        print(f"Creating combined matrix of shape ({N}, {M})")
        print(f"Using {len(filtered_wfs)} WFSs and {len(mode_map)} DMs")
        print("Mode map:")
        for dm_idx, modes in mode_map.items():
            print(f"  DM{dm_idx}: {len(modes)} modes at indices {modes}")
    
    # Load and combine matrices
    for i, wfs in enumerate(filtered_wfs):
        wfs_idx = int(wfs['index'])
        
        for dm in dm_list:
            dm_idx = int(dm['index'])
            
            # Skip if this DM is not in the mode map (has 0 modes or was filtered out)
            if dm_idx not in mode_map:
                if verbose:
                    print(f"  Skipping DM{dm_idx} (not in mode map)")
                continue
            
            im_filename = generate_im_filename(yaml_file, wfs_type=wfs_type, 
                                             wfs_index=wfs_idx, dm_index=dm_idx)
            im_path = os.path.join(output_im_dir, im_filename)
            
            if verbose:
                print(f"  Loading {im_filename}")
            
            # Check if file exists
            if not os.path.exists(im_path):
                if verbose:
                    print(f"  File not found: {im_path}")
                continue
            
            # Load the interaction matrix
            intmat_obj = Intmat.restore(im_path)
            
            # Get mode indices for this DM
            mode_idx = mode_map[dm_idx]
            
            # Insert into the full matrix
            slope_idx_start = i * n_slopes
            slope_idx_end = (i + 1) * n_slopes
            
            # Use the utility function to insert the matrix part
            insert_interaction_matrix_part(
                im_full, intmat_obj, mode_idx, slope_idx_start, slope_idx_end, verbose=verbose
            )
    
    # Display the combined matrix
    if display:
        plt.figure(figsize=(10, 8))
        plt.imshow(im_full, cmap='viridis')
        plt.colorbar()
        plt.title("Combined Interaction Matrix")
        plt.tight_layout()
        plt.show()
        
        if verbose:
            # Print statistics
            print(f"Combined matrix shape: {im_full.shape}")
            print(f"Matrix min: {im_full.min()}, max: {im_full.max()}, mean: {np.mean(im_full)}")
            
            # Display as pandas DataFrame for better readability
            import pandas as pd
            print("Full interaction matrix:")
            df = pd.DataFrame(im_full)
            print(df.to_string(float_format=lambda x: f"{x:.6e}"))
    
    return im_full