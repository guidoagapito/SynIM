import os
import re
import numpy as np
import matplotlib.pyplot as plt
import synim.synim as synim

# Import all utility functions from params_common_utils
from synim.utils import *

import specula
specula.init(device_idx=-1, precision=1)

from specula.calib_manager import CalibManager
from specula.data_objects.intmat import Intmat

class ParamsManager:
    """
    Class for managing parameters needed to compute interaction matrices 
    for all combinations of DMs and WFSs without redundant loading.
    """

    def __init__(self, params_file, root_dir=None, verbose=False):
        """
        Initialize the manager and load all common parameters.
        
        Args:
            params_file (str or dict): Path to YAML/PRO configuration file or dictionary
            root_dir (str, optional): Root directory to override in params
            verbose (bool): Whether to print detailed information
        """
        # Load configuration
        self.params_file = params_file
        if isinstance(params_file, str):
            self.params = parse_params_file(params_file)
        else:
            self.params = params_file

        self.verbose = verbose

        # Set root_dir if provided
        if root_dir:
            if 'main' in self.params:
                self.params['main']['root_dir'] = root_dir
                if verbose:
                    print(f"Root directory set to: {self.params['main']['root_dir']}")

        # Initialize the CalibManager
        self.main_params = self.params['main']
        self.cm = CalibManager(self.main_params['root_dir'])

        # Extract common parameters
        self.pixel_pupil = self.main_params['pixel_pupil']
        self.pixel_pitch = self.main_params['pixel_pitch']
        self.pup_diam_m = self.pixel_pupil * self.pixel_pitch

        # Determine configuration type
        self.is_simple_config = is_simple_config(self.params)

        # Load all WFS and DM configurations
        self.wfs_list = extract_wfs_list(self.params)
        self.dm_list = extract_dm_list(self.params)

        if self.verbose:
            print(f"Found {len(self.wfs_list)} WFS(s) and {len(self.dm_list)} DM(s)")
            for wfs in self.wfs_list:
                print(f"  WFS: {wfs['name']} (type: {wfs['type']}, index: {wfs['index']})")
            for dm in self.dm_list:
                print(f"  DM: {dm['name']} (index: {dm['index']})")

        # Pre-load pupil mask
        self.pup_mask = self._load_pupil_mask()

        # Cache for DM and WFS parameters (loaded on demand)
        self.dm_cache = {}  # Key: dm_index, Value: dict with dm_array, dm_mask, etc.
        self.wfs_cache = {}  # Key: (wfs_type, wfs_index), Value: dict with wfs params

    def _load_pupil_mask(self):
        """
        Load or create the pupil mask.
        
        Returns:
            numpy.ndarray: Pupil mask array
        """
        pup_mask = None
        if 'pupilstop' in self.params:
            pupilstop_params = self.params['pupilstop']
            if self.verbose:
                print("Found 'pupilstop' in params")
        elif 'pupil_stop' in self.params:
            pupilstop_params = self.params['pupil_stop']
            if self.verbose:
                print("Found 'pupil_stop' in params")
        pup_mask = load_pupilstop(self.cm, pupilstop_params, self.pixel_pupil, self.pixel_pitch,
                                verbose=self.verbose)

        # If no pupilstop defined, create a default circular pupil
        if pup_mask is None:
            simul_params = SimulParams(pixel_pupil=self.pixel_pupil, pixel_pitch=self.pixel_pitch)
            pupilstop = Pupilstop(
                simul_params,
                mask_diam=1.0,
                obs_diam=0.0,
                target_device_idx=-1,
                precision=0
            )
            pup_mask = pupilstop.A

        print('---> valid pixels: ', np.sum(pup_mask > 0.5))

        return pup_mask

    def count_mcao_stars(self):
        """
        Count the number of LGS, NGS, reference stars, DMs, optimisation optics, science stars and layers
        in the parameter configuration, similar to count_mcao_stars of IDL.

            Returns:
        dict: Dictionary with counts.
        """
        def count_keys_with_prefix(params, prefix):
            # Count keys starting with prefix and followed by a number
            return len([k for k in params.keys() if re.match(rf"{prefix}\d+$", k)])

        params = self.params

        out = {}
        out['n_lgs'] = count_keys_with_prefix(params, 'source_lgs')
        out['n_ngs'] = count_keys_with_prefix(params, 'source_ngs')
        out['n_ref'] = count_keys_with_prefix(params, 'source_ref')

        if 'source_opt1' in params:
            out['n_opt'] = count_keys_with_prefix(params, 'source_opt')
            out['n_gs'] = out['n_lgs'] + out['n_ngs'] + out['n_opt']
        else:
            out['n_opt'] = 0
            out['n_gs'] = out['n_lgs'] + out['n_ngs']

        if 'dm1' in params:
            out['n_dm'] = count_keys_with_prefix(params, 'dm')
        else:
            out['n_dm'] = 1

        if 'science_source1' in params:
            out['n_star'] = count_keys_with_prefix(params, 'science_source')
        else:
            out['n_star'] = 0

        if 'layer1' in params:
            out['n_rec_layer'] = count_keys_with_prefix(params, 'layer')
        else:
            out['n_rec_layer'] = 0

        return out

    def get_component_params(self, component_idx, is_layer=False, cut_start_mode=False):
        """
        Get DM or layer parameters, loading from cache if available.

        Args:
            component_idx (int): Index of the DM or layer to load
            is_layer (bool): Whether to load a layer instead of a DM

        Returns:
            dict: DM or layer parameters
        """
        component_type = "layer" if is_layer else "dm"
        cache_key = f"{component_type}_{component_idx}"

        if cache_key in self.dm_cache:
            return self.dm_cache[cache_key]

        # Try with index first (dm1, dm2, layer1, layer2...)
        component_key = f"{component_type}{component_idx}"

        # If not found and it's a DM with index 1, try without index (simple SCAO case)
        if component_key not in self.params:
            if component_type == "dm" and component_idx == 1 and "dm" in self.params:
                component_key = "dm"
            else:
                raise ValueError(f"{component_type.capitalize()} {component_idx} not found")

        component_params = self.params[component_key]
        dm_array, dm_mask = load_influence_functions(
            self.cm, component_params, self.pixel_pupil, verbose=self.verbose
        )

        if cut_start_mode and 'start_mode' in component_params:
            dm_array = dm_array[:, :, component_params['start_mode']:]

        self.dm_cache[cache_key] = {
            'dm_array': dm_array,
            'dm_mask': dm_mask,
            'dm_height': component_params.get('height', 0.0),
            'dm_rotation': component_params.get('rotation', 0.0),
            'component_key': component_key
        }

        return self.dm_cache[cache_key]

    def get_wfs_params(self, wfs_type=None, wfs_index=None):
        """
        Get WFS parameters for a specific WFS.
        
        Args:
            wfs_type (str, optional): Type of WFS ('sh', 'pyr') or source type ('lgs', 'ngs', 'ref')
            wfs_index (int, optional): Index of the WFS (1-based)
            
        Returns:
            dict: WFS parameters
        """
        # Create a cache key
        cache_key = (wfs_type, wfs_index)

        # Check if already loaded in cache
        if cache_key in self.wfs_cache:
            return self.wfs_cache[cache_key]

        # WFS selection logic
        selected_wfs = None
        source_type = None

        if self.verbose:
            print("WFS -- Looking for WFS parameters...")

        # Simple SCAO configuration case
        if self.is_simple_config:
            if self.verbose:
                print("     Simple SCAO configuration detected")
            if len(self.wfs_list) > 0:
                selected_wfs = self.wfs_list[0]
                source_type = 'ngs'  # Default for simple configs
                if self.verbose:
                    print(f"     Using WFS: {selected_wfs['name']} of type {selected_wfs['type']}")
            else:
                raise ValueError("No WFS configuration found in the configuration file.")
        else:
            # Complex MCAO configuration
            if self.verbose:
                print("     Complex MCAO configuration detected")

            # Case 1: wfs_type specifies the sensor type ('sh', 'pyr')
            if wfs_type in ['sh', 'pyr']:
                if self.verbose:
                    print(f"     Looking for WFS of type: {wfs_type}")
                matching_wfs = [wfs for wfs in self.wfs_list if wfs['type'] == wfs_type]

                if wfs_index is not None:
                    # Try to find specific index
                    for wfs in matching_wfs:
                        if wfs['index'] == str(wfs_index):
                            selected_wfs = wfs
                            if self.verbose:
                                print(f"     Found WFS with specified index: {wfs['name']}")
                            break

                # If no specific index found, use the first one
                if selected_wfs is None and matching_wfs:
                    selected_wfs = matching_wfs[0]
                    if self.verbose:
                        print(f"     Using first WFS of type {wfs_type}: {selected_wfs['name']}")

            # Case 2: wfs_type specifies the source type ('lgs', 'ngs', 'ref')
            elif wfs_type in ['lgs', 'ngs', 'ref']:
                source_type = wfs_type
                if self.verbose:
                    print(f"     Looking for WFS associated with {wfs_type} source")

                # Pattern for WFS names corresponding to the source type
                pattern = f"sh_{source_type}"
                matching_wfs = [wfs for wfs in self.wfs_list if pattern in wfs['name']]

                if wfs_index is not None:
                    # Try to find specific index within the source type
                    target_name = f"{pattern}{wfs_index}"
                    for wfs in matching_wfs:
                        if wfs['name'] == target_name:
                            selected_wfs = wfs
                            if self.verbose:
                                print(f"     Found WFS with specified index: {wfs['name']}")
                            break

                # If no specific index found, use the first one
                if selected_wfs is None and matching_wfs:
                    selected_wfs = matching_wfs[0]
                    if self.verbose:
                        print(f"     Using first WFS for {wfs_type}: {selected_wfs['name']}")

            # Case 3: Only wfs_index is specified (no wfs_type)
            elif wfs_index is not None:
                if self.verbose:
                    print(f"     Looking for WFS with index: {wfs_index}")
                for wfs in self.wfs_list:
                    if wfs['index'] == str(wfs_index):
                        selected_wfs = wfs
                        if self.verbose:
                            print(f"     Found WFS with specified index: {wfs['name']}")
                        break

            # Case 4: No specific criteria, use the first available WFS
            if selected_wfs is None and self.wfs_list:
                selected_wfs = self.wfs_list[0]
                if self.verbose:
                    print(f"     Using first available WFS: {selected_wfs['name']}")

        # If no WFS found, raise error
        if selected_wfs is None:
            raise ValueError("No matching WFS configuration found in the configuration file.")

        wfs_key = selected_wfs['name']
        wfs_params = selected_wfs['config']
        wfs_type_detected = selected_wfs['type']

        # Determine source type from WFS name if not already set
        if source_type is None:
            source_type = determine_source_type(wfs_key)

        if self.verbose:
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
        idx_valid_sa = find_subapdata(
            self.cm, wfs_params, wfs_key, self.params, verbose=self.verbose
        )

        # Guide star parameters
        if source_type == 'lgs':
            # LGS is at finite height
            # Try to get height from source or use typical LGS height
            gs_height = None

            # Check if there's a specific source for this WFS and try to get height
            source_match = re.search(r'(lgs\d+)', wfs_key)
            if source_match:
                source_key = f'source_{source_match.group(1)}'
                if source_key in self.params:
                    gs_height = self.params[source_key].get('height', None)

            # If still no height, use default
            if gs_height is None:
                gs_height = 90000.0  # Default LGS height in meters
        else:
            # NGS and REF are at infinite distance
            gs_height = float('inf')

        # Get source polar coordinates
        gs_pol_coo = extract_source_coordinates(self.params, wfs_key)

        # Create dictionary with WFS parameters
        wfs_data = {
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
            'source_type': source_type
        }

        # Cache for future use
        self.wfs_cache[cache_key] = wfs_data

        return wfs_data

    def prepare_interaction_matrix_params(self, wfs_type=None, wfs_index=None,
                                        dm_index=None, layer_index=None):
        """
        Prepare parameters for computing an interaction matrix.
        
        Args:
            wfs_type (str, optional): Type of WFS ('sh', 'pyr') or source type ('lgs', 'ngs', 'ref')
            wfs_index (int, optional): Index of the WFS (1-based)
            dm_index (int, optional): Index of the DM (1-based)
            layer_index (int, optional): Index of the Layer (1-based)
            
        Returns:
            dict: Parameters ready to be passed to synim.interaction_matrix
        """
        # Check that only one of dm_index or layer_index is specified
        if dm_index is not None and layer_index is not None:
            raise ValueError("Cannot specify both dm_index and layer_index")

        if dm_index is None and layer_index is None:
            raise ValueError("Must specify either dm_index or layer_index")

        # Get DM or Layer parameters
        if dm_index is not None:
            component_params = self.get_component_params(dm_index)
            component_key = component_params['component_key']
        else:
            component_params = self.get_component_params(layer_index, is_layer=True)
            component_key = component_params['component_key']

        # Get WFS parameters
        wfs_params = self.get_wfs_params(wfs_type, wfs_index)

        # Combine them into a single dictionary with all needed parameters
        params = {
            'pup_diam_m': self.pup_diam_m,
            'pup_mask': self.pup_mask,
            'dm_array': component_params['dm_array'],
            'dm_mask': component_params['dm_mask'],
            'dm_height': component_params['dm_height'],
            'dm_rotation': component_params['dm_rotation'],
            'wfs_key': wfs_params['wfs_key'],
            'wfs_type': wfs_params['wfs_type'],
            'wfs_nsubaps': wfs_params['wfs_nsubaps'],
            'wfs_rotation': wfs_params['wfs_rotation'],
            'wfs_translation': wfs_params['wfs_translation'],
            'wfs_magnification': wfs_params['wfs_magnification'],
            'wfs_fov_arcsec': wfs_params['wfs_fov_arcsec'],
            'gs_pol_coo': wfs_params['gs_pol_coo'],
            'gs_height': wfs_params['gs_height'],
            'idx_valid_sa': wfs_params['idx_valid_sa'],
            'dm_key': component_key if dm_index is not None else None,
            'layer_key': component_key if layer_index is not None else None,
            'component_key': component_key,
            'source_type': wfs_params['source_type']
        }

        return params

    def compute_interaction_matrix(self, wfs_type=None, wfs_index=None, dm_index=None,
                                layer_index=None, verbose=None, display=False):
        """
        Compute an interaction matrix for a specific WFS-DM/Layer combination.

        Args:
            wfs_type (str, optional): Type of WFS ('sh', 'pyr') or source type ('lgs', 'ngs', 'ref')
            wfs_index (int, optional): Index of the WFS (1-based)
            dm_index (int, optional): Index of the DM (1-based)
            layer_index (int, optional): Index of the Layer (1-based)
            verbose (bool, optional): Override the class's verbose setting
            display (bool): Whether to display plots

        Returns:
            numpy.ndarray: Computed interaction matrix
        """
        # Check that only one of dm_index or layer_index is specified
        if dm_index is not None and layer_index is not None:
            raise ValueError("Cannot specify both dm_index and layer_index")

        if dm_index is None and layer_index is None:
            raise ValueError("Must specify either dm_index or layer_index")

        # Use class verbose setting if not overridden
        verbose_flag = self.verbose if verbose is None else verbose

        # Prepare parameters
        params = self.prepare_interaction_matrix_params(wfs_type, wfs_index, dm_index, layer_index)

        component_name = params['dm_key'] if dm_index is not None else params['layer_key']

        if verbose_flag:
            print("Computing interaction matrix with parameters:")
            print(f"      WFS: {params['wfs_key']}, Component: {component_name}")
            print(f"      WFS type: {params['wfs_type']}, nsubaps: {params['wfs_nsubaps']}")
            print(f"      Guide star: {params['gs_pol_coo']} at height {params['gs_height']} m")

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
            verbose=verbose_flag,
            display=display
        )

        return im

    def compute_interaction_matrices(self, output_im_dir, output_rec_dir,
                                wfs_type=None, overwrite=False, verbose=None, display=False):
        """
        Compute and save interaction matrices for all combinations of WFSs and DMs/Layers.
        Uses multi-WFS optimization when possible.
        
        Args:
            output_im_dir (str): Output directory for saved matrices
            output_rec_dir (str): Output directory for reconstruction matrices
            wfs_type (str, optional): Type of WFS ('ngs', 'lgs', 'ref') to use
            overwrite (bool, optional): Whether to overwrite existing files
            verbose (bool, optional): Override the class's verbose setting
            display (bool, optional): Whether to display plots
            
        Returns:
            dict: Dictionary mapping WFS-Component pairs to saved interaction matrix paths
        """
        saved_matrices = {}
        os.makedirs(output_im_dir, exist_ok=True)
        os.makedirs(output_rec_dir, exist_ok=True)

        verbose_flag = self.verbose if verbose is None else verbose

        # Filter WFS list by type if specified
        if wfs_type is not None:
            filtered_wfs_list = [wfs for wfs in self.wfs_list if wfs_type in wfs['name']]
        else:
            filtered_wfs_list = self.wfs_list

        # Combine DM and layer in a single components list
        components = []
        for dm in self.dm_list:
            components.append({
                'type': 'dm',
                'index': int(dm['index']),
                'name': dm['name']
            })
        for layer in extract_layer_list(self.params):
            components.append({
                'type': 'layer',
                'index': int(layer['index']),
                'name': layer['name']
            })

        if verbose_flag:
            print(f"Computing interaction matrices for {len(filtered_wfs_list)} WFS(s) "
                f"and {len(components)} component(s)")
            for wfs in filtered_wfs_list:
                print(f"  WFS: {wfs['name']} (type: {wfs['type']}, index: {wfs['index']})")
            for comp in components:
                print(f"  {comp['type'].upper()}: {comp['name']} (index: {comp['index']})")

        # ==================== PROCESS EACH COMPONENT ====================
        for component in components:
            comp_idx = component['index']
            comp_name = component['name']
            comp_type = component['type']

            if verbose_flag:
                print(f"\n{'='*60}")
                print(f"Loading {comp_type.upper()} {comp_name} (index {comp_idx})")
                print(f"{'='*60}")

            # Load component parameters once
            component_params = self.get_component_params(
                comp_idx,
                is_layer=(comp_type == 'layer')
            )

            if verbose_flag:
                print(f"  Component array shape: {component_params['dm_array'].shape}")
                print(f"  Component height: {component_params['dm_height']}")
                print(f"  Component rotation: {component_params['dm_rotation']}")

            # ========== BUILD WFS CONFIGURATIONS FOR MULTI-WFS ==========
            wfs_configs = []
            wfs_to_compute = []  # Track which WFS need computation

            for wfs in filtered_wfs_list:
                wfs_idx = int(wfs['index'])
                wfs_name = wfs['name']
                source_type = determine_source_type(wfs_name)

                # Generate filename
                im_filename = generate_im_filename(
                    self.params_file,
                    wfs_type=source_type,
                    wfs_index=wfs_idx,
                    dm_index=comp_idx if comp_type == 'dm' else None,
                    layer_index=comp_idx if comp_type == 'layer' else None
                )

                im_path = os.path.join(output_im_dir, im_filename)

                # Check if file exists
                if os.path.exists(im_path) and not overwrite:
                    if verbose_flag:
                        print(f"  File already exists: {im_filename}")
                    saved_matrices[f"{wfs_name}_{comp_name}"] = im_path
                    continue

                # Get WFS parameters
                wfs_params = self.get_wfs_params(source_type, wfs_idx)

                # Add to configurations for multi-WFS computation
                wfs_configs.append({
                    'nsubaps': wfs_params['wfs_nsubaps'],
                    'rotation': wfs_params['wfs_rotation'],
                    'translation': wfs_params['wfs_translation'],
                    'magnification': wfs_params['wfs_magnification'],
                    'fov_arcsec': wfs_params['wfs_fov_arcsec'],
                    'idx_valid_sa': wfs_params['idx_valid_sa'],
                    'gs_pol_coo': wfs_params['gs_pol_coo'],
                    'gs_height': wfs_params['gs_height'],
                    'name': wfs_name
                })

                wfs_to_compute.append({
                    'name': wfs_name,
                    'index': wfs_idx,
                    'source_type': source_type,
                    'filename': im_filename,
                    'path': im_path,
                    'params': wfs_params
                })

            # ========== COMPUTE INTERACTION MATRICES ==========
            if len(wfs_configs) > 0:
                if verbose_flag:
                    print(f"\n  Computing {len(wfs_configs)} interaction"
                          f" matrices using multi-WFS...")

                # Use multi-WFS computation
                im_dict, derivatives_info = synim.interaction_matrices_multi_wfs(
                    pup_diam_m=self.pup_diam_m,
                    pup_mask=self.pup_mask,
                    dm_array=component_params['dm_array'],
                    dm_mask=component_params['dm_mask'],
                    dm_height=component_params['dm_height'],
                    dm_rotation=component_params['dm_rotation'],
                    wfs_configs=wfs_configs,
                    verbose=verbose_flag,
                    specula_convention=True
                )

                if verbose_flag:
                    print(f"  Used {derivatives_info['workflow'].upper()} workflow")

                # Save each computed interaction matrix
                for wfs_info in wfs_to_compute:
                    wfs_name = wfs_info['name']
                    im = im_dict[wfs_name]

                    # Transpose to be coherent with SPECULA convention
                    im = im.transpose()

                    if verbose_flag:
                        print(f"  Saving {wfs_name}: {wfs_info['filename']}")
                        print(f"    IM shape: {im.shape}")

                    # Display if requested
                    if display:
                        plt.figure(figsize=(10, 8))
                        plt.imshow(im, cmap='viridis')
                        plt.colorbar()
                        plt.title(f"IM: {wfs_name} - {comp_name}")
                        plt.tight_layout()
                        plt.show()

                    # Save interaction matrix
                    config_name = (os.path.basename(self.params_file).split('.')[0]
                                if isinstance(self.params_file, str) else "config")
                    pupdata_tag = f"{config_name}_{wfs_info['params']['wfs_type']}"
                    pupdata_tag += f"_{wfs_info['params']['wfs_nsubaps']}"

                    intmat_obj = Intmat(
                        im,
                        pupdata_tag=pupdata_tag,
                        norm_factor=1.0,
                        target_device_idx=None,
                        precision=None
                    )
                    intmat_obj.save(wfs_info['path'])

                    saved_matrices[f"{wfs_name}_{comp_name}"] = wfs_info['path']

        if verbose_flag:
            print(f"\n{'='*60}")
            print(f"Completed: {len(saved_matrices)} interaction matrices computed/loaded")
            print(f"{'='*60}\n")

        return saved_matrices


    def assemble_interaction_matrices(self, wfs_type='ngs', output_im_dir=None, 
                                    component_type='dm', save=False):
        """
        Assemble interaction matrices for a specific type of WFS into
        a single full interaction matrix.

        Args:
            wfs_type (str): The type of WFS to assemble matrices for ('ngs', 'lgs', 'ref')
            output_im_dir (str, optional): Directory where IM files are stored
            component_type (str): Type of component to assemble ('dm' or 'layer')
            save (bool): Whether to save the assembled matrix to disk
            
        Returns:
            tuple: (im_full, n_slopes_per_wfs, mode_indices, component_indices) -
                Assembled matrix and associated parameters
        """
        # Set up output directory
        if output_im_dir is None:
            raise ValueError("output_im_dir must be specified.")

        # Validate component_type
        if component_type not in ['dm', 'layer']:
            raise ValueError("component_type must be either 'dm' or 'layer'")

        # Count WFSs of the specified type in the configuration
        wfs_list = [wfs for wfs in self.wfs_list if wfs_type in wfs['name']]
        n_wfs = len(wfs_list)

        if self.verbose:
            print(f"Found {n_wfs} {wfs_type.upper()} WFSs")

        # Get the number of slopes per WFS (from idx_valid_sa)
        n_tot_slopes = 0
        n_slopes_list = []
        for wfs in wfs_list:
            wfs_params = self.get_wfs_params(wfs_type, int(wfs['index']))
            if wfs_params['idx_valid_sa'] is not None:
                # Each valid subaperture produces X and Y slopes
                n_slopes_this_wfs = len(wfs_params['idx_valid_sa']) * 2
                n_slopes_list.append(n_slopes_this_wfs)
                n_tot_slopes += n_slopes_this_wfs
        n_slopes_per_wfs = n_tot_slopes // n_wfs

        if self.verbose:
            print(f"Each WFS has {n_slopes_per_wfs} slopes")

        # Get component list based on type
        if component_type == 'dm':
            component_list = self.dm_list
        else:
            component_list = extract_layer_list(self.params)

        # Component indices and start modes based on config
        component_indices = []
        component_start_modes = []
        mode_indices = []
        total_modes = 0

        # Check if modal_combination exists for this WFS type and component type
        modal_key = f'modes_{wfs_type}_{component_type}'
        has_modal_combination = ('modal_combination' in self.params and 
                                modal_key in self.params['modal_combination'])

        if has_modal_combination:
            if self.verbose:
                print(f"Using modal_combination for {component_type}s")

            modes_config = self.params['modal_combination'][modal_key]
            for i, n_modes in enumerate(modes_config):
                if n_modes > 0:
                    # Check for start_mode in component config
                    comp_key = f'{component_type}{i+1}'
                    if comp_key in self.params and 'start_mode' in self.params[comp_key]:
                        comp_start_mode = self.params[comp_key]['start_mode']
                    else:
                        comp_start_mode = 0

                    component_start_modes.append(comp_start_mode)
                    component_indices.append(i + 1)
                    mode_indices.append(list(range(comp_start_mode, comp_start_mode + n_modes)))
                    total_modes += n_modes
        else:
            # Default: use all components with all modes
            if self.verbose:
                print(f"No modal_combination found, using all {component_type}s with all modes")

            for comp in component_list:
                comp_idx = int(comp['index'])
                comp_params = self.get_component_params(
                    comp_idx, 
                    is_layer=(component_type == 'layer')
                )
                n_modes = comp_params['dm_array'].shape[2]

                # Check for start_mode
                comp_key = f'{component_type}{comp_idx}'
                if comp_key in self.params and 'start_mode' in self.params[comp_key]:
                    comp_start_mode = self.params[comp_key]['start_mode']
                else:
                    comp_start_mode = 0

                component_start_modes.append(comp_start_mode)
                component_indices.append(comp_idx)
                mode_indices.append(list(range(comp_start_mode, comp_start_mode + n_modes)))
                total_modes += n_modes

        n_tot_modes = total_modes  # Total number of modes

        if self.verbose:
            print(f"Total modes: {n_tot_modes}, Total slopes: {n_tot_slopes}")
            print(f"{component_type.upper()} indices for {wfs_type}: {component_indices}")
            print(f"{component_type.upper()} start modes: {component_start_modes}")

        # Create the full interaction matrix
        im_full = np.zeros((n_tot_modes, n_tot_slopes))

        # Load and assemble the interaction matrices
        for ii in range(n_wfs):
            for jj, comp_ind in enumerate(component_indices):
                # Get the appropriate mode indices for this component
                mode_idx = mode_indices[jj]

                # Generate and load the interaction matrix file
                im_filename = generate_im_filename(
                    self.params_file,
                    wfs_type=wfs_type,
                    wfs_index=ii+1,
                    dm_index=comp_ind if component_type == 'dm' else None,
                    layer_index=comp_ind if component_type == 'layer' else None
                )
                im_path = os.path.join(output_im_dir, im_filename)

                if self.verbose:
                    print(f"--> Loading IM: {im_filename}")

                # Load the interaction matrix
                intmat_obj = Intmat.restore(im_path)

                if self.verbose:
                    print(f"    IM shape: {intmat_obj.intmat.shape}")

                # Fill the appropriate section of the full interaction matrix
                if ii == 0:
                    idx_start = 0
                else:
                    idx_start = sum(n_slopes_list[:ii])

                # Check if we have enough modes in the loaded IM
                n_modes_available = intmat_obj.intmat.shape[0]
                n_modes_requested = len(mode_idx)

                if n_modes_available < n_modes_requested:
                    if self.verbose:
                        print(f"    Warning: IM has only {n_modes_available} modes, "
                            f"requested {n_modes_requested}. Using available modes.")
                    # Use only available modes
                    actual_mode_idx = mode_idx[:n_modes_available]
                else:
                    actual_mode_idx = mode_idx

                im_full[actual_mode_idx, idx_start:idx_start+n_slopes_list[ii]] = \
                    intmat_obj.intmat[actual_mode_idx, :]

        # Display summary
        if self.verbose:
            print(f"\nAssembled interaction matrix shape: {im_full.shape}")

        # Save the full interaction matrix if requested
        if save:
            output_filename = f"im_full_{wfs_type}_{component_type}.npy"
            np.save(os.path.join(output_im_dir, output_filename), im_full)
            if self.verbose:
                print(f"Saved full interaction matrix to {output_filename}")

        return im_full, n_slopes_per_wfs, mode_indices, component_indices


    def _load_base_inv_array(self, verbose):
        """Extract base_inv_array loading logic."""
        base_inv_array = None

        # Priority 1: modal_analysis
        if 'modal_analysis' in self.params:
            if verbose:
                print("Loading from modal_analysis")
            base_inv_array, _ = load_influence_functions(
                self.cm, self.params['modal_analysis'], self.pixel_pupil,
                verbose=verbose, is_inverse_basis=True
            )

        # Priority 2: modalrec1.tag_ifunc4proj
        elif 'modalrec1' in self.params and 'tag_ifunc4proj' in self.params['modalrec1']:
            if verbose:
                print("Loading from modalrec1.tag_ifunc4proj")
            inv_tag = f"{self.params['modalrec1']['tag_ifunc4proj']}_inv"
            dm_inv_params = {'ifunc_tag': inv_tag}
            if 'tag_m2c4proj' in self.params['modalrec1']:
                dm_inv_params['m2c_tag'] = self.params['modalrec1']['tag_m2c4proj']
            base_inv_array, _ = load_influence_functions(
                self.cm, dm_inv_params, self.pixel_pupil,
                verbose=verbose, is_inverse_basis=True
            )

        # Priority 3: modalrec.tag_ifunc4proj
        elif 'modalrec' in self.params and 'tag_ifunc4proj' in self.params['modalrec']:
            if verbose:
                print("Loading from modalrec.tag_ifunc4proj")
            inv_tag = f"{self.params['modalrec']['tag_ifunc4proj']}_inv"
            dm_inv_params = {'ifunc_tag': inv_tag}
            if 'tag_m2c4proj' in self.params['modalrec']:
                dm_inv_params['m2c_tag'] = self.params['modalrec']['tag_m2c4proj']
            base_inv_array, _ = load_influence_functions(
                self.cm, dm_inv_params, self.pixel_pupil,
                verbose=verbose, is_inverse_basis=True
            )

        if base_inv_array is None:
            raise ValueError("No valid base_inv_array found in configuration")

        return base_inv_array


    def _save_projection_matrix(self, pm, source_info, comp_name, saved_matrices, verbose):
        """Save projection matrix helper."""
        if verbose:
            print(f"    Saving {source_info['name']}: {source_info['filename']}")
            print(f"    PM shape: {pm.shape}")

        plot_debug = False
        if plot_debug:
            plt.figure(figsize=(10, 8))
            plt.imshow(pm, cmap='viridis')
            plt.colorbar()
            plt.title(f"PM: {source_info['name']} - {comp_name}")
            plt.tight_layout()
            plt.show()

        config_name = (os.path.basename(self.params_file).split('.')[0]
                    if isinstance(self.params_file, str) else "config")
        pupdata_tag = f"{config_name}_opt{source_info['index']}"

        pm_obj = Intmat(pm, pupdata_tag=pupdata_tag, norm_factor=1.0,
                        target_device_idx=None, precision=None)
        pm_obj.save(source_info['path'])
        saved_matrices[f"{source_info['name']}_{comp_name}"] = source_info['path']


    def compute_projection_matrices(self, output_dir=None, overwrite=False,
                                    verbose=None):
        """
        Compute and save projection matrices for all combinations of optical sources and DMs/layers.
        Uses multi-base optimization when possible.
        
        Args:
            output_dir (str, optional): Output directory for saved matrices
            overwrite (bool, optional): Whether to overwrite existing files
            verbose (bool, optional): Override the class's verbose setting
            display (bool, optional): Whether to display plots

        Returns:
            dict: Dictionary mapping Source-Component pairs to saved projection matrix paths
        """
        saved_matrices = {}

        # Set up directories
        if output_dir is None:
            raise ValueError("output_dir must be specified.")

        os.makedirs(output_dir, exist_ok=True)

        # Use verbose flag from instance if not overridden
        verbose_flag = self.verbose if verbose is None else verbose

        # ==================== LOAD BASE (ONCE) ====================
        base_inv_array = self._load_base_inv_array(verbose_flag)

        # ==================== GET COMPONENTS ====================
        components = []
        for dm in self.dm_list:
            components.append({'type': 'dm', 'index': int(dm['index']), 'name': dm['name']})
        for layer in extract_layer_list(self.params):
            components.append({'type': 'layer', 'index': int(layer['index']), 'name': layer['name']})

        # ==================== GET OPTICAL SOURCES ====================
        opt_sources = extract_opt_list(self.params)

        if verbose_flag:
            print(f"Computing PMs for {len(opt_sources)} sources × {len(components)} components")

        # ==================== PROCESS EACH COMPONENT ====================
        for component in components:
            comp_idx = component['index']
            comp_name = component['name']
            comp_type = component['type']

            if verbose_flag:
                print(f"\n{'='*60}")
                print(f"Processing {comp_type.upper()} {comp_name} (index {comp_idx})")
                print(f"{'='*60}")

            # Load component ONCE
            component_params = self.get_component_params(
                comp_idx, is_layer=(comp_type == 'layer'), cut_start_mode=True
            )

            # ==================== CHECK ALL SOURCES ====================
            sources_to_compute = []
            for source in opt_sources:
                source_name = source['name']
                source_idx = source['index']
                source_config = source['config']

                pm_filename = generate_pm_filename(
                    self.params_file, opt_index=source_idx,
                    dm_index=comp_idx if comp_type == 'dm' else None,
                    layer_index=comp_idx if comp_type == 'layer' else None
                )
                pm_path = os.path.join(output_dir, pm_filename)

                if os.path.exists(pm_path) and not overwrite:
                    if verbose_flag:
                        print(f"  Exists: {pm_filename}")
                    saved_matrices[f"{source_name}_{comp_name}"] = pm_path
                    continue

                sources_to_compute.append({
                    'name': source_name,
                    'index': source_idx,
                    'filename': pm_filename,
                    'path': pm_path,
                    'gs_pol_coo': source_config.get('polar_coordinates', [0.0, 0.0]),
                    'gs_height': source_config.get('height', float('inf'))
                })

            if not sources_to_compute:
                continue

            # ==================== CHECK IF ALL SOURCES SAME POSITION ====================
            # This determines if we can use multi-base optimization
            all_gs_same = all(
                s['gs_pol_coo'] == sources_to_compute[0]['gs_pol_coo'] and
                s['gs_height'] == sources_to_compute[0]['gs_height']
                for s in sources_to_compute
            )

            if all_gs_same and len(sources_to_compute) > 1:
                # ========== OPTIMIZED: All sources same position ==========
                if verbose_flag:
                    print(f"\n  All {len(sources_to_compute)} sources at same position")
                    print(f"  Using SINGLE projection_matrix call")

                # Use FIRST source's position (they're all the same)
                gs_pol_coo_ref = sources_to_compute[0]['gs_pol_coo']
                gs_height_ref = sources_to_compute[0]['gs_height']

                # *** SINGLE CALL ***
                pm = synim.projection_matrix(
                    pup_diam_m=self.pup_diam_m,
                    pup_mask=self.pup_mask,
                    dm_array=component_params['dm_array'],
                    dm_mask=component_params['dm_mask'],
                    base_inv_array=base_inv_array,  # Same for all!
                    dm_height=component_params['dm_height'],
                    dm_rotation=component_params['dm_rotation'],
                    base_rotation=0.0,
                    base_translation=(0.0, 0.0),
                    base_magnification=(1.0, 1.0),
                    gs_pol_coo=gs_pol_coo_ref,
                    gs_height=gs_height_ref,
                    verbose=verbose_flag,
                    specula_convention=True
                )

                # *** SAVE SAME PM FOR ALL SOURCES ***
                for source_info in sources_to_compute:
                    self._save_projection_matrix(
                        pm, source_info, comp_name, saved_matrices, verbose_flag
                    )

            else:
                # ========== DIFFERENT POSITIONS: Compute individually ==========
                if verbose_flag:
                    print(f"\n  Computing {len(sources_to_compute)} PMs individually")

                for source_info in sources_to_compute:
                    if verbose_flag:
                        print(f"\n  Processing {source_info['name']}:")

                    pm = synim.projection_matrix(
                        pup_diam_m=self.pup_diam_m,
                        pup_mask=self.pup_mask,
                        dm_array=component_params['dm_array'],
                        dm_mask=component_params['dm_mask'],
                        base_inv_array=base_inv_array,
                        dm_height=component_params['dm_height'],
                        dm_rotation=component_params['dm_rotation'],
                        base_rotation=0.0,
                        base_translation=(0.0, 0.0),
                        base_magnification=(1.0, 1.0),
                        gs_pol_coo=source_info['gs_pol_coo'],
                        gs_height=source_info['gs_height'],
                        verbose=verbose_flag,
                        specula_convention=True
                    )

                    self._save_projection_matrix(
                        pm, source_info, comp_name, saved_matrices, verbose_flag
                    )

        if verbose_flag:
            print(f"\n{'='*60}")
            print(f"Completed: {len(saved_matrices)} projection matrices")
            print(f"{'='*60}\n")

        return saved_matrices

    def compute_projection_matrix(self, regFactor=1e-8, output_dir=None, save=False):
        """
        Assemble 4D projection matrices from individual PM files and
        calculate the final projection matrix using the full DM and layer matrices.
        
        Optimized version: loads each DM/layer only once and iterates over optical sources.
        
        Args:
            regFactor (float, optional): Regularization factor for the pseudoinverse calculation.
                Default is 1e-8.
            output_dir (str, optional): Directory where PM files are stored and where
                assembled matrices will be saved.
            save (bool): Whether to save the assembled matrices to disk
        
        Returns:
            popt (numpy.ndarray): Final projection matrix (n_dm_modes, n_layer_modes)
            pm_full_dm (numpy.ndarray): Full DM projection matrix 
                (n_opt_sources, n_dm_modes, n_pupil_modes)
            pm_full_layer (numpy.ndarray): Full Layer projection matrix 
                (n_opt_sources, n_layer_modes, n_pupil_modes)
        """

        # Set up output directory
        if output_dir is None:
            raise ValueError("output_dir must be specified.")

        os.makedirs(output_dir, exist_ok=True)

        # Extract all necessary lists
        opt_sources = extract_opt_list(self.params)
        dm_list = extract_dm_list(self.params)
        layer_list = extract_layer_list(self.params)

        n_opt = len(opt_sources)
        n_dm = len(dm_list)
        n_layer = len(layer_list)

        if self.verbose:
            print(f"Found {n_opt} optical sources, {n_dm} DMs, and {n_layer} layers")

        # Extract weights
        weights_array = np.array([src['config'].get('weight', 1.0) for src in opt_sources])

        # Initialize output arrays (we'll determine size from first PM loaded)
        pm_full_dm = None
        pm_full_layer = None

        # ==================== PROCESS DMs ====================
        if n_dm > 0:
            if self.verbose:
                print("\n=== Processing DM Projection Matrices ===")

            for jj, dm in enumerate(dm_list):
                dm_index = dm['index']
                dm_name = dm['name']

                if self.verbose:
                    print(f"\nLoading PMs for {dm_name} (index {dm_index})...")

                # Load PMs for all optical sources for this DM
                dm_pms_list = []

                for ii, opt_source in enumerate(opt_sources):
                    opt_index = opt_source['index']

                    pm_filename = generate_pm_filename(
                        self.params_file, opt_index=opt_index, dm_index=dm_index
                    )

                    if pm_filename is None:
                        raise ValueError(f"Could not generate filename"
                                         f" for opt{opt_index}, dm{dm_index}")

                    pm_path = os.path.join(output_dir, pm_filename)

                    if not os.path.exists(pm_path):
                        raise FileNotFoundError(f"File {pm_path} does not exist")

                    if self.verbose:
                        print(f"  Loading opt{opt_index}: {pm_filename}")

                    intmat_obj = Intmat.restore(pm_path)
                    dm_pms_list.append(intmat_obj.intmat)

                # Stack all optical sources for this DM
                # dm_pms_list[i] has shape (n_dm_modes_i, n_pupil_modes)
                # We stack along axis 0 to get (n_opt, n_dm_modes_i, n_pupil_modes)
                dm_stack = np.stack(dm_pms_list, axis=0)

                if self.verbose:
                    print(f"  Stacked PM shape for {dm_name}: {dm_stack.shape}")

                # Concatenate along DM modes (axis 1)
                if pm_full_dm is None:
                    pm_full_dm = dm_stack
                else:
                    pm_full_dm = np.concatenate((pm_full_dm, dm_stack), axis=1)

                if self.verbose:
                    print(f"  Current pm_full_dm shape: {pm_full_dm.shape}")

        # ==================== PROCESS LAYERS ====================
        if n_layer > 0:
            if self.verbose:
                print("\n=== Processing Layer Projection Matrices ===")

            for jj, layer in enumerate(layer_list):
                layer_index = layer['index']
                layer_name = layer['name']

                if self.verbose:
                    print(f"\nLoading PMs for {layer_name} (index {layer_index})...")

                # Load PMs for all optical sources for this layer
                layer_pms_list = []

                for ii, opt_source in enumerate(opt_sources):
                    opt_index = opt_source['index']

                    pm_filename = generate_pm_filename(
                        self.params_file, opt_index=opt_index, layer_index=layer_index
                    )

                    if pm_filename is None:
                        if self.verbose:
                            print(f"  Warning: Could not generate filename"
                                  f" for opt{opt_index}, layer{layer_index}")
                        # Create zero matrix as placeholder
                        if layer_pms_list:
                            layer_pms_list.append(np.zeros_like(layer_pms_list[0]))
                        continue

                    pm_path = os.path.join(output_dir, pm_filename)

                    if not os.path.exists(pm_path):
                        if self.verbose:
                            print(f"  Warning: File {pm_path} does not exist, using zeros")
                        if layer_pms_list:
                            layer_pms_list.append(np.zeros_like(layer_pms_list[0]))
                        continue

                    if self.verbose:
                        print(f"  Loading opt{opt_index}: {pm_filename}")

                    intmat_obj = Intmat.restore(pm_path)
                    layer_pms_list.append(intmat_obj.intmat)

                # Stack all optical sources for this layer
                layer_stack = np.stack(layer_pms_list, axis=0)

                if self.verbose:
                    print(f"  Stacked PM shape for {layer_name}: {layer_stack.shape}")

                # Concatenate along layer modes (axis 1)
                if pm_full_layer is None:
                    pm_full_layer = layer_stack
                else:
                    pm_full_layer = np.concatenate((pm_full_layer, layer_stack), axis=1)

                if self.verbose:
                    print(f"  Current pm_full_layer shape: {pm_full_layer.shape}")

        # Display summary information
        if self.verbose:
            print("\n=== Final 3D Projection Matrices ===")
            if pm_full_dm is not None:
                print(f"DM projection matrix shape: {pm_full_dm.shape} "
                    f"(n_opt_sources, n_dm_modes, n_pupil_modes)")
            if pm_full_layer is not None:
                print(f"Layer projection matrix shape: {pm_full_layer.shape} "
                    f"(n_opt_sources, n_layer_modes, n_pupil_modes)")

        # Save the matrices if requested
        if save:
            if pm_full_dm is not None:
                dm_output_filename = "pm_full_dm.npy"
                np.save(os.path.join(output_dir, dm_output_filename), pm_full_dm)
                if self.verbose:
                    print(f"Saved DM projection matrix to {dm_output_filename}")

            if pm_full_layer is not None:
                layer_output_filename = "pm_full_layer.npy"
                np.save(os.path.join(output_dir, layer_output_filename), pm_full_layer)
                if self.verbose:
                    print(f"Saved Layer projection matrix to {layer_output_filename}")

        # ==================== COMPUTE OPTIMAL PROJECTION ====================
        if self.verbose:
            print("\n=== Computing Optimal Projection Matrix ===")

        # Weighted combination
        tpdm_pdm = np.zeros((pm_full_dm.shape[1], pm_full_dm.shape[1]))
        tpdm_pl = np.zeros((pm_full_dm.shape[1], pm_full_layer.shape[1]))

        total_weight = np.sum(weights_array)

        for i in range(n_opt):
            pdm_i = pm_full_dm[i, :, :]      # shape: (n_dm_modes, n_pupil_modes)
            pl_i = pm_full_layer[i, :, :]    # shape: (n_layer_modes, n_pupil_modes)
            w = weights_array[i] / total_weight

            tpdm_pdm += pdm_i @ pdm_i.T * w
            tpdm_pl += pdm_i @ pl_i.T * w

            if self.verbose:
                print(f"  Processed opt{opt_sources[i]['index']} with weight {w:.3f}")

        # Pseudoinverse with regularization
        if self.verbose:
            print(f"\nApplying regularization (regFactor={regFactor})")

        eps = 1e-14
        tpdm_pdm_inv = np.linalg.pinv(
            tpdm_pdm + regFactor * np.eye(tpdm_pdm.shape[0]),
            rcond=eps
        )
        p_opt = tpdm_pdm_inv @ tpdm_pl

        if self.verbose:
            print(f"Final projection matrix shape: {p_opt.shape} "
                f"(n_dm_modes, n_layer_modes)")
            print("=== Computation Complete ===\n")

        return p_opt, pm_full_dm, pm_full_layer

    def list_wfs(self):
        """Return a list of all WFS names and types."""
        return [(wfs['name'], wfs['type'], wfs['index']) for wfs in self.wfs_list]

    def list_dm(self):
        """Return a list of all DM names and indices."""
        return [(dm['name'], dm['index']) for dm in self.dm_list]

    def get_source_info(self, wfs_name):
        """Return source info for a given WFS."""
        return extract_source_info(self.params, wfs_name)

    def generate_im_filename(self, wfs_type=None, wfs_index=None,
                            dm_index=None, layer_index=None, timestamp=False, verbose=False):
        """Generate the interaction matrix filename for a given WFS-DM/Layer combination."""
        return generate_im_filename(
            self.params_file,
            wfs_type=wfs_type,
            wfs_index=wfs_index,
            dm_index=dm_index,
            layer_index=layer_index,  # Aggiungi esplicitamente questo parametro
            timestamp=timestamp,
            verbose=verbose
        )

    def compute_covariance_matrices(self, r0, L0, component_type='layer',
                                    output_dir=None, overwrite=False,
                                    full_modes=True, verbose=None):
        """
        Compute atmospheric covariance matrices for all components (DMs or layers).
        
        Args:
            r0 (float): Fried parameter in meters
            L0 (float): Outer scale in meters
            component_type (str): Type of component ('dm' or 'layer')
            output_dir (str, optional): Directory to save covariance matrices
            overwrite (bool): Whether to overwrite existing files
            full_modes (bool): If True, compute covariance for ALL modes available.
                            If False, use only modes from modal_combination.
            verbose (bool, optional): Override the class's verbose setting
            
        Returns:
            dict: Dictionary with:
                - 'C_atm_blocks': List of covariance matrices for each component
                - 'component_indices': List of component indices
                - 'n_modes_per_component': List of number of modes for each component
                - 'start_modes': List of start mode indices for each component
        """
        from specula.lib.modal_base_generator import compute_ifs_covmat

        verbose_flag = self.verbose if verbose is None else verbose

        # Validate component_type
        if component_type not in ['dm', 'layer']:
            raise ValueError("component_type must be either 'dm' or 'layer'")

        # Get component list
        if component_type == 'dm':
            component_list = self.dm_list
        else:
            component_list = extract_layer_list(self.params)

        if verbose_flag:
            print(f"\nComputing covariance matrices:")
            print(f"  r0 = {r0} m")
            print(f"  L0 = {L0} m")
            print(f"  Component type: {component_type}")
            print(f"  Full modes: {full_modes}")

        component_indices = []
        C_atm_blocks = []
        n_modes_per_component = []
        start_modes = []

        # Compute covariance for each component
        for comp in component_list:
            comp_idx = int(comp['index'])
            component_indices.append(comp_idx)

            if verbose_flag:
                print(f"\n[{len(component_indices)}/{len(component_list)}] "
                    f"Processing {component_type}{comp_idx}...")

            # Get component parameters
            comp_params = self.get_component_params(
                comp_idx,
                is_layer=(component_type == 'layer'),
                cut_start_mode=False  # Load ALL modes, we'll cut later if needed
            )

            # Check for start_mode
            comp_key = f'{component_type}{comp_idx}'
            if comp_key in self.params and 'start_mode' in self.params[comp_key]:
                start_mode = self.params[comp_key]['start_mode']
            else:
                start_mode = 0
            start_modes.append(start_mode)

            # Total modes available
            total_modes = comp_params['dm_array'].shape[2]

            if full_modes:
                # Use ALL modes
                modes_to_use = list(range(total_modes))
                mode_label = "all"
            else:
                # Use only modes after start_mode (already handled by get_component_params)
                modes_to_use = list(range(total_modes))
                mode_label = f"start{start_mode}"

            n_modes = len(modes_to_use)
            n_modes_per_component.append(total_modes)  # Store total available modes

            if verbose_flag:
                print(f"  Total modes available: {total_modes}")
                print(f"  Start mode: {start_mode}")
                print(f"  Computing covariance for: {n_modes} modes ({mode_label})")

            # Check if file already exists
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                cov_filename = (f"C_atm_{component_type}{comp_idx}_"
                            f"r0{r0:.3f}_L0{L0:.1f}_modes{mode_label}.npy")
                cov_path = os.path.join(output_dir, cov_filename)

                if os.path.exists(cov_path) and not overwrite:
                    if verbose_flag:
                        print(f"  Loading existing: {cov_filename}")
                    C_atm = np.load(cov_path)
                    C_atm_blocks.append(C_atm)
                    continue

            # Convert 3D DM array to 2D
            dm2d = dm3d_to_2d(comp_params['dm_array'], comp_params['dm_mask'])

            # Select modes
            dm2d_selected = dm2d[modes_to_use, :]

            if verbose_flag:
                print(f"  dm2d_selected shape: {dm2d_selected.shape}")
                print(f"  Computing covariance matrix...")

            # Compute covariance matrix
            C_atm = compute_ifs_covmat(
                comp_params['dm_mask'],
                self.pup_diam_m,
                dm2d_selected,
                r0,
                L0,
                oversampling=1,
                verbose=False
            )

            if verbose_flag:
                print(f"  ✓ Covariance computed: {C_atm.shape}")

            # Save if output_dir is specified
            if output_dir is not None:
                np.save(cov_path, C_atm)
                if verbose_flag:
                    print(f"  ✓ Saved to: {cov_filename}")

            C_atm_blocks.append(C_atm)

        if verbose_flag:
            print(f"\n{'='*60}")
            print(f"Completed covariance computation for {len(C_atm_blocks)} components")
            print(f"{'='*60}\n")

        return {
            'C_atm_blocks': C_atm_blocks,
            'component_indices': component_indices,
            'n_modes_per_component': n_modes_per_component,
            'start_modes': start_modes,
            'r0': r0,
            'L0': L0
        }


    def assemble_covariance_matrix(self, C_atm_blocks, component_indices, 
                                    mode_indices=None, weights=None,
                                    wfs_type=None, component_type='layer',
                                    verbose=None):
        """
        Assemble the full covariance matrix from individual blocks,
        extracting only the modes specified in mode_indices.
        
        Args:
            C_atm_blocks (list): List of covariance matrices for each component
            component_indices (list): List of component indices
            mode_indices (list, optional): List of mode index arrays for each component.
                                        If None, uses modal_combination.
            weights (list, optional): Weights for each component. If None, uses equal weights.
            wfs_type (str, optional): WFS type for modal_combination lookup
            component_type (str): Type of component ('dm' or 'layer')
            verbose (bool, optional): Override the class's verbose setting
            
        Returns:
            np.ndarray: Full covariance matrix with selected modes
        """
        verbose_flag = self.verbose if verbose is None else verbose

        # If mode_indices not provided, get from modal_combination
        if mode_indices is None:
            if wfs_type is None:
                raise ValueError("Either mode_indices or wfs_type must be provided")

            modal_key = f'modes_{wfs_type}_{component_type}'
            if ('modal_combination' not in self.params or
                modal_key not in self.params['modal_combination']):
                raise ValueError(f"modal_combination.{modal_key} not found in params")

            modes_config = self.params['modal_combination'][modal_key]
            mode_indices = []

            for i, (comp_idx, n_modes_cfg) in enumerate(zip(component_indices, modes_config)):
                if n_modes_cfg > 0:
                    # Get start_mode
                    comp_key = f'{component_type}{comp_idx}'
                    if comp_key in self.params and 'start_mode' in self.params[comp_key]:
                        start_mode = self.params[comp_key]['start_mode']
                    else:
                        start_mode = 0

                    mode_indices.append(list(range(start_mode, start_mode + n_modes_cfg)))
                else:
                    mode_indices.append([])

            if verbose_flag:
                print(f"Using modal_combination: {modal_key}")
                print(f"  Mode counts: {modes_config}")

        # Set default weights
        if weights is None:
            weights = [1.0] * len(component_indices)

        # Calculate total modes
        total_modes = sum(len(mi) for mi in mode_indices)

        if verbose_flag:
            print(f"\nAssembling covariance matrix:")
            print(f"  Components: {component_indices}")
            print(f"  Modes per component: {[len(mi) for mi in mode_indices]}")
            print(f"  Total modes: {total_modes}")
            print(f"  Weights: {weights}")

        # Initialize full covariance matrix
        C_atm_full = np.zeros((total_modes, total_modes))

        # Conversion factor (nm to rad^2 at 500nm)
        conversion_factor = (500 / 2 / np.pi) ** 2

        # Fill the blocks
        current_idx = 0
        for i, (C_atm_block, modes, weight) in enumerate(zip(C_atm_blocks, mode_indices, weights)):
            if len(modes) == 0:
                continue

            n_modes = len(modes)

            # Extract the sub-block for selected modes
            # C_atm_block has shape (n_total_modes, n_total_modes)
            # We want to extract modes[i] × modes[j]
            idx_modes = np.ix_(modes, modes)
            C_atm_sub = C_atm_block[idx_modes]

            # Place in full matrix
            idx_full = slice(current_idx, current_idx + n_modes)
            C_atm_full[idx_full, idx_full] = C_atm_sub * weight * conversion_factor

            if verbose_flag:
                print(f"  Component {i+1}: modes {modes[0]}-{modes[-1]} → "
                    f"full matrix [{current_idx}:{current_idx + n_modes}]")

            current_idx += n_modes

        if verbose_flag:
            print(f"\n  ✓ Full covariance matrix assembled: {C_atm_full.shape}")

        return C_atm_full
