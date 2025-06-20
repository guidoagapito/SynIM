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
            pup_mask = load_pupilstop(self.cm, pupilstop_params, self.pixel_pupil, self.pixel_pitch, 
                                    verbose=self.verbose)

        # If no pupilstop defined, create a default circular pupil
        if pup_mask is None:
            pupilstop = Pupilstop(
                pixel_pupil=self.pixel_pupil,
                pixel_pitch=self.pixel_pitch,
                mask_diam=1.0,
                obs_diam=0.0,
                target_device_idx=-1,
                precision=0
            )
            pup_mask = pupilstop.A
 
        return pup_mask

    def get_dm_params(self, component_idx, is_layer=False, cut_start_mode=False):
        """
        Get DM or layer parameters, loading from cache if available.

        Args:
            component_idx (int): Index of the DM or layer to load
            is_layer (bool): Whether to load a layer instead of a DM

        Returns:
            dict: DM or layer parameters
        """
        # Convert to string for dictionary lookup
        component_idx_str = str(component_idx)

        # Determine component type
        component_type = "layer" if is_layer else "dm"
        cache_key = f"{component_type}_{component_idx_str}"

        # Check if already in cache
        if cache_key in self.dm_cache:
            return self.dm_cache[cache_key]

        # Find the component in the config
        component_key = f"{component_type}{component_idx_str}"

        if component_key not in self.params:
            raise ValueError(f"{component_type.capitalize()} with index {component_idx} not found in configuration")

        component_params = self.params[component_key]

        # Load influence functions
        dm_array, dm_mask = load_influence_functions(self.cm, component_params, self.pixel_pupil, verbose=self.verbose)

        if cut_start_mode:
            if 'start_mode' in component_params:
                dm_array = dm_array[:,:,component_params['start_mode']:]

        # Extract other parameters
        dm_height = component_params.get('height', 0.0)
        dm_rotation = component_params.get('rotation', 0.0)

        # Store in cache
        self.dm_cache[cache_key] = {
            'dm_array': dm_array,
            'dm_mask': dm_mask,
            'dm_height': dm_height,
            'dm_rotation': dm_rotation,
            'dm_key': component_key,  # Add the component key here
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
        idx_valid_sa = find_subapdata(self.cm, wfs_params, wfs_key, self.params, verbose=self.verbose)
        
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

    def prepare_interaction_matrix_params(self, wfs_type=None, wfs_index=None, dm_index=None):
        """
        Prepare parameters for computing an interaction matrix.
        
        Args:
            wfs_type (str, optional): Type of WFS ('sh', 'pyr') or source type ('lgs', 'ngs', 'ref')
            wfs_index (int, optional): Index of the WFS (1-based)
            dm_index (int, optional): Index of the DM (1-based)
            
        Returns:
            dict: Parameters ready to be passed to synim.interaction_matrix
        """
        # Get DM parameters
        dm_params = self.get_dm_params(dm_index)

        # Get WFS parameters
        wfs_params = self.get_wfs_params(wfs_type, wfs_index)

        # Combine them into a single dictionary with all needed parameters
        params = {
            'pup_diam_m': self.pup_diam_m,
            'pup_mask': self.pup_mask,
            'dm_array': dm_params['dm_array'],
            'dm_mask': dm_params['dm_mask'],
            'dm_height': dm_params['dm_height'],
            'dm_rotation': dm_params['dm_rotation'],
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
            'dm_key': dm_params['dm_key'],
            'source_type': wfs_params['source_type']
        }

        return params

    def compute_interaction_matrix(self, wfs_type=None, wfs_index=None, dm_index=None, verbose=None, display=False):
        """
        Compute an interaction matrix for a specific WFS-DM combination.

        Args:
            wfs_type (str, optional): Type of WFS ('sh', 'pyr') or source type ('lgs', 'ngs', 'ref')
            wfs_index (int, optional): Index of the WFS (1-based)
            dm_index (int, optional): Index of the DM (1-based)
            verbose (bool, optional): Override the class's verbose setting
            display (bool): Whether to display plots

        Returns:
            numpy.ndarray: Computed interaction matrix
        """
        # Use class verbose setting if not overridden
        verbose_flag = self.verbose if verbose is None else verbose

        # Prepare parameters
        params = self.prepare_interaction_matrix_params(wfs_type, wfs_index, dm_index)

        if verbose_flag:
            print("Computing interaction matrix with parameters:")
            print(f"      WFS: {params['wfs_key']}, DM: {params['dm_key']}")
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

        # Transpose to be coherent with the specula convention
        im = im.transpose()

        return im

    def compute_interaction_matrices(self, output_im_dir=None, output_rec_dir=None,
                                wfs_type=None, overwrite=False, verbose=None, display=False):
        """
        Compute and save interaction matrices for all combinations of WFSs and DMs.
        Reuses cached parameters to avoid redundant loading.
        
        Args:
            output_im_dir (str, optional): Output directory for saved matrices
            output_rec_dir (str, optional): Output directory for reconstruction matrices
            wfs_type (str, optional): Type of WFS ('ngs', 'lgs', 'ref') to use
            overwrite (bool, optional): Whether to overwrite existing files
            verbose (bool, optional): Override the class's verbose setting
            display (bool, optional): Whether to display plots
            
        Returns:
            dict: Dictionary mapping WFS-DM pairs to saved interaction matrix paths
        """
        saved_matrices = {}

        # Find the SPECULA repository path for default paths
        specula_init_path = specula.__file__
        specula_package_dir = os.path.dirname(specula_init_path)
        specula_repo_path = os.path.dirname(specula_package_dir)

        # Set up directories
        if output_im_dir is None:
            output_im_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "MCAO", "im")

        if output_rec_dir is None:
            output_rec_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "MCAO", "rec")

        # Create directories if they don't exist
        os.makedirs(output_im_dir, exist_ok=True)
        os.makedirs(output_rec_dir, exist_ok=True)

        # Use verbose flag from instance if not overridden
        verbose_flag = self.verbose if verbose is None else verbose

        # Filter WFSs by type if specified
        filtered_wfs_list = self.wfs_list
        if wfs_type is not None:
            filtered_wfs_list = []
            for wfs in self.wfs_list:
                if wfs_type in wfs['name']:
                    filtered_wfs_list.append(wfs)

        if verbose_flag:
            print(f"Computing interaction matrices for {len(filtered_wfs_list)} WFS(s) and {len(self.dm_list)} DM(s)")

        # Process each WFS-DM combination using cached parameters
        for wfs in filtered_wfs_list:
            wfs_idx = int(wfs['index'])
            wfs_name = wfs['name']

            # Determine source type from WFS name
            source_type = determine_source_type(wfs_name)

            for dm in self.dm_list:
                dm_idx = int(dm['index'])
                dm_name = dm['name']

                if verbose_flag:
                    print(f"\nProcessing WFS {wfs_name} (index {wfs_idx}) and DM {dm_name} (index {dm_idx})")

                # Generate filename for this combination
                im_filename = generate_im_filename(self.params_file, wfs_type=source_type,
                                                wfs_index=wfs_idx, dm_index=dm_idx)

                # Full path for the file
                im_path = os.path.join(output_im_dir, im_filename)

                # Check if the file already exists
                if os.path.exists(im_path) and not overwrite:
                    if verbose_flag:
                        print(f"  File {im_filename} already exists. Skipping computation.")
                    saved_matrices[f"{wfs_name}_{dm_name}"] = im_path
                    continue

                # Calculate the interaction matrix using our cached parameters
                im = self.compute_interaction_matrix(
                    wfs_type=source_type,
                    wfs_index=wfs_idx,
                    dm_index=dm_idx,
                    verbose=verbose_flag,
                    display=display
                )

                if verbose_flag:
                    print(f"  Interaction matrix shape: {im.shape}")
                    print(f"  First few values of IM: {im[:5, :5]}")

                # Display the matrix if requested
                if display:
                    plt.figure(figsize=(10, 8))
                    plt.imshow(im, cmap='viridis')
                    plt.colorbar()
                    plt.title(f"Interaction Matrix: {wfs_name} - {dm_name}")
                    plt.tight_layout()
                    plt.show()

                # Create the Intmat object
                # Get parameters to include in the tag
                wfs_params = self.get_wfs_params(source_type, wfs_idx)
                wfs_info = f"{wfs_params['wfs_type']}_{wfs_params['wfs_nsubaps']}"

                # Create tag for the pupdata
                if isinstance(self.params_file, str):
                    config_name = os.path.basename(self.params_file).split('.')[0]
                else:
                    config_name = "config"

                # TODO: this must be the subapdata name
                pupdata_tag = f"{config_name}_{wfs_info}"

                # Create Intmat object and save it
                intmat_obj = Intmat(
                    im, 
                    pupdata_tag=pupdata_tag,
                    norm_factor=1.0,
                    target_device_idx=None,  # Use default device
                    precision=None    # Use default precision
                )

                # Save the interaction matrix
                intmat_obj.save(im_path)
                if verbose_flag:
                    print(f"  Interaction matrix saved as: {im_path}")

                saved_matrices[f"{wfs_name}_{dm_name}"] = im_path

        return saved_matrices

    def assemble_interaction_matrices(self, wfs_type='ngs', output_im_dir=None, save=False):
        """
        Assemble interaction matrices for a specific type of WFS into a single full interaction matrix.

        Args:
            wfs_type (str): The type of WFS to assemble matrices for ('ngs', 'lgs', 'ref')
            output_im_dir (str, optional): Directory where IM files are stored
            save (bool): Whether to save the assembled matrix to disk
            
        Returns:
            tuple: (im_full, n_slopes_per_wfs, mode_indices, dm_indices) - Assembled matrix and associated parameters
        """
        # Set up output directory
        if output_im_dir is None:
            specula_init_path = specula.__file__
            specula_package_dir = os.path.dirname(specula_init_path)
            specula_repo_path = os.path.dirname(specula_package_dir)
            output_im_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "MCAO", "im")

        # Count WFSs of the specified type in the configuration
        wfs_list = [wfs for wfs in self.wfs_list if wfs_type in wfs['name']]
        n_wfs = len(wfs_list)

        if self.verbose:
            print(f"Found {n_wfs} {wfs_type.upper()} WFSs")

        # Get the number of slopes per WFS (from idx_valid_sa)
        n_slopes_per_wfs = 0
        for wfs in wfs_list:
            wfs_params = self.get_wfs_params(wfs_type, int(wfs['index']))
            if wfs_params['idx_valid_sa'] is not None:
                # Each valid subaperture produces X and Y slopes
                n_slopes_this_wfs = len(wfs_params['idx_valid_sa']) * 2
                if n_slopes_per_wfs == 0:
                    n_slopes_per_wfs = n_slopes_this_wfs
                elif n_slopes_per_wfs != n_slopes_this_wfs:
                    print(f"Warning: Inconsistent number of slopes across WFSs")

        if self.verbose:
            print(f"Each WFS has {n_slopes_per_wfs} slopes")

        # DM indices and start modes based on config
        dm_indices = []
        dm_start_modes = []
        mode_indices = []
        total_modes = 0

        if 'modal_combination' in self.params:
            if f'modes_{wfs_type}' in self.params['modal_combination']:
                modes_config = self.params['modal_combination'][f'modes_{wfs_type}']
                for i, n_modes in enumerate(modes_config):
                    if n_modes > 0:
                        # Check for start_mode in DM config
                        if f'dm{i+1}' in self.params and 'start_mode' in self.params[f'dm{i+1}']:
                            dm_start_mode = self.params[f'dm{i+1}']['start_mode']
                        else:
                            dm_start_mode = 0

                        dm_start_modes.append(dm_start_mode)
                        dm_indices.append(i + 1)
                        mode_indices.append(list(range(dm_start_mode, dm_start_mode + n_modes)))
                        total_modes += n_modes

        # Calculate total dimensions
        n_tot_modes = total_modes  # Total number of modes
        n_tot_slopes = n_wfs * n_slopes_per_wfs  # Total number of slopes

        if self.verbose:
            print(f"Total modes: {n_tot_modes}, Total slopes: {n_tot_slopes}")
            print(f"DM indices for {wfs_type}: {dm_indices}")
            print(f"DM start modes: {dm_start_modes}")
            print(f"Mode indices: {mode_indices}")

        # Create the full interaction matrix
        im_full = np.zeros((n_tot_modes, n_tot_slopes))

        # Load and assemble the interaction matrices
        for ii in range(n_wfs):
            for jj, dm_ind in enumerate(dm_indices):
                # Get the appropriate mode indices for this DM
                mode_idx = mode_indices[jj]

                # Generate and load the interaction matrix file
                im_filename = self.generate_im_filename(wfs_type=wfs_type, wfs_index=ii+1, dm_index=dm_ind)
                im_path = os.path.join(output_im_dir, im_filename)

                if self.verbose:
                    print(f"--> Loading IM: {im_filename}")

                # Load the interaction matrix
                intmat_obj = Intmat.restore(im_path)

                if self.verbose:
                    print(f"    IM shape: {intmat_obj.intmat.shape}")

                # Fill the appropriate section of the full interaction matrix
                im_full[mode_idx, n_slopes_per_wfs*ii:n_slopes_per_wfs*(ii+1)] = intmat_obj.intmat[mode_idx, :]

        # Display summary
        if self.verbose:
            print(f"\nAssembled interaction matrix shape: {im_full.shape}")

        # Save the full interaction matrix if requested
        if save:
            output_filename = f"im_full_{wfs_type}.npy"
            np.save(os.path.join(output_im_dir, output_filename), im_full)
            if self.verbose:
                print(f"Saved full interaction matrix to {output_filename}")

        return im_full, n_slopes_per_wfs, mode_indices, dm_indices

    def compute_projection_matrices(self, output_dir=None, overwrite=False,
                                verbose=None, display=False):
        """
        Compute and save projection matrices for all combinations of optical sources and DMs/layers.
        Reuses cached parameters to avoid redundant loading.
        Uses modal_analysis or dm_inv from the parameters file as the basis.

        Args:
            output_dir (str, optional): Output directory for saved matrices
            overwrite (bool, optional): Whether to overwrite existing files
            verbose (bool, optional): Override the class's verbose setting
            display (bool, optional): Whether to display plots

        Returns:
            dict: Dictionary mapping Source-Component pairs to saved projection matrix paths
        """
        saved_matrices = {}

        # Find the SPECULA repository path for default paths
        specula_init_path = specula.__file__
        specula_package_dir = os.path.dirname(specula_init_path)
        specula_repo_path = os.path.dirname(specula_package_dir)

        # Set up directories
        if output_dir is None:
            output_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "MCAO", "pm")

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Use verbose flag from instance if not overridden
        verbose_flag = self.verbose if verbose is None else verbose

        # Load base_inv_array from dm_inv or modal_analysis
        base_inv_array = None
        if 'dm_inv' in self.params or 'modal_analysis' in self.params:
            if 'dm_inv' in self.params:
                if verbose_flag:
                    print("Loading inverse basis functions from dm_inv")
                dm_inv_params = self.params['dm_inv']
            else:
                if verbose_flag:
                    print("Loading inverse basis functions from modal_analysis")
                dm_inv_params = self.params['modal_analysis']

            # Load influence functions from dm_inv
            base_inv_array, inv_mask = load_influence_functions(
                self.cm,
                dm_inv_params,
                self.pixel_pupil,
                verbose=verbose_flag
            )

            n_valid_pixels = np.sum(inv_mask > 0.5)

            if base_inv_array is not None:
                if verbose_flag:
                    print(f"Loaded inverse basis with shape {base_inv_array.shape}")
                    print(f"Mask has {n_valid_pixels} valid pixels")

        if base_inv_array is None:
            raise ValueError("No valid base_inv_array found in the configuration file.")

        # Extract all DM and layer configurations
        dm_list = self.dm_list
        layer_list = extract_layer_list(self.params)

        if verbose_flag:
            print(f"Found {len(dm_list)} DMs and {len(layer_list)} layers")
            for dm in dm_list:
                print(f"  DM: {dm['name']} (index: {dm['index']})")
            for layer in layer_list:
                print(f"  Layer: {layer['name']} (index: {layer['index']})")

        # Find all optical sources
        opt_sources = extract_opt_list(self.params)

        if verbose_flag:
            print(f"Found {len(opt_sources)} optical sources")
            for src in opt_sources:
                print(f"  Source: {src['name']} (index: {src['index']})")
            print(f"Computing projection matrices for optical sources with DMs and layers")

        # Process each Source-DM combination
        for source in opt_sources:
            source_name = source['name']
            source_idx = source['index']
            source_config = source['config']

            # Get source parameters
            gs_pol_coo = source_config.get('polar_coordinates', [0.0, 0.0])
            gs_height = source_config.get('height', float('inf'))

            # Process DMs first
            for dm in dm_list:
                dm_idx = int(dm['index'])
                dm_name = dm['name']

                if verbose_flag:
                    print(f"\nProcessing Source {source_name} (index {source_idx}) and DM {dm_name} (index {dm_idx})")

                # Generate filename for this combination
                pm_filename = generate_pm_filename(self.params_file, opt_index=source_idx, dm_index=dm_idx)

                # Full path for the file
                pm_path = os.path.join(output_dir, pm_filename)

                # Check if the file already exists
                if os.path.exists(pm_path) and not overwrite:
                    if verbose_flag:
                        print(f"  File {pm_filename} already exists. Skipping computation.")
                    saved_matrices[f"{source_name}_{dm_name}"] = pm_path
                    continue

                # Get DM parameters
                dm_params = self.get_dm_params(dm_idx,cut_start_mode=True)

                # Set default Basis parameters (no rotation/translation/magnification)
                base_rotation = 0.0
                base_translation = (0.0, 0.0)
                base_magnification = (1.0, 1.0)

                # Check if base_inv_array is properly loaded
                if base_inv_array is None:
                    if verbose_flag:
                        print("  No base_inv_array provided. Creating a simple identity matrix.")
                    n_valid_pixels = np.sum(self.pup_mask > 0.5)
                    base_inv_array = np.eye(n_valid_pixels)

                if verbose_flag:
                    print(f"  Computing projection matrix with base_inv_array shape: {base_inv_array.shape}")
                    print(f"  Source coordinates: {gs_pol_coo}, height: {gs_height}")
                    print(f"  DM height: {dm_params['dm_height']}, rotation: {dm_params['dm_rotation']}")

                # Calculate the projection matrix
                pm = synim.projection_matrix(
                    pup_diam_m=self.pup_diam_m,
                    pup_mask=self.pup_mask,
                    dm_array=dm_params['dm_array'],
                    dm_mask=dm_params['dm_mask'],
                    base_inv_array=base_inv_array,
                    dm_height=dm_params['dm_height'],
                    dm_rotation=dm_params['dm_rotation'],
                    base_rotation=base_rotation,
                    base_translation=base_translation,
                    base_magnification=base_magnification,
                    gs_pol_coo=gs_pol_coo,
                    gs_height=gs_height,
                    verbose=verbose_flag,
                    display=display,
                    specula_convention=True
                )

                if verbose_flag:
                    print(f"  Projection matrix shape: {pm.shape}")
                    #print(f"  First few values of PM: {pm[:min(5, pm.shape[0]), :min(5, pm.shape[1])]}")

                # Display the matrix if requested
                if display:
                    plt.figure(figsize=(10, 8))
                    plt.imshow(pm, cmap='viridis')
                    plt.colorbar()
                    plt.title(f"Projection Matrix: {source_name} - {dm_name}")
                    plt.tight_layout()
                    plt.show()

                # Create tag for the pupdata
                if isinstance(self.params_file, str):
                    config_name = os.path.basename(self.params_file).split('.')[0]
                else:
                    config_name = "config"

                pupdata_tag = f"{config_name}_opt{source_idx}"

                # Create Intmat object and save it (we reuse the Intmat class for storage)
                pm_obj = Intmat(
                    pm,
                    pupdata_tag=pupdata_tag,
                    norm_factor=1.0,
                    target_device_idx=None,  # Use default device
                    precision=None    # Use default precision
                )

                # Save the projection matrix
                pm_obj.save(pm_path)
                if verbose_flag:
                    print(f"  Projection matrix saved as: {pm_path}")

                saved_matrices[f"{source_name}_{dm_name}"] = pm_path

            # Process Layers
            for layer in layer_list:
                layer_idx = int(layer['index'])
                layer_name = layer['name']

                if verbose_flag:
                    print(f"\nProcessing Source {source_name} (index {source_idx}) and Layer {layer_name} (index {layer_idx})")

                # Generate filename for this combination
                pm_filename = generate_pm_filename(self.params_file, opt_index=source_idx, layer_index=layer_idx)

                # Full path for the file
                pm_path = os.path.join(output_dir, pm_filename)

                # Check if the file already exists
                if os.path.exists(pm_path) and not overwrite:
                    if verbose_flag:
                        print(f"  File {pm_filename} already exists. Skipping computation.")
                    saved_matrices[f"{source_name}_{layer_name}"] = pm_path
                    continue

                # Get layer parameters (using the same method as for DMs since structure is similar)
                layer_params = self.get_dm_params(layer_idx, is_layer=True, cut_start_mode=True)

                # Set default Basis parameters (no rotation/translation/magnification)
                base_rotation = 0.0
                base_translation = (0.0, 0.0)
                base_magnification = (1.0, 1.0)

                # Check if base_inv_array is properly loaded
                if base_inv_array is None:
                    if verbose_flag:
                        print("  No base_inv_array provided. Creating a simple identity matrix.")
                    n_valid_pixels = np.sum(self.pup_mask > 0.5)
                    base_inv_array = np.eye(n_valid_pixels)

                if verbose_flag:
                    print(f"  Computing projection matrix with base_inv_array shape: {base_inv_array.shape}")
                    print(f"  Source coordinates: {gs_pol_coo}, height: {gs_height}")
                    print(f"  Layer height: {layer_params['dm_height']}, rotation: {layer_params['dm_rotation']}")

                # Calculate the projection matrix
                pm = synim.projection_matrix(
                    pup_diam_m=self.pup_diam_m,
                    pup_mask=self.pup_mask,
                    dm_array=layer_params['dm_array'],
                    dm_mask=layer_params['dm_mask'],
                    base_inv_array=base_inv_array,
                    dm_height=layer_params['dm_height'],
                    dm_rotation=layer_params['dm_rotation'],
                    base_rotation=base_rotation,
                    base_translation=base_translation,
                    base_magnification=base_magnification,
                    gs_pol_coo=gs_pol_coo,
                    gs_height=gs_height,
                    verbose=verbose_flag,
                    display=display,
                    specula_convention=True
                )

                if verbose_flag:
                    print(f"  Projection matrix shape: {pm.shape}")
                    print(f"  First few values of PM: {pm[:min(5, pm.shape[0]), :min(5, pm.shape[1])]}")

                # Display the matrix if requested
                if display:
                    plt.figure(figsize=(10, 8))
                    plt.imshow(pm, cmap='viridis')
                    plt.colorbar()
                    plt.title(f"Projection Matrix: {source_name} - {layer_name}")
                    plt.tight_layout()
                    plt.show()

                # Create tag for the pupdata
                if isinstance(self.params_file, str):
                    config_name = os.path.basename(self.params_file).split('.')[0]
                else:
                    config_name = "config"

                pupdata_tag = f"{config_name}_opt{source_idx}"

                # Create Intmat object and save it (we reuse the Intmat class for storage)
                pm_obj = Intmat(
                    pm,
                    pupdata_tag=pupdata_tag,
                    norm_factor=1.0,
                    target_device_idx=None,  # Use default device
                    precision=None    # Use default precision
                )

                # Save the projection matrix
                pm_obj.save(pm_path)
                if verbose_flag:
                    print(f"  Projection matrix saved as: {pm_path}")

                saved_matrices[f"{source_name}_{layer_name}"] = pm_path

        return saved_matrices

    def compute_projection_matrix(self,  regFactor=1e-8, output_dir=None, save=False):
        """
        Assemble 4D projection matrices from individual PM files and
        calculate the final projection matrix using the full DM and layer matrices.
        This function computes the projection matrix using a weighted average of the individual matrices,
        and applies a regularization term to ensure numerical stability.

        The regularization term is added to the diagonal of the pseudoinverse to prevent singularities.
        The function returns the final projection matrix.

        Args:
            regFactor (float, optional): Regularization factor for the pseudoinverse calculation
                Default is 1e-8.
            output_dir (str, optional): Directory where PM files are stored and where assembled matrices will be saved
            save (bool): Whether to save the assembled matrices to disk

        Returns:
            popt (numpy.ndarray): Final projection matrix (n_dm_modes, n_layer_modes)
            pm_full_dm (numpy.ndarray): Full DM projection matrix (n_opt_sources, n_dm_modes, n_dm_modes)
            pm_full_layer (numpy.ndarray): Full Layer projection matrix (n_opt_sources, n_layer_modes, n_layer_modes)
        """

        # Set up output directory
        if output_dir is None:
            specula_init_path = specula.__file__
            specula_package_dir = os.path.dirname(specula_init_path)
            specula_repo_path = os.path.dirname(specula_package_dir)
            output_dir = os.path.join(specula_repo_path, "main", "scao", "calib", "MCAO", "pm")

        # Extract all necessary lists
        opt_sources = extract_opt_list(self.params)
        dm_list = extract_dm_list(self.params)
        layer_list = extract_layer_list(self.params)

        if self.verbose:
            print(f"Found {len(opt_sources)} optical sources, {len(dm_list)} DMs, and {len(layer_list)} layers")

        weights_array = np.zeros(len(opt_sources))
        for i, source in enumerate(opt_sources):
            source_config = source['config']
            # Get weight (default to 1.0 if not specified)
            weight = source_config.get('weight', 1.0)
            weights_array[i] = weight

        # Build the full projection matrices for DMs and layers
        for ii, opt_source in enumerate(opt_sources):
            opt_index = opt_source['index']

            for jj, dm in enumerate(dm_list):
                dm_index = dm['index']

                pm_filename = generate_pm_filename(self.params_file, opt_index=opt_index, dm_index=dm_index)
                if pm_filename is None:
                    raise ValueError(f"Could not generate filename for opt{opt_index}, dm{dm_index}")

                pm_path = os.path.join(output_dir, pm_filename)
                if self.verbose:
                    print(f"--> Loading DM PM: {pm_filename}")

                # check if the file exists
                if not os.path.exists(pm_path):
                    raise FileNotFoundError(f"File {pm_path} does not exist")

                intmat_obj = Intmat.restore(pm_path)
                intmat_data = intmat_obj.intmat

                # Pile the arrays on the second dimension
                if jj == 0:
                    pm_full_i = intmat_data
                else:
                    pm_full_i = np.concatenate((pm_full_i, intmat_data), axis=1)

                if self.verbose:
                    print(f"    Filled array with opt{opt_index}, dm{dm_index} projection data")

            # Build a 3D array piling the arrays on the new dimension
            if ii == 0:
                pm_full_dm = pm_full_i[np.newaxis, :, :]
            else:
                pm_full_dm = np.concatenate((pm_full_dm, pm_full_i[np.newaxis, :, :]), axis=0)

        for ii, opt_source in enumerate(opt_sources):
            opt_index = opt_source['index']

            for jj, layer in enumerate(layer_list):
                layer_index = layer['index']

                pm_filename = generate_pm_filename(self.params_file, opt_index=opt_index, layer_index=layer_index)
                if pm_filename is None:
                    if self.verbose:
                        print(f"Could not generate filename for opt{opt_index}, layer{layer_index}")
                    continue

                pm_path = os.path.join(output_dir, pm_filename)
                if self.verbose:
                    print(f"--> Loading Layer PM: {pm_filename}")
                intmat_obj = Intmat.restore(pm_path)
                intmat_data = intmat_obj.intmat

                # Pile the arrays on the second dimension
                if jj == 0:
                    pm_full_i = intmat_data
                else:
                    pm_full_i = np.concatenate((pm_full_i, intmat_data), axis=0)

                if self.verbose:
                    print(f"    Filled array with opt{opt_index}, dm{dm_index} projection data")

            # Build a 3D array piling the arrays on the new dimension
            if ii == 0:
                pm_full_layer = pm_full_i[np.newaxis, :, :]
            else:
                pm_full_layer = np.concatenate((pm_full_layer, pm_full_i[np.newaxis, :, :]), axis=0)

        # Display summary information
        if self.verbose:
            print("\nFinal 3D projection matrices:")
            if pm_full_dm is not None:
                print(f"DM projection matrix shape: {pm_full_dm.shape} (n_modes, n_sources, n_dm_modes)")
            if pm_full_layer is not None:
                print(f"Layer projection matrix shape: {pm_full_layer.shape} (n_modes, n_sources, n_layer_modes)")

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

        nopt = pm_full_dm.shape[0]
        tpdm_pdm = np.zeros((pm_full_dm.shape[1], pm_full_dm.shape[1]))
        tpdm_pl = np.zeros((pm_full_dm.shape[1], pm_full_layer.shape[1]))

        total_weight = np.sum(weights_array)
        for i in range(nopt):
            pdm_i = pm_full_dm[i, :, :]      # shape: (n_dm_modes, n_pupil_modes)
            pl_i = pm_full_layer[i, :, :]    # shape: (n_layer_modes, n_pupil_modes)
            w = weights_array[i] / total_weight

            tpdm_pdm += pdm_i @ pdm_i.T * w
            tpdm_pl +=  pdm_i @ pl_i.T * w

        # Pseudoinverse with regularization (tune eps and regFactor as needed)
        eps = 1e-14
        # tpdm_pdm is square, so we can use np.linalg.pinv directly
        tpdm_pdm_inv = np.linalg.pinv(tpdm_pdm + regFactor * np.eye(tpdm_pdm.shape[0]), rcond=eps)
        p_opt = tpdm_pdm_inv @ tpdm_pl

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

    def generate_im_filename(self, wfs_type=None, wfs_index=None, dm_index=None, timestamp=False, verbose=False):
        """Generate the interaction matrix filename for a given WFS-DM combination."""
        return generate_im_filename(self.params_file, wfs_type, wfs_index, dm_index, timestamp, verbose)

    def generate_im_filenames(self, timestamp=False):
        """Generate all possible interaction matrix filenames."""
        return generate_im_filenames(self.params_file, timestamp)