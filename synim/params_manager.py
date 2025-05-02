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
    
    def get_dm_params(self, dm_index):
        """
        Get DM parameters for a specific DM index.
        
        Args:
            dm_index (int): Index of the DM (1-based)
            
        Returns:
            dict: DM parameters
        """
        # Check if already loaded in cache
        if dm_index in self.dm_cache:
            return self.dm_cache[dm_index]
        
        # Find the appropriate DM based on dm_index
        selected_dm = None
        
        for dm in self.dm_list:
            if dm['index'] == str(dm_index):
                selected_dm = dm
                if self.verbose:
                    print(f"DM -- Using specified DM: {dm['name']}")
                break
        
        # If no DM found with specified index, use first DM
        if selected_dm is None and self.dm_list:
            selected_dm = self.dm_list[0]
            if self.verbose:
                print(f"DM -- Using first available DM: {selected_dm['name']}")
        
        if selected_dm is None:
            raise ValueError("No DM configuration found in the configuration file.")
        
        dm_key = selected_dm['name']
        dm_params = selected_dm['config']
        
        # Extract DM parameters
        dm_height = dm_params.get('height', 0)
        dm_rotation = dm_params.get('rotation', 0.0)
        
        # Load influence functions
        dm_array, dm_mask = load_influence_functions(self.cm, dm_params, self.pixel_pupil, 
                                                   verbose=self.verbose)
        
        if 'nmodes' in dm_params:
            nmodes = dm_params['nmodes']
            if dm_array.shape[2] > nmodes:
                if self.verbose:
                    print(f"     Trimming DM array to first {nmodes} modes")
                dm_array = dm_array[:, :, :nmodes]
        
        # Create dictionary with DM parameters
        dm_data = {
            'dm_key': dm_key,
            'dm_array': dm_array,
            'dm_mask': dm_mask,
            'dm_height': dm_height,
            'dm_rotation': dm_rotation
        }
        
        # Cache for future use
        self.dm_cache[dm_index] = dm_data
        
        return dm_data
    
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

    def compute_projection_matrices(self, output_dir=None, wfs_type=None, overwrite=False, 
                                verbose=None, display=False):
        """
        Compute and save projection matrices for all combinations of WFSs and DMs.
        Reuses cached parameters to avoid redundant loading.
        Uses ifunc from the parameters file to create the inverse basis array.
        
        Args:
            output_dir (str, optional): Output directory for saved matrices
            wfs_type (str, optional): Type of WFS ('opt', 'ngs', 'lgs', 'ref') to use
            overwrite (bool, optional): Whether to overwrite existing files
            verbose (bool, optional): Override the class's verbose setting
            display (bool, optional): Whether to display plots
            
        Returns:
            dict: Dictionary mapping WFS-DM pairs to saved projection matrix paths
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

        # Load base_inv_array from dm_inv
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

            try:
                # Load influence functions from dm_inv
                base_inv_array, inv_mask = load_influence_functions(
                    self.cm,
                    dm_inv_params,
                    self.pixel_pupil,
                    verbose=verbose_flag
                )

                if inv_array is not None:
                    # Convert 3D influence function array to 2D base_inv_array
                    # Each mode becomes a row in base_inv_array
                    height, width, n_modes = inv_array.shape
                    valid_pixels = inv_mask > 0.5
                    n_valid_pixels = np.sum(valid_pixels)

                    if verbose_flag:
                        print(f"Loaded inverse basis with shape {inv_array.shape}")
                        print(f"Mask has {n_valid_pixels} valid pixels")

            except Exception as e:
                print(f"Error loading dm_inv influence functions: {e}")

        if base_inv_array is None:
            if verbose_flag:
                print("Warning: Could not load base_inv_array from ifunc. Using default identity matrix.")

        print('base_inv_array.shape:', base_inv_array.shape)

        # Find all optical sources
        opt_sources = []
        for key, value in self.params.items():
            if key.startswith('source_opt'):
                try:
                    index = int(key.replace('source_opt', ''))
                    opt_sources.append({
                        'name': key,
                        'index': index,
                        'config': value
                    })
                except ValueError:
                    # Skip if we can't extract a valid index
                    pass

        # Sort by index
        opt_sources.sort(key=lambda x: x['index'])

        if verbose_flag:
            print(f"Found {len(opt_sources)} optical sources")
            for src in opt_sources:
                print(f"  Source: {src['name']} (index: {src['index']})")
            print(f"Computing projection matrices for these sources and {len(self.dm_list)} DM(s)")

        # Process each Source-DM combination
        for source in opt_sources:
            source_name = source['name']
            source_idx = source['index']
            source_config = source['config']

            # Get source parameters
            gs_pol_coo = source_config.get('polar_coordinates', [0.0, 0.0])
            gs_height = source_config.get('height', float('inf'))

            for dm in self.dm_list:
                dm_idx = int(dm['index'])
                dm_name = dm['name']

                if verbose_flag:
                    print(f"\nProcessing Source {source_name} (index {source_idx}) and DM {dm_name} (index {dm_idx})")

                # Generate filename for this combination
                pm_filename = f"PM_{os.path.basename(self.params_file).split('.')[0]}_opt{source_idx}_dm{dm_idx}.fits"

                # Full path for the file
                pm_path = os.path.join(output_dir, pm_filename)

                # Check if the file already exists
                if os.path.exists(pm_path) and not overwrite:
                    if verbose_flag:
                        print(f"  File {pm_filename} already exists. Skipping computation.")
                    saved_matrices[f"{source_name}_{dm_name}"] = pm_path
                    continue

                # Get DM parameters
                dm_params = self.get_dm_params(dm_idx)

                # Set default WFS parameters for optical source (no rotation/translation/magnification)
                wfs_rotation = 0.0
                wfs_translation = (0.0, 0.0)
                wfs_magnification = (1.0, 1.0)

                # Check if base_inv_array is properly loaded
                if base_inv_array is None:
                    if verbose_flag:
                        print("  No base_inv_array provided. Creating a simple identity matrix.")
                    # Create a default identity matrix for basic projection
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
                    wfs_rotation=wfs_rotation,
                    wfs_translation=wfs_translation,
                    wfs_magnification=wfs_magnification,
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

        return saved_matrices

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