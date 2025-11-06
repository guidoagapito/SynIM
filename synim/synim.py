
import numpy as np
import matplotlib.pyplot as plt
from synim.utils import (
    apply_mask,
    rebin,
    rotshiftzoom_array,
    shiftzoom_from_source_dm_params,
    apply_extrapolation,
    calculate_extrapolation_indices_coeffs,
    has_transformations
)

def compute_derivatives_with_extrapolation(data,mask=None):
    """
    Compute x and y derivatives using numpy.gradient on a 2D or 3D numpy array
    if mask is present does an extrapolation to avoid issue at the edges
    
    Parameters:
    - data: numpy 3D array
    - mask: optional, numpy 2D array, mask

    Returns:
    - dx: numpy 3D array, x derivative
    - dy: numpy 3D array, y derivative
    """

    if mask is not None:
        # Calculate indices and coefficients for extrapolation
        edge_pixels, reference_indices, coefficients = calculate_extrapolation_indices_coeffs(
            mask, debug=False, debug_pixels=None)
        for i in range(data.shape[2]):
            # Apply extrapolation
            temp = data[:,:,i].copy()
            data[:,:,i] = apply_extrapolation(
                data[:,:,i], edge_pixels, reference_indices, coefficients,
                debug=True, problem_indices=None
            )
            debug_extrapolation = False
            if i == 0 and debug_extrapolation:
                plt.figure(figsize=(8, 6))
                plt.imshow(temp, cmap='seismic', interpolation='nearest')
                plt.colorbar()
                plt.title(f'Original data slice {i}')
                plt.figure(figsize=(8, 6))
                plt.imshow(data[:,:,i], cmap='seismic', interpolation='nearest')
                plt.colorbar()
                plt.title(f'Extrapolated data slice {i}')
                plt.figure(figsize=(8, 6))
                plt.imshow(data[:,:,i] - temp, cmap='seismic', interpolation='nearest')
                plt.colorbar()
                plt.title(f'Difference after extrapolation for slice {i}')
                plt.show()

    # Compute x derivative
    dx = np.gradient(data, axis=(1), edge_order=1)

    # Compute y derivative
    dy = np.gradient(data, axis=(0), edge_order=1)

    if mask is not None:
        idx = np.ravel(np.array(np.where(mask.flatten() == 0)))
        dx_2d = dx.reshape((-1,dx.shape[2]))
        dx_2d[idx,:] = np.nan
        dy_2d = dy.reshape((-1,dy.shape[2]))
        dy_2d[idx,:] = np.nan
        dx = dx_2d.reshape(dx.shape)
        dy = dy_2d.reshape(dy.shape)

    return dx, dy

def integrate_derivatives(dx, dy):
    """
    Numerical integration of derivatives using numpy.cumsum
    along the x and y axes.

    Parameters:
        dx (ndarray): x derivative of the data.
        dy (ndarray): y derivative of the data.

    Returns:
        tuple: (integrated_x, integrated_y)
            - integrated_x: Integrated x derivative.
            - integrated_y: Integrated y derivative.
    """

    # Integrate x derivative along the x-axis
    integrated_x = np.cumsum(dx, axis=1)

    # Integrate y derivative along the y-axis
    integrated_y = np.cumsum(dy, axis=0)

    return integrated_x, integrated_y


def apply_dm_transformations_separated(pup_diam_m, pup_mask, dm_array, dm_mask,
                                       dm_height, dm_rotation,
                                       gs_pol_coo, gs_height,
                                       verbose=False, specula_convention=True):
    """
    Apply ONLY DM transformations (for separated workflow).
    Returns derivatives that need WFS transformations applied separately.
    """

    if specula_convention:
        dm_array = np.transpose(dm_array, (1, 0, 2))
        dm_mask = np.transpose(dm_mask)
        pup_mask = np.transpose(pup_mask)

    pup_diam_pix = pup_mask.shape[0]
    pixel_pitch = pup_diam_m / pup_diam_pix

    if dm_mask.shape[0] != dm_array.shape[0]:
        raise ValueError('DM and mask arrays must have the same dimensions.')

    dm_translation, dm_magnification = shiftzoom_from_source_dm_params(
        gs_pol_coo, gs_height, dm_height, pixel_pitch
    )
    output_size = (pup_diam_pix, pup_diam_pix)

    if verbose:
        print(f'DM transformations (separated):')
        print(f'  Translation: {dm_translation} pixels')
        print(f'  Rotation: {dm_rotation} deg')
        print(f'  Magnification: {dm_magnification}')

    # Apply ONLY DM transformations
    trans_dm_array = rotshiftzoom_array(
        dm_array,
        dm_translation=dm_translation,
        dm_rotation=dm_rotation,
        dm_magnification=dm_magnification,
        wfs_translation=(0, 0),
        wfs_rotation=0,
        wfs_magnification=(1, 1),
        output_size=output_size
    )

    trans_dm_mask = rotshiftzoom_array(
        dm_mask,
        dm_translation=dm_translation,
        dm_rotation=dm_rotation,
        dm_magnification=dm_magnification,
        wfs_translation=(0, 0),
        wfs_rotation=0,
        wfs_magnification=(1, 1),
        output_size=output_size
    )
    trans_dm_mask[trans_dm_mask < 0.5] = 0

    if np.max(trans_dm_mask) <= 0:
        raise ValueError('Transformed DM mask is empty.')

    trans_dm_array = apply_mask(trans_dm_array, trans_dm_mask)

    # Compute derivatives on DM-transformed array
    derivatives_x, derivatives_y = compute_derivatives_with_extrapolation(
        trans_dm_array, mask=trans_dm_mask
    )

    if verbose:
        print(f'  ✓ DM array transformed, shape: {trans_dm_array.shape}')
        print(f'  ✓ Derivatives computed')

    # Return the ORIGINAL pupil mask (not transformed), so WFS transformations can be applied later
    return trans_dm_array, trans_dm_mask, pup_mask, derivatives_x, derivatives_y


def apply_dm_transformations_combined(pup_diam_m, pup_mask, dm_array, dm_mask,
                                      dm_height, dm_rotation,
                                      wfs_rotation, wfs_translation, wfs_magnification,
                                      gs_pol_coo, gs_height,
                                      verbose=False, specula_convention=True):
    """
    Apply DM and WFS transformations COMBINED (single interpolation step).
    This avoids cumulative interpolation errors when both DM and WFS have rotations.
    """

    if specula_convention:
        dm_array = np.transpose(dm_array, (1, 0, 2))
        dm_mask = np.transpose(dm_mask)
        pup_mask = np.transpose(pup_mask)

    pup_diam_pix = pup_mask.shape[0]
    pixel_pitch = pup_diam_m / pup_diam_pix

    if dm_mask.shape[0] != dm_array.shape[0]:
        raise ValueError('DM and mask arrays must have the same dimensions.')

    dm_translation, dm_magnification = shiftzoom_from_source_dm_params(
        gs_pol_coo, gs_height, dm_height, pixel_pitch
    )
    output_size = (pup_diam_pix, pup_diam_pix)

    if verbose:
        print(f'Combined DM+WFS transformations:')
        print(f'  DM translation: {dm_translation} pixels')
        print(f'  DM rotation: {dm_rotation} deg')
        print(f'  DM magnification: {dm_magnification}')
        print(f'  WFS translation: {wfs_translation} pixels')
        print(f'  WFS rotation: {wfs_rotation} deg')
        print(f'  WFS magnification: {wfs_magnification}')

    # Apply ALL transformations in one step
    trans_dm_array = rotshiftzoom_array(
        dm_array,
        dm_translation=dm_translation,
        dm_rotation=dm_rotation,
        dm_magnification=dm_magnification,
        wfs_translation=wfs_translation,  # Include WFS
        wfs_rotation=wfs_rotation,
        wfs_magnification=wfs_magnification,
        output_size=output_size
    )

    # DM mask (only DM transformations)
    trans_dm_mask = rotshiftzoom_array(
        dm_mask,
        dm_translation=dm_translation,
        dm_rotation=dm_rotation,
        dm_magnification=dm_magnification,
        wfs_translation=(0, 0),
        wfs_rotation=0,
        wfs_magnification=(1, 1),
        output_size=output_size
    )
    trans_dm_mask[trans_dm_mask < 0.5] = 0

    # Pupil mask (only WFS transformations)
    trans_pup_mask = rotshiftzoom_array(
        pup_mask,
        dm_translation=(0, 0),
        dm_rotation=0,
        dm_magnification=(1, 1),
        wfs_translation=wfs_translation,
        wfs_rotation=wfs_rotation,
        wfs_magnification=wfs_magnification,
        output_size=output_size
    )
    trans_pup_mask[trans_pup_mask < 0.5] = 0

    if np.max(trans_dm_mask) <= 0:
        raise ValueError('Transformed DM mask is empty.')
    if np.max(trans_pup_mask) <= 0:
        raise ValueError('Transformed pupil mask is empty.')

    trans_dm_array = apply_mask(trans_dm_array, trans_dm_mask)

    # Compute derivatives on already-transformed array
    derivatives_x, derivatives_y = compute_derivatives_with_extrapolation(
        trans_dm_array, mask=trans_dm_mask
    )

    if verbose:
        print(f'  ✓ Combined transformation applied, shape: {trans_dm_array.shape}')
        print(f'  ✓ Derivatives computed')

    return trans_dm_array, trans_dm_mask, trans_pup_mask, derivatives_x, derivatives_y


def apply_wfs_transformations_separated(derivatives_x, derivatives_y,
                                        pup_mask, dm_mask,
                                        wfs_nsubaps, wfs_rotation,
                                        wfs_translation, wfs_magnification,
                                        wfs_fov_arcsec, pup_diam_m,
                                        idx_valid_sa=None, verbose=False,
                                        specula_convention=True):
    """
    Apply WFS transformations to derivatives (for separated workflow).
    """

    output_size = pup_mask.shape

    # Transform pupil mask
    trans_pup_mask = rotshiftzoom_array(
        pup_mask,
        dm_translation=(0, 0),
        dm_rotation=0,
        dm_magnification=(1, 1),
        wfs_translation=wfs_translation,
        wfs_rotation=wfs_rotation,
        wfs_magnification=wfs_magnification,
        output_size=output_size
    )
    trans_pup_mask[trans_pup_mask < 0.5] = 0

    if np.max(trans_pup_mask) <= 0:
        raise ValueError('Transformed pupil mask is empty.')

    if verbose:
        print(f'WFS transformations (separated):')
        print(f'  Translation: {wfs_translation} pixels')
        print(f'  Rotation: {wfs_rotation} deg')
        print(f'  Magnification: {wfs_magnification}')

    # Transform derivatives
    trans_der_x = rotshiftzoom_array(
        derivatives_x,
        dm_translation=(0, 0),
        dm_rotation=0,
        dm_magnification=(1, 1),
        wfs_translation=wfs_translation,
        wfs_rotation=wfs_rotation,
        wfs_magnification=wfs_magnification,
        output_size=output_size
    )

    trans_der_y = rotshiftzoom_array(
        derivatives_y,
        dm_translation=(0, 0),
        dm_rotation=0,
        dm_magnification=(1, 1),
        wfs_translation=wfs_translation,
        wfs_rotation=wfs_rotation,
        wfs_magnification=wfs_magnification,
        output_size=output_size
    )

    # Continue with rebinning and slope computation
    return _compute_slopes_from_derivatives(
        trans_der_x, trans_der_y, trans_pup_mask, dm_mask,
        wfs_nsubaps, wfs_fov_arcsec, pup_diam_m, idx_valid_sa,
        verbose, specula_convention
    )


def apply_wfs_transformations_combined(derivatives_x, derivatives_y, trans_pup_mask, dm_mask,
                                       wfs_nsubaps, wfs_fov_arcsec, pup_diam_m, idx_valid_sa=None,
                                       verbose=False, specula_convention=True):
    """
    Compute slopes from pre-transformed derivatives (for combined workflow).
    No additional transformations needed.
    """

    # Derivatives are already transformed - just compute slopes
    return _compute_slopes_from_derivatives(
        derivatives_x, derivatives_y, trans_pup_mask, dm_mask,
        wfs_nsubaps, wfs_fov_arcsec, pup_diam_m, idx_valid_sa,
        verbose, specula_convention
    )


def _compute_slopes_from_derivatives(derivatives_x, derivatives_y, pup_mask, dm_mask,
                                     wfs_nsubaps, wfs_fov_arcsec, pup_diam_m, idx_valid_sa,
                                     verbose, specula_convention):
    """
    Common function to compute slopes from derivatives.
    Used by both separated and combined workflows.
    """

    # Clean up masks
    if np.isnan(pup_mask).any():
        np.nan_to_num(pup_mask, copy=False, nan=0.0)
    if np.isnan(dm_mask).any():
        np.nan_to_num(dm_mask, copy=False, nan=0.0)

    # Rebin masks to WFS resolution
    pup_mask_sa = rebin(pup_mask, (wfs_nsubaps, wfs_nsubaps), method='sum')
    pup_mask_sa = pup_mask_sa / np.max(pup_mask_sa) if np.max(pup_mask_sa) > 0 else pup_mask_sa

    dm_mask_sa = rebin(dm_mask, (wfs_nsubaps, wfs_nsubaps), method='sum')
    if np.max(dm_mask_sa) <= 0:
        raise ValueError('DM mask is empty after rebinning.')
    dm_mask_sa = dm_mask_sa / np.max(dm_mask_sa)

    # Clean derivatives
    if np.isnan(derivatives_x).any():
        np.nan_to_num(derivatives_x, copy=False, nan=0.0)
    if np.isnan(derivatives_y).any():
        np.nan_to_num(derivatives_y, copy=False, nan=0.0)

    # Apply pupil mask
    trans_der_x = apply_mask(derivatives_x, pup_mask, fill_value=np.nan)
    trans_der_y = apply_mask(derivatives_y, pup_mask, fill_value=np.nan)

    # Rebin derivatives
    scale_factor = (trans_der_x.shape[0] / wfs_nsubaps) / \
                   np.median(rebin(pup_mask, (wfs_nsubaps, wfs_nsubaps), method='average'))

    wfs_signal_x = rebin(trans_der_x, (wfs_nsubaps, wfs_nsubaps), method='nanmean') * scale_factor
    wfs_signal_y = rebin(trans_der_y, (wfs_nsubaps, wfs_nsubaps), method='nanmean') * scale_factor

    # Combined mask
    combined_mask_sa = (dm_mask_sa > 0.0) & (pup_mask_sa > 0.0)

    # Apply mask
    wfs_signal_x = apply_mask(wfs_signal_x, combined_mask_sa, fill_value=0)
    wfs_signal_y = apply_mask(wfs_signal_y, combined_mask_sa, fill_value=0)

    # Reshape
    wfs_signal_x_2d = wfs_signal_x.reshape((-1, wfs_signal_x.shape[2]))
    wfs_signal_y_2d = wfs_signal_y.reshape((-1, wfs_signal_y.shape[2]))

    # Select valid subapertures
    if idx_valid_sa is not None:
        if specula_convention and len(idx_valid_sa.shape) > 1 and idx_valid_sa.shape[1] == 2:
            sa_2d = np.zeros((wfs_nsubaps, wfs_nsubaps))
            sa_2d[idx_valid_sa[:, 0], idx_valid_sa[:, 1]] = 1
            sa_2d = np.transpose(sa_2d)
            idx_temp = np.where(sa_2d > 0)
            idx_valid_sa_new = np.zeros_like(idx_valid_sa)
            idx_valid_sa_new[:, 0] = idx_temp[0]
            idx_valid_sa_new[:, 1] = idx_temp[1]
        else:
            idx_valid_sa_new = idx_valid_sa

        if len(idx_valid_sa_new.shape) > 1 and idx_valid_sa_new.shape[1] == 2:
            width = wfs_nsubaps
            linear_indices = idx_valid_sa_new[:, 0] * width + idx_valid_sa_new[:, 1]
            wfs_signal_x_2d = wfs_signal_x_2d[linear_indices.astype(int), :]
            wfs_signal_y_2d = wfs_signal_y_2d[linear_indices.astype(int), :]
        else:
            wfs_signal_x_2d = wfs_signal_x_2d[idx_valid_sa_new.astype(int), :]
            wfs_signal_y_2d = wfs_signal_y_2d[idx_valid_sa_new.astype(int), :]

    # Concatenate
    if specula_convention:
        im = np.concatenate((wfs_signal_y_2d, wfs_signal_x_2d))
    else:
        im = np.concatenate((wfs_signal_x_2d, wfs_signal_y_2d))

    # Convert to slope units
    pup_diam_pix = pup_mask.shape[0]
    pixel_pitch = pup_diam_m / pup_diam_pix
    coeff = 1e-9 / (pup_diam_m / wfs_nsubaps) * 206265
    coeff *= 1 / (0.5 * wfs_fov_arcsec)
    im = im * coeff

    if verbose:
        print(f'  ✓ Slopes computed, shape: {im.shape}')

    return im


def interaction_matrix(pup_diam_m, pup_mask, dm_array, dm_mask, dm_height, dm_rotation,
                       wfs_nsubaps, wfs_rotation, wfs_translation, wfs_magnification,
                       wfs_fov_arcsec, gs_pol_coo, gs_height, idx_valid_sa=None,
                       verbose=False, display=False, specula_convention=True):
    """
    Computes interaction matrix using intelligent workflow selection.
    
    Automatically chooses between:
    - Separated workflow: When transformations are only in DM OR WFS (2 interpolation steps)
    - Combined workflow: When both DM and WFS have transformations (1 interpolation step)
    """

    # Detect which transformations are present
    has_dm_transform = has_transformations(dm_rotation, (0, 0), (1, 1)) or \
                       gs_pol_coo != (0, 0) or dm_height != 0
    has_wfs_transform = has_transformations(wfs_rotation, wfs_translation, wfs_magnification)

    # Choose workflow
    use_combined = has_dm_transform and has_wfs_transform

    if verbose:
        print(f"\n{'='*60}")
        print(f"Interaction Matrix Computation")
        print(f"{'='*60}")
        print(f"DM transformations: {has_dm_transform}")
        print(f"WFS transformations: {has_wfs_transform}")
        print(f"Using {'COMBINED' if use_combined else 'SEPARATED'} workflow")
        print(f"{'='*60}\n")

    if use_combined:
        # Combined workflow: single interpolation
        trans_dm_array, trans_dm_mask, trans_pup_mask, derivatives_x, derivatives_y = \
            apply_dm_transformations_combined(
                pup_diam_m, pup_mask, dm_array, dm_mask,
                dm_height, dm_rotation,
                wfs_rotation, wfs_translation, wfs_magnification,
                gs_pol_coo, gs_height,
                verbose=verbose, specula_convention=specula_convention
            )

        im = apply_wfs_transformations_combined(
            derivatives_x, derivatives_y, trans_pup_mask, trans_dm_mask,
            wfs_nsubaps, wfs_fov_arcsec, pup_diam_m, idx_valid_sa=idx_valid_sa,
            verbose=verbose, specula_convention=specula_convention
        )
    else:
        # Separated workflow: two interpolation steps (more flexible)
        trans_dm_array, trans_dm_mask, pup_mask_conv, derivatives_x, derivatives_y = \
            apply_dm_transformations_separated(
                pup_diam_m, pup_mask, dm_array, dm_mask,
                dm_height, dm_rotation,
                gs_pol_coo, gs_height,
                verbose=verbose, specula_convention=specula_convention
            )

        im = apply_wfs_transformations_separated(
            derivatives_x, derivatives_y, pup_mask_conv, trans_dm_mask,
            wfs_nsubaps, wfs_rotation, wfs_translation, wfs_magnification,
            wfs_fov_arcsec, pup_diam_m, idx_valid_sa=idx_valid_sa,
            verbose=verbose, specula_convention=specula_convention
        )

    if display:
        idx_plot = [2, 5]
        fig, axs = plt.subplots(2, 2)
        im3 = axs[0, 0].imshow(trans_dm_array[:, :, idx_plot[0]], cmap='seismic')
        axs[0, 1].imshow(trans_dm_array[:, :, idx_plot[0]], cmap='seismic')
        axs[1, 0].imshow(trans_dm_array[:, :, idx_plot[1]], cmap='seismic')
        axs[1, 1].imshow(trans_dm_array[:, :, idx_plot[1]], cmap='seismic')
        fig.suptitle(f'DM shapes (modes {idx_plot[0]} and {idx_plot[1]})')
        fig.colorbar(im3, ax=axs.ravel().tolist(), fraction=0.02)
        plt.show()

    return im


def interaction_matrices_multi_wfs(pup_diam_m, pup_mask,
                                   dm_array, dm_mask,
                                   dm_height, dm_rotation,
                                   wfs_configs, gs_pol_coo=None,
                                   gs_height=None, verbose=False,
                                   specula_convention=True):
    """
    Computes interaction matrices for multiple WFS configurations.
    
    Each WFS can have its own guide star position (gs_pol_coo) and height (gs_height).
    
    Parameters:
    - pup_diam_m: float, pupil diameter in meters
    - pup_mask: numpy 2D array, pupil mask (n_pup x n_pup)
    - dm_array: numpy 3D array, DM modes (n x n x n_dm_modes)
    - dm_mask: numpy 2D array, DM mask (n x n)
    - dm_height: float, DM conjugation altitude
    - dm_rotation: float, DM rotation in degrees
    - wfs_configs: list of dict, each containing WFS parameters:
        {
            'nsubaps': int,
            'rotation': float (default 0.0),
            'translation': tuple (default (0.0, 0.0)),
            'magnification': tuple or float (default (1.0, 1.0)),
            'fov_arcsec': float,
            'idx_valid_sa': array or None,
            'gs_pol_coo': tuple (radius_arcsec, angle_deg) - REQUIRED if gs_pol_coo=None
            'gs_height': float - REQUIRED if gs_height=None
            'name': str (optional)
        }
    - gs_pol_coo: tuple or None (DEPRECATED - use wfs_config['gs_pol_coo'] instead)
        If provided, uses this for all WFS (backward compatibility)
    - gs_height: float or None (DEPRECATED - use wfs_config['gs_height'] instead)
        If provided, uses this for all WFS (backward compatibility)
    - verbose: bool, optional
    - specula_convention: bool, optional
    
    Returns:
    - im_dict: dict, interaction matrices keyed by WFS name or index
    - derivatives_info: dict with metadata about the computation
    """

    if verbose:
        print(f"\n{'='*60}")
        print(f"Computing interaction matrices for {len(wfs_configs)} WFS")
        print(f"{'='*60}")

    # Check if using deprecated global gs_pol_coo/gs_height
    use_global_gs = gs_pol_coo is not None and gs_height is not None

    if use_global_gs and verbose:
        print("WARNING: Using global gs_pol_coo and gs_height for all WFS (deprecated)")
        print("         Consider specifying gs_pol_coo and gs_height in each wfs_config")

    # Extract gs_pol_coo and gs_height for each WFS
    wfs_gs_info = []
    for i, wfs_config in enumerate(wfs_configs):
        if use_global_gs:
            # Backward compatibility: use global values
            wfs_gs_pol_coo = gs_pol_coo
            wfs_gs_height = gs_height
        else:
            # New method: get from wfs_config
            if 'gs_pol_coo' not in wfs_config:
                raise ValueError(f"WFS {i}: 'gs_pol_coo' must be"
                                 f" specified in wfs_config when gs_pol_coo=None")
            if 'gs_height' not in wfs_config:
                raise ValueError(f"WFS {i}: 'gs_height' must be"
                                 f"specified in wfs_config when gs_height=None")

            wfs_gs_pol_coo = wfs_config['gs_pol_coo']
            wfs_gs_height = wfs_config['gs_height']

        wfs_gs_info.append((wfs_gs_pol_coo, wfs_gs_height))

    # Check if all WFS see DM from the same direction
    all_gs_same = all(gs_info == wfs_gs_info[0] for gs_info in wfs_gs_info)

    # Detect transformations
    has_dm_transform_list = []
    wfs_transforms = []

    for gs_pol_coo_wfs, gs_height_wfs in wfs_gs_info:
        has_dm = has_transformations(dm_rotation, (0, 0), (1, 1)) or \
                 gs_pol_coo_wfs != (0, 0) or gs_height_wfs != 0
        has_dm_transform_list.append(has_dm)

    for config in wfs_configs:
        wfs_rot = config.get('rotation', 0.0)
        wfs_trans = config.get('translation', (0.0, 0.0))
        wfs_mag = config.get('magnification', (1.0, 1.0))
        wfs_transforms.append((wfs_rot, wfs_trans, wfs_mag))

    # Check if all WFS transforms are identical
    all_wfs_same = all(t == wfs_transforms[0] for t in wfs_transforms)

    # Decide workflow:
    # SEPARATED: if all WFS see DM from same direction AND have same WFS transforms
    # COMBINED: otherwise (each WFS computed independently)
    use_separated = all_gs_same and all_wfs_same

    if verbose:
        print(f"All WFS see DM from same direction: {all_gs_same}")
        print(f"All WFS have same transforms: {all_wfs_same}")
        print(f"Using {'SEPARATED' if use_separated else 'COMBINED'} workflow")
        print(f"{'='*60}\n")

    im_dict = {}
    derivatives_info = {
        'workflow': 'separated' if use_separated else 'combined',
        'all_gs_same': all_gs_same,
        'all_wfs_same': all_wfs_same
    }

    if use_separated:
        # SEPARATED WORKFLOW: Compute DM transformations once
        if verbose:
            print("[Step 1/2] Computing DM transformations and derivatives...")

        # Use first WFS's gs_pol_coo and gs_height (they're all the same)
        gs_pol_coo_ref, gs_height_ref = wfs_gs_info[0]

        trans_dm_array, trans_dm_mask, pup_mask_conv, derivatives_x, derivatives_y = \
            apply_dm_transformations_separated(
                pup_diam_m, pup_mask, dm_array, dm_mask,
                dm_height, dm_rotation,
                gs_pol_coo_ref, gs_height_ref,
                verbose=verbose, specula_convention=specula_convention
            )

        if verbose:
            print(f"  ✓ DM transformed: {trans_dm_array.shape}")
            print(f"  ✓ Derivatives computed: {derivatives_x.shape}")
            print(f"\n[Step 2/2] Computing slopes for each WFS...")

        # Store derivatives for potential reuse
        derivatives_info['derivatives_x'] = derivatives_x
        derivatives_info['derivatives_y'] = derivatives_y
        derivatives_info['dm_array'] = trans_dm_array
        derivatives_info['dm_mask'] = trans_dm_mask

        # Apply WFS transformations for each WFS
        for i, wfs_config in enumerate(wfs_configs):
            wfs_name = wfs_config.get('name', f'wfs_{i}')
            wfs_nsubaps = wfs_config['nsubaps']
            wfs_rotation = wfs_config.get('rotation', 0.0)
            wfs_translation = wfs_config.get('translation', (0.0, 0.0))
            wfs_magnification = wfs_config.get('magnification', (1.0, 1.0))
            wfs_fov_arcsec = wfs_config['fov_arcsec']
            idx_valid_sa = wfs_config.get('idx_valid_sa', None)

            if verbose:
                print(f"\n  Processing {wfs_name}:")
                print(f"    Subapertures: {wfs_nsubaps}x{wfs_nsubaps}")
                print(f"    FOV: {wfs_fov_arcsec}''")

            im = apply_wfs_transformations_separated(
                derivatives_x, derivatives_y, pup_mask_conv, trans_dm_mask,
                wfs_nsubaps, wfs_rotation, wfs_translation, wfs_magnification,
                wfs_fov_arcsec, pup_diam_m, idx_valid_sa=idx_valid_sa,
                verbose=verbose, specula_convention=specula_convention
            )

            im_dict[wfs_name] = im

            if verbose:
                print(f"    ✓ IM shape: {im.shape}")

    else:
        # COMBINED WORKFLOW: Compute each WFS independently
        if verbose:
            print("Computing each WFS independently with combined DM+WFS transformations...")

        for i, wfs_config in enumerate(wfs_configs):
            wfs_name = wfs_config.get('name', f'wfs_{i}')
            wfs_nsubaps = wfs_config['nsubaps']
            wfs_rotation = wfs_config.get('rotation', 0.0)
            wfs_translation = wfs_config.get('translation', (0.0, 0.0))
            wfs_magnification = wfs_config.get('magnification', (1.0, 1.0))
            wfs_fov_arcsec = wfs_config['fov_arcsec']
            idx_valid_sa = wfs_config.get('idx_valid_sa', None)

            # Get WFS-specific gs_pol_coo and gs_height
            gs_pol_coo_wfs, gs_height_wfs = wfs_gs_info[i]

            if verbose:
                print(f"\n  [{i+1}/{len(wfs_configs)}] Processing {wfs_name}:")
                print(f"    Subapertures: {wfs_nsubaps}x{wfs_nsubaps}")
                print(f"    FOV: {wfs_fov_arcsec}''")
                print(f"    GS position: {gs_pol_coo_wfs}")
                print(f"    GS height: {gs_height_wfs} m")

            # Use interaction_matrix which already handles workflow selection
            im = interaction_matrix(
                pup_diam_m, pup_mask, dm_array, dm_mask,
                dm_height, dm_rotation,
                wfs_nsubaps, wfs_rotation, wfs_translation, wfs_magnification,
                wfs_fov_arcsec, gs_pol_coo_wfs, gs_height_wfs,
                idx_valid_sa=idx_valid_sa,
                verbose=verbose, display=False, specula_convention=specula_convention
            )

            im_dict[wfs_name] = im

            if verbose:
                print(f"    ✓ IM shape: {im.shape}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Completed {len(im_dict)} interaction matrices")
        print(f"Workflow: {derivatives_info['workflow'].upper()}")
        print(f"{'='*60}\n")

    return im_dict, derivatives_info
