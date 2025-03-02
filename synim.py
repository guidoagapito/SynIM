import numpy as np
from scipy.special import eval_legendre
from scipy.linalg import cholesky, svd
from scipy.interpolate import interp2d
from scipy.interpolate import griddata
from scipy.ndimage import rotate, shift, zoom
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform

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

def compute_derivatives_with_extrapolation(data,mask=None):
    # Compute x and y derivatives using numpy.gradient on a 2D or 3D numpy array
    # if mask is present does an extrapolation to avoid issue at the edges

    if mask is not None:
        data = extrapolate_edge_pixel(data, mask)
        print('Using extrapolation to compute derivatives.')
          
    # Compute x derivative
    dx = np.gradient(data, axis=(1), edge_order=1)
    
    # Compute y derivative
    dy = np.gradient(data, axis=(0), edge_order=1)
    
    if mask is not None:
        idx = np.ravel(np.array(np.where(mask.flatten() == 0)))
        dx2D = dx.reshape((-1,dx.shape[2]))
        dx2D[idx,:] = np.nan
        dy2D = dy.reshape((-1,dy.shape[2]))
        dy2D[idx,:] = np.nan
        dx = dx2D.reshape(dx.shape)
        dy = dy2D.reshape(dy.shape)
        
    return dx, dy

def integrate_derivatives(dx, dy):
    # Numerical integration of derivatives using numpy.cumsum

    # Integrate x derivative along the x-axis
    integrated_x = np.cumsum(dx, axis=1)

    # Integrate y derivative along the y-axis
    integrated_y = np.cumsum(dy, axis=0)

    return integrated_x, integrated_y

def extrapolate_edge_pixel(data, mask):
    """
    Extrapolate the edge pixels of a 2D array using linear interpolation.

    Parameters:
    - data: numpy 2D array, input data
    - mask: numpy 2D array, mask indicating valid data points (1 for valid, 0 for invalid)

    Returns:
    - extrapolated_data: numpy 2D array, data with extrapolated edge pixels
    """
    # Get the coordinates of valid and invalid points
    valid_points = np.argwhere(mask)
    invalid_points = np.argwhere(mask == 0)

    # Get the values at the valid points
    valid_values = data[mask == 1]

    # Perform linear interpolation
    extrapolated_values = griddata(valid_points, valid_values, invalid_points, method='linear')

    # Create a copy of the data to avoid modifying the original
    extrapolated_data = data.copy()

    # Fill in the extrapolated values
    for (i, j), value in zip(invalid_points, extrapolated_values):
        extrapolated_data[i, j] = value

    return extrapolated_data

def shiftzoom_from_source_dm_params(source_pol_coo, source_height, dm_height, pixel_pitch):
    arcsec2rad = np.pi/180/3600
    
    mag_factor = source_height/(source_height-dm_height)
    source_rec_coo_asec = polar_to_xy(source_pol_coo[0],source_pol_coo[1]*np.pi/180)
    source_rec_coo_m = source_rec_coo_asec*dm_height*arcsec2rad
    source_rec_coo_pix = source_rec_coo_m / pixel_pitch

    shift = tuple(source_rec_coo_pix)
    zoom = (mag_factor, mag_factor)
    
    return shift, zoom

def rotshiftzoom_array_noaffine(input_array, dm_translation=(0.0, 0.0),  dm_rotation=0.0,   dm_magnification=(1.0, 1.0),
                                    wfs_translation=(0.0, 0.0), wfs_rotation=0.0, wfs_magnification=(1.0, 1.0), output_size=None):
    # This function applies magnification, rotation, shift and resize of a 2D or 3D numpy array
    
    if np.isnan(input_array).any():
        np.nan_to_num(input_array, copy=False, nan=0.0, posinf=None, neginf=None)
    
    # Check if phase is 2D or 3D
    if len(input_array.shape) == 3:
        dm_translation_ = dm_translation + (0,)
        dm_magnification_ = dm_magnification + (1,)
        wfs_translation_ = wfs_translation + (0,)
        wfs_magnification_ = wfs_magnification + (1,)
    else:
        dm_translation_ = dm_translation
        dm_magnification_ = dm_magnification
        wfs_translation_ = wfs_translation
        wfs_magnification_ = wfs_magnification
        
    # resize
    if output_size == None:
        output_size = input_array.shape

    # (1) DM magnification
    if all(element == 1 for element in dm_magnification_):
        array_mag = input_array
    else:
        array_mag = zoom(input_array, dm_magnification_)
    
    # (2) DM rotation
    if dm_rotation == 0:
        array_rot = array_mag
    else:
        array_rot = rotate(array_mag, dm_rotation, axes=(1, 0), reshape=False)

    # (3) DM translation
    if all(element == 0 for element in dm_translation_):
        array_shi = array_rot
    else:
        array_shi = shift(array_rot, dm_translation_)

    # (4) WFS rotation
    if wfs_rotation == 0:
        array_rot = array_shi
    else:
        array_rot = rotate(array_shi, wfs_rotation, axes=(1, 0), reshape=False)

    # (5) WFS translation
    if all(element == 0 for element in wfs_translation_):
        array_shi = array_rot
    else:
        array_shi = shift(array_rot, wfs_translation_)

    # (6) WFS magnification
    if all(element == 1 for element in wfs_magnification_):
        array_mag = array_shi
    else:
        array_mag = zoom(array_shi, wfs_magnification_)

    if (array_mag.shape[0] > output_size[0]) | (array_mag.shape[1] > output_size[1]):
        # smaller output size
        if len(input_array.shape) == 3:
            output = array_mag[int(0.5*(array_mag.shape[0]-output_size[0])):int(0.5*(array_mag.shape[0]+output_size[0])), \
                               int(0.5*(array_mag.shape[1]-output_size[1])):int(0.5*(array_mag.shape[1]+output_size[1])),:]
        else:
            output = array_mag[int(0.5*(array_mag.shape[0]-output_size[0])):int(0.5*(array_mag.shape[0]+output_size[0])), \
                               int(0.5*(array_mag.shape[1]-output_size[1])):int(0.5*(array_mag.shape[1]+output_size[1]))]
    elif (array_mag.shape[0] < output_size[0]) | (array_mag.shape[1] < output_size[1]):
        # bigger output size
        if len(input_array.shape) == 3:
            output = np.zeros(output_size+(input_array.shape[2],))
            output[int(0.5*(output_size[0]-array_mag.shape[0])):int(0.5*(output_size[0]+array_mag.shape[0])), \
                   int(0.5*(output_size[1]-array_mag.shape[1])):int(0.5*(output_size[1]+array_mag.shape[1])),:] = array_mag
        else:
            output[int(0.5*(output_size[0]-array_mag.shape[0])):int(0.5*(output_size[0]+array_mag.shape[0])), \
                   int(0.5*(output_size[1]-array_mag.shape[1])):int(0.5*(output_size[1]+array_mag.shape[1]))] = array_mag
    else:
        output = array_mag

    return output

def rotshiftzoom_array(input_array, dm_translation=(0.0, 0.0), dm_rotation=0.0, dm_magnification=(1.0, 1.0),
                              wfs_translation=(0.0, 0.0), wfs_rotation=0.0, wfs_magnification=(1.0, 1.0), output_size=None):
    """
    This function applies magnification, rotation, shift and resize of a 2D or 3D numpy array using affine transformation.
    Rotation is applied in the same direction as the first function.
    """
    import numpy as np
    from scipy.ndimage import affine_transform
    
    if np.isnan(input_array).any():
        input_array = np.nan_to_num(input_array, copy=True, nan=0.0, posinf=None, neginf=None)
    
    # Check if array is 2D or 3D
    is_3d = len(input_array.shape) == 3
    
    # resize
    if output_size is None:
        output_size = input_array.shape[:2]  # Only take the first two dimensions
    
    # Center of the input array
    center = np.array(input_array.shape[:2]) / 2.0
    
    # Convert rotations to radians
    # Note: Inverting the sign of rotation to match the first function's direction
    dm_rot_rad = np.deg2rad(-dm_rotation)  # Negative sign to reverse direction
    wfs_rot_rad = np.deg2rad(-wfs_rotation)  # Negative sign to reverse direction
    
    # Initialize the output array
    if is_3d:
        output = np.zeros((output_size[0], output_size[1], input_array.shape[2]), dtype=input_array.dtype)
    else:
        output = np.zeros(output_size, dtype=input_array.dtype)
    
    # Create the transformation matrices
    # For DM transformation
    dm_scale_matrix = np.array([[1.0/dm_magnification[0], 0], [0, 1.0/dm_magnification[1]]])
    dm_rot_matrix = np.array([[np.cos(dm_rot_rad), -np.sin(dm_rot_rad)], [np.sin(dm_rot_rad), np.cos(dm_rot_rad)]])
    dm_matrix = np.dot(dm_rot_matrix, dm_scale_matrix)
    
    # For WFS transformation
    wfs_scale_matrix = np.array([[1.0/wfs_magnification[0], 0], [0, 1.0/wfs_magnification[1]]])
    wfs_rot_matrix = np.array([[np.cos(wfs_rot_rad), -np.sin(wfs_rot_rad)], [np.sin(wfs_rot_rad), np.cos(wfs_rot_rad)]])
    wfs_matrix = np.dot(wfs_rot_matrix, wfs_scale_matrix)
    
    # Combine transformations (first DM, then WFS)
    combined_matrix = np.dot(wfs_matrix, dm_matrix)
    
    # Calculate offset
    output_center = np.array(output_size) / 2.0
    offset = center - np.dot(combined_matrix, output_center) - np.dot(dm_matrix, dm_translation) - wfs_translation
    
    # Apply transformation
    if is_3d:
        # For 3D arrays, apply transformation to each slice
        for i in range(input_array.shape[2]):
            output[:, :, i] = affine_transform(
                input_array[:, :, i],
                combined_matrix,
                offset=offset,
                output_shape=output_size,
                order=1
            )
    else:
        # For 2D arrays
        output = affine_transform(
            input_array,
            combined_matrix,
            offset=offset,
            output_shape=output_size,
            order=1
        )
    
    return output

def interaction_matrix(pup_diam_m,pup_mask,dm_array,dm_mask,dm_height,dm_rotation,wfs_nsubaps,wfs_rotation,wfs_translation,wfs_magnification,
                       gs_pol_coo,gs_height,idx_valid_sa=None,verbose=False,display=False):
    """
    Computes a single interaction matrix.
    From Guido Agapito.

    Parameters:
    - pup_diam_m: float, size in m of the side of the pupil
    - pup_mask: numpy 2D array, mask
    - dm_array: numpy 3D array, Deformable Mirror 2D shapes
    - dm_mask: numpy 2D array, mask
    - dm_height: float, conjugation altitude of the Deformable Mirror
    - dm_rotation: float, rotation in deg of the Deformable Mirror with respect to the pupil
    - wfs_nsubaps: int, number of sub-aperture of the wavefront sensor
    - wfs_rotation
    - wfs_translation
    - wfs_magnification
    - gs_pol_coo: tuple, polar coordinates of the gudie star radius in arcsec and angle in deg
    - gs_height: float, altitude of the guide star
    - idx_valid_sa: numpy 1D array, indices of the valid sub-apertures
    - verbose, optional
    - display, optional

    Returns:
    - im: numpy 2D array, set of signals
    """

    pup_diam_pix = pup_mask.shape[0]
    pixel_pitch = pup_diam_m/pup_diam_pix
    dm_diam_pix = dm_mask.shape[0]
    if dm_mask.shape[0] != dm_array.shape[0]:
        raise ValueError('Error in input data, the dm and mask array must have the same dimensions.')

    pixel_pitch = pup_diam_m / pup_diam_pix

    dm_translation, dm_magnification = shiftzoom_from_source_dm_params(gs_pol_coo, gs_height, dm_height, pixel_pitch)
    output_size = (pup_diam_pix,pup_diam_pix)

    #Extraction of patch seen by GS and application of DM rotation
    trans_dm_array = rotshiftzoom_array(dm_array, dm_translation=dm_translation,   dm_rotation=dm_rotation,   dm_magnification=dm_magnification,
                                                  wfs_translation=wfs_translation, wfs_rotation=wfs_rotation, wfs_magnification=wfs_magnification,
                                                  output_size=output_size)
    trans_dm_mask  = rotshiftzoom_array(dm_mask,  dm_translation=dm_translation,   dm_rotation=dm_rotation,   dm_magnification=dm_magnification,
                                                  wfs_translation=(0,0), wfs_rotation=0, wfs_magnification=(1,1),
                                                  output_size=output_size)
    trans_dm_mask[trans_dm_mask<0.5] = 0

    if verbose:
        print('Rotation, shift and zoom done.')

    # apply mask
    trans_dm_array = apply_mask(trans_dm_array,trans_dm_mask)

    if verbose:
        print('Mask applied.')

    # Derivative od DM modes shape
    der_dx, der_dy = compute_derivatives_with_extrapolation(trans_dm_array,mask=trans_dm_mask)

    if verbose:
        print('Derivatives done.')

    # apply transformation to the mask
    if wfs_rotation == -1:
        pup_mask_rot = pup_mask
    else:
        pup_mask_rot = rotate(pup_mask, wfs_rotation, axes=(0, 0), reshape=False)
   
    if all(element == 0 for element in wfs_magnification):
        pup_mask_mag2 = pup_mask_rot
    else:
        pup_mask_mag = zoom(pup_mask_rot, np.array(wfs_magnification))
        if (pup_mask_mag.shape[0] > pup_mask.shape[0]) | (pup_mask_mag.shape[1] > pup_mask.shape[1]): # smaller output size
            pup_mask_mag2 = pup_mask_mag[int(0.5*(pup_mask_mag.shape[0]-pup_mask.shape[0])):int(0.5*(pup_mask_mag.shape[0]+pup_mask.shape[0])), \
                                    int(0.5*(pup_mask_mag.shape[1]-pup_mask.shape[1])):int(0.5*(pup_mask_mag.shape[1]+pup_mask.shape[1]))]
        elif (pup_mask_mag.shape[0] < pup_mask.shape[0]) | (pup_mask_mag.shape[1] < pup_mask.shape[1]): # bigger output size
            pup_mask_mag2 = np.zeros(pup_mask.shape)
            pup_mask_mag2[int(0.5*(pup_mask.shape[0]-pup_mask_mag.shape[0])):int(0.5*(pup_mask.shape[0]+pup_mask_mag.shape[0])), \
                     int(0.5*(pup_mask.shape[1]-pup_mask_mag.shape[1])):int(0.5*(pup_mask.shape[1]+pup_mask_mag.shape[1]))] = pup_mask_mag
        else: # same size
            pup_mask_mag2 = pup_mask_mag

    if all(element == -1 for element in wfs_translation):
        pup_mask = pup_mask_mag2
    else:
        pup_mask = shift(pup_mask_mag2, wfs_translation)

    # estimate an array proportional to flux per sub-aperture from the mask   
    if np.isnan(pup_mask_mag).any():
        np.nan_to_num(pup_mask_mag, copy=False, nan=0.0, posinf=None, neginf=None)

    pup_mask_sa = rebin(pup_mask, (wfs_nsubaps,wfs_nsubaps), method='sum')
    pup_mask_sa = pup_mask_sa * 1/np.max(pup_mask_sa)

    # rebin the array to get the correct signal size
    if np.isnan(der_dx).any():
        np.nan_to_num(der_dx, copy=False, nan=0.0, posinf=None, neginf=None)
    if np.isnan(der_dy).any():
        np.nan_to_num(der_dy, copy=False, nan=0.0, posinf=None, neginf=None)
    WFS_signal_x = rebin(der_dx, (wfs_nsubaps,wfs_nsubaps), method='average')
    WFS_signal_y = rebin(der_dy, (wfs_nsubaps,wfs_nsubaps), method='average')

    if verbose:
        print('Rebin done.')

    # normalize by pup_mask_sa to get the correct value at the edge of the pupil
    # because at the edge the average of the rebin is done with pixel outside the
    # pupil that have 0 values
    pup_mask_sa[pup_mask_sa<0.5] = 0
    WFS_signal_x = apply_mask(WFS_signal_x,pup_mask_sa,norm=True)
    WFS_signal_y = apply_mask(WFS_signal_y,pup_mask_sa,norm=True)

    if verbose:
        print('Mask applied.')
    
    WFS_signal_x_2D = WFS_signal_x.reshape((-1,WFS_signal_x.shape[2]))
    WFS_signal_y_2D = WFS_signal_y.reshape((-1,WFS_signal_y.shape[2]))

    if idx_valid_sa is not None:
        WFS_signal_x_2D = WFS_signal_x_2D[idx_valid_sa,:]
        WFS_signal_y_2D = WFS_signal_y_2D[idx_valid_sa,:]
        if verbose:
            print('Indices selected.')

    im = np.concatenate((WFS_signal_x_2D, WFS_signal_y_2D))

    # TODO missing normaliaztion!

    if verbose:
        print('WFS signals reformed.')

    if display:
        fig, _ = plt.subplots()
        plt.imshow(pup_mask_sa)
        plt.title('Pupil masks rebinned on WFS sub-apertures')
        plt.colorbar()
        
        fig, axs = plt.subplots(2,2)
        im3 = axs[0,0].imshow(trans_dm_array[:,:,2], cmap='seismic')
        im3 = axs[0,1].imshow(trans_dm_array[:,:,2], cmap='seismic')
        im3 = axs[1,0].imshow(trans_dm_array[:,:,5], cmap='seismic')
        im3 = axs[1,1].imshow(trans_dm_array[:,:,5], cmap='seismic')
        fig.suptitle('DM shapes seen on the WFS direction (idx 2 and 5)')
        fig.colorbar(im3, ax=axs.ravel().tolist(),fraction=0.02)
        
        fig, axs = plt.subplots(2,2)
        im4 = axs[0,0].imshow(der_dx[:,:,2], cmap='seismic')
        im4 = axs[0,1].imshow(der_dy[:,:,2], cmap='seismic')
        im4 = axs[1,0].imshow(der_dx[:,:,5], cmap='seismic')
        im4 = axs[1,1].imshow(der_dy[:,:,5], cmap='seismic')
        fig.suptitle('X and Y derivative of DM shapes seen on the WFS direction (idx 2 and 5)')
        fig.colorbar(im4, ax=axs.ravel().tolist(),fraction=0.02)
        
        fig, axs = plt.subplots(2,2)
        im5 = axs[0,0].imshow(WFS_signal_x[:,:,2], cmap='seismic')
        im5 = axs[0,1].imshow(WFS_signal_y[:,:,2], cmap='seismic')
        im5 = axs[1,0].imshow(WFS_signal_x[:,:,5], cmap='seismic')
        im5 = axs[1,1].imshow(WFS_signal_y[:,:,5], cmap='seismic')
        fig.suptitle('X and Y WFS signals')
        fig.colorbar(im5, ax=axs.ravel().tolist(),fraction=0.02)
        plt.show()

    return im
