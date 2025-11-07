"""Zernike polynomial utilities for SynIM."""

import numpy as np
from scipy.special import eval_legendre
from scipy.ndimage import rotate

from synim.utils import (
    make_xy,
    make_mask,
    make_orto_modes
)

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

        idx_1d = np.where(mask.flatten())
        idx_2d = np.where(mask)

        z2phi_temp = z2phi.reshape(-1, maxZernNumber + 1)
        z2phi_on_pupil = z2phi_temp[idx_1d,:]
        z2phi_on_pupil = z2phi_on_pupil.reshape(-1,maxZernNumber + 1)

        z2phi_matrix_ortho = make_orto_modes(z2phi_on_pupil)

        #z2phi = np.zeros((dim, dim, maxZernNumber + 1), dtype=float)
        z2phi = np.full((dim,dim, maxZernNumber + 1),np.nan)

        for i in range(maxZernNumber + 1):
            temp = np.zeros((dim, dim), dtype=float)
            temp[idx_2d[0],idx_2d[1]] = z2phi_matrix_ortho[:, i] * 1/np.std(z2phi_matrix_ortho[:, i])
            temp[idx_2d[0],idx_2d[1]] = temp[idx_2d[0],idx_2d[1]] - np.mean(temp[idx_2d[0],idx_2d[1]])
            z2phi[:, :, i] = temp

        if verbose:
            print('Zernike cube orthogonalized!')

    z2phi = z2phi[:, :, 1:]

    return z2phi
