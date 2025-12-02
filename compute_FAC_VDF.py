import numpy as np
NAX = np.newaxis

import functions as fn

def get_cartesian_velocity_grid(r, theta, phi):
    # Define the Cartesian Coordinates
    vx = r * np.cos(np.radians(theta)) * np.cos(np.radians(phi))
    vy = r * np.cos(np.radians(theta)) * np.sin(np.radians(phi))
    vz = r * np.sin(np.radians(theta))

    return vx, vy, vz

def get_FAC_grids(vx, vy, vz, B, U):
    # shifting to plasma frame
    ux = vx - U[:,0,NAX,NAX,NAX]
    uy = vy - U[:,1,NAX,NAX,NAX]
    uz = vz - U[:,2,NAX,NAX,NAX]

    # Rotate the plasma frame data into the magnetic field aligned frame.
    vpara, vperp1, vperp2 = np.array(fn.rotate_vector_field_aligned(ux, uy, uz, *fn.field_aligned_coordinates(B))) 

    # computing gyrotropic grids
    vperp = np.sqrt(vperp1**2 + vperp2**2)

    # returning all the grids converted to the gyroframe
    return vperp, vpara

def find_bin_mask(count_arr, count_threshold):
    '''
    Ideally this should also filter out the isolated bins, but need to come up
    with a better algorithm.
    '''
    bin_mask = count_arr < count_threshold
    return bin_mask


def FAC_VDF_filtered(xr, count_threshold=1):
    '''
    Parameters:
    -----------
    xr : xarray
              Dataframe in xarray format which contains the required L2 and L3 data for the good-FOV VDFs.
    count_threshold : int
              The minimum number of counts to be used to evaluate if a bin is to be used.
    '''
    m_p = 0.010438870    # eV/c^2 where c = 299792 km/s
    q_p = 1
    velocity = np.sqrt(2 * q_p * xr['ENERGY'].data / m_p)

    # building the vx, vy, vz grids from the polar grids
    r, theta, phi = velocity * 1.0, xr['THETA'].data * 1.0, xr['PHI'].data * 1.0
    vx, vy, vz = get_cartesian_velocity_grid(r, theta, phi)

    # rotating the vx, vy, vz to vperp and vpara magnetic Field-Aligned-Frame (FAC)
    vperp, vpara = get_FAC_grids(vx, vy, vz, xr['MAGF_INST'].data, xr['VEL_INST'].data)

    # filter isolated grids (use the algorithm used in hampy)
    # L2_data['COUNTS'] = remove_isolated_bins(L2_data['COUNTS'])

    # removing the bins which are below count threshold
    nanmask = find_bin_mask(xr['COUNTS'], count_threshold)

    return vperp, vpara, xr['VDF'].data, nanmask
