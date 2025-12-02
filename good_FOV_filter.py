import numpy as np
import pandas as pd
import pyspedas, cdflib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt; plt.ion()

import functions as fn
import encounter

def download_VDF_file(user_datetime, CREDENTIALS=None):
    tstart = f'{user_datetime.year:04d}-{user_datetime.month:02d}-{user_datetime.day:02d}/00:00:00'
    tend = f'{user_datetime.year:04d}-{user_datetime.month:02d}-{user_datetime.day:02d}/23:59:59'
    trange = [tstart, tend]
    
    if CREDENTIALS:
        files = pyspedas.psp.spi(trange, datatype='spi_sf00', level='L2', notplot=True, time_clip=True,
                downloadonly=True, last_version=True, username=CREDENTIALS[0], password=CREDENTIALS[1])
    else:
        files = pyspedas.psp.spi(trange, datatype='spi_sf00_8dx32ex8a', level='l2', notplot=True, time_clip=True, downloadonly=True, last_version=True)

    dat_raw = cdflib.CDF(files[0])
    dat = {}

    # creating the data slice (1 day max)
    dat['EPOCH']  = dat_raw['Epoch']
    dat['THETA']  = dat_raw['THETA'].reshape((-1,8,32,8))
    dat['PHI']    = dat_raw['PHI'].reshape((-1,8,32,8))
    dat['ENERGY'] = dat_raw['ENERGY'].reshape((-1,8,32,8))
    dat['EFLUX']  = dat_raw['EFLUX'].reshape((-1,8,32,8))

    return dat

def gen_good_FOV_mask(xr, good_anode_idx=2):
    # finding the maximum 
    VDF_flattened = np.reshape(xr['EFLUX'].data, (len(xr['EFLUX'].data), -1), 'C')
    maxindices_flat = np.nanargmax(VDF_flattened, axis=1)
    maxindices_3D = np.array(np.unravel_index(maxindices_flat, xr['EFLUX'].shape[1:])).T

    # finding the maximum in phi for each time
    VDF_phimax_t = xr['PHI'].data[0,0,0,maxindices_3D[:,-1]]

    # good FOV only where VDF_phimax_t is larger than the goodanode
    good_FOV_mask = VDF_phimax_t <= (xr['PHI'][0,0,0].data[good_anode_idx] + 1e-6)

    return VDF_phimax_t, good_FOV_mask


if __name__=='__main__':
    enc = 'E22'
    date = encounter.get_enc_dates(enc).astype(datetime) - timedelta(days=0)
    start_time = np.datetime64(date + timedelta(days=0)).astype('datetime64[s]').astype('str')
    end_time = np.datetime64(date + timedelta(days=1)).astype('datetime64[s]').astype('str')
    trange = [start_time, end_time]

    L2_data = fn.init_psp_vdf(trange, CREDENTIALS=None, CLIP=False, filename=None) #download_VDF_file(date)

    # extracting the datetime array
    dt = pd.to_datetime(L2_data['time'].data).to_pydatetime()

    # extracting the VDF and summing on E and theta and then normalizing max to 1
    VDF_t_phi = np.nansum(L2_data['EFLUX'], axis=(1,2))
    VDF_t_phi_max = np.nanmax(VDF_t_phi, axis=1)
    VDF_t_phi_norm = VDF_t_phi / VDF_t_phi_max[:,np.newaxis]

    # filling with nan is the max is nan
    VDF_t_phi_norm[VDF_t_phi_max==0] = np.nan

    # finding the maximum 
    VDF_flattened = np.reshape(L2_data['EFLUX'].data, (len(dt), -1), 'C')
    maxindices_flat = np.nanargmax(VDF_flattened, axis=1)
    maxindices_3D = np.array(np.unravel_index(maxindices_flat, L2_data['EFLUX'].data.shape[1:])).T

    # finding the maximum in phi for each time
    VDF_phimax_t, __ =  gen_good_FOV_mask(L2_data) #L2_data['PHI'][0,maxindices_3D[:,0],0,0]
    VDFsummed_phimax_t =  L2_data['PHI'].data[0,0,0,np.nanargmax(VDF_t_phi, axis=1)]

    # plotting
    plt.figure(figsize=(14,4))
    plt.pcolormesh(dt, L2_data['PHI'][0,0,0].data, VDF_t_phi_norm.T, vmax=1, cmap='magma', rasterized=True)
    plt.plot(dt, VDFsummed_phimax_t, 'xk', lw=3)
    plt.plot(dt, VDF_phimax_t, '.w', lw=3)
    plt.colorbar()
    plt.tight_layout()

