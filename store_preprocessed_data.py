import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt; plt.ion(); plt.rcParams['font.size'] = 10

import functions as fn
import good_FOV_filter
import encounter
import compute_FAC_VDF

def preprocess(trange, CREDENTIALS=None, CLIP=False):
    # getting the xarray from the L2 data
    xr = fn.init_psp_vdf(trange, CREDENTIALS=CREDENTIALS, CLIP=CLIP, filename=None)

    # extracting some other necessary parameters from L3 data
    xr_l3 = fn.init_psp_moms(trange, CREDENTIALS=CREDENTIALS, CLIP=CLIP)

    # exhancing the L2 x-array with only the quantities from L3 that will be needed downstream
    xr['MAGF_INST'] = (("time", "dim0"), xr_l3["MAGF_INST"].data)
    xr['VEL_INST']  = (("time", "dim0"), xr_l3["VEL_INST"].data)
    xr['DENS']      = ("time", xr_l3["DENS"].data)

    # extracting the good FOV mask
    __, time_mask = good_FOV_filter.gen_good_FOV_mask(xr)

    # only retaining good FOV times
    xr = xr.sel(time=time_mask)

    return xr, time_mask

def plot_gyrovdf(vperp, vpara, vdf, nanmask, ax, time_str, tidx=0):
    vperp_tidx = vperp[tidx][~nanmask[tidx]]
    vpara_tidx = vpara[tidx][~nanmask[tidx]]
    vdf_tidx = vdf[tidx][~nanmask[tidx]]

    # making the opposite side of this and repeating the indices
    vperp_tidx = np.concatenate([vperp_tidx, -vperp_tidx])
    vpara_tidx = np.concatenate([vpara_tidx, vpara_tidx])
    vdf_tidx = np.concatenate([vdf_tidx, vdf_tidx])

    ax.tricontourf(vperp_tidx, vpara_tidx, np.log10(vdf_tidx), cmap='magma')
    ax.set_aspect('equal')
    ax.set_xlim([-500,500])
    ax.set_ylim([-500,500])
    ax.text(0.85, 0.95, time_str, transform=ax.transAxes,
            fontsize=8, color='black', ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='lightgray', ec='lightgray', lw=1, alpha=0.8))


if __name__=='__main__':
    enc = 'E17'
    date = encounter.get_enc_dates(enc).astype(datetime) - timedelta(days=0)
    start_time = np.datetime64(date + timedelta(days=0)).astype('datetime64[s]').astype('str')
    end_time = np.datetime64(date + timedelta(days=1)).astype('datetime64[s]').astype('str')
    trange = [start_time, end_time]

    # computing the xarray, only for the filtered times with good FOV
    xr, time_mask = preprocess(trange)

    # computing the FAC representation for the good FOV events
    vperp, vpara, vdf, nanmask = compute_FAC_VDF.FAC_VDF_filtered(xr, count_threshold=3)

    # plotting the gyrovdf
    random_time_indices = np.random.randint(0, len(vperp), 25)
    # sorting the random time indices
    random_time_indices = np.sort(random_time_indices)

    fig, ax = plt.subplots(5, 5, figsize=(12,15), sharex=True, sharey=True)

    for idx, tidx in enumerate(random_time_indices):
        row, col = idx // 5, idx % 5
        plot_gyrovdf(vperp, vpara, vdf, nanmask, ax[row,col], xr.time.data[tidx].astype('datetime64[s]'), tidx=tidx)

    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.98, wspace=0.08, hspace=0.05)