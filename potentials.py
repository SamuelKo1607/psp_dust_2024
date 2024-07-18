import numpy as np
import datetime as dt
import cdflib
from tqdm.auto import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import alive_progress

from conversions import date2YYYYMMDD
from conversions import tt2000_to_date
from conversions import YYYYMMDD_to_tt2000
from conversions import tt2000_to_YYYYMMDD
from load_data import list_cdf
from ephemeris import get_state

from paths import l3_dust_location
from paths import dfb_location
from paths import psp_ephemeris_file
from paths import figures_location

import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi'] = 600


def get_epoch_range(vdcs):
    """
    Gets the approx. min and max epoch from the list of CDFs (vdc).

    Parameters
    ----------
    vdcs : list of str
        The list of VDC cdfs.

    Returns
    -------
    min_epoch : float
        The approximate minimum epoch.
    max_epoch : float
        The approximate maximum epoch.

    """
    YYYYMMDDs = [vdc[vdc.find("wf_vdc_2")+7:][:-10] for vdc in vdcs]
    epochs = [YYYYMMDD_to_tt2000(YYYYMMDD) for YYYYMMDD in YYYYMMDDs]
    min_epoch = min(epochs)
    max_epoch = max(epochs) + 86400000000000
    return min_epoch, max_epoch


def hand_cdf(epoch,vdcs):
    """
    Finds the cdf that covers the requested epoch time.

    Parameters
    ----------
    epoch : float
        The time of interest (tt2000).
    vdcs : list of str
        The list of cdfs to go through..

    Returns
    -------
    vdc : str or none
        The file, which covers the requested epoch.

    """
    YYYYMMDDs = [vdc[vdc.find("wf_vdc_2")+7:][:-10] for vdc in vdcs]
    start_hours = [vdc[vdc.find("wf_vdc_2")+15:][:-8] for vdc in vdcs]
    start_epochs = np.array([YYYYMMDD_to_tt2000(YYYYMMDD)
                             +3.6e12*int(start_hour)
                             for YYYYMMDD, start_hour
                             in zip(YYYYMMDDs,start_hours)])
    end_epochs = start_epochs+3.6e12*6
    criteria = (start_epochs<epoch)*(end_epochs>epoch)
    if np.sum(criteria)==0:
        vdc = None
    else:
        index = np.where(criteria)[0][0]
        vdc = vdcs[index]
    return vdc


def hand_potential(epoch,vdc):
    """
    Provides the monopole body voltage estimate as the sum of 
    the four antennas. Checks if the data is available.

    Parameters
    ----------
    epoch : float
        The time of interest, tt2000.
    vdc : str
        The location of the CDF of interest.

    Returns
    -------
    v : float or np.nan
        The body potential approximation, 
        returns np.nan if we don't have data.

    """

    vdc_cdf = cdflib.CDF(vdc)
    #epochs
    v1e = vdc_cdf.varget("v1_epoch")
    v2e = vdc_cdf.varget("v2_epoch")
    v3e = vdc_cdf.varget("v3_epoch")
    v4e = vdc_cdf.varget("v4_epoch")
    #waveforms
    v1w = vdc_cdf.varget("psp_fld_l2_dfb_wf_V1dc")
    v2w = vdc_cdf.varget("psp_fld_l2_dfb_wf_V2dc")
    v3w = vdc_cdf.varget("psp_fld_l2_dfb_wf_V3dc")
    v4w = vdc_cdf.varget("psp_fld_l2_dfb_wf_V4dc")
    #interpolate
    v1 = np.interp(epoch,v1e,v1w,left=np.nan,right=np.nan)
    v2 = np.interp(epoch,v2e,v2w,left=np.nan,right=np.nan)
    v3 = np.interp(epoch,v3e,v3w,left=np.nan,right=np.nan)
    v4 = np.interp(epoch,v4e,v4w,left=np.nan,right=np.nan)
    v = (v1+v2+v3+v4)/4
    return v


def get_samples(vdcs,
                samples=100):
    """
    Gives a sample of epochs and corresponding potentials.  

    Parameters
    ----------
    vdcs : list of str
        The list of cdf files available.
    samples : int, optional
        The number of samples requested. 
        The default is 1000.

    Returns
    -------
    epochs : np.array of float
        The times, randomly sampled, only valid (vdc available) returned.
    body_potentials : np.array of float
        The body potential approximation. One for every epoch returned.
    total_tries : int
        How many attempts dod we have to make to get the 
        requested number of valid samples.

    """
    e_min, e_max = get_epoch_range(vdcs)
    body_potentials = np.zeros(0,dtype=float)
    epochs = np.zeros(0,dtype=float)
    total_tries = 0
    with alive_progress.alive_bar(samples) as bar:
        while len(body_potentials) < samples:
            total_tries += 1
            epoch = np.random.uniform(e_min,e_max)
            cdf = hand_cdf(epoch,vdcs)
            if cdf is None:
                pass
            else:
                body_potential = hand_potential(epoch,cdf)
                if np.isfinite(body_potential) and -20<body_potential<20:
                    body_potentials = np.append(body_potentials,body_potential)
                    epochs = np.append(epochs,epoch)
                    bar()
                else:
                    pass
    return epochs, body_potentials, total_tries


def main(samples=100,
         quantile=0.90,
         filename="potentials"):
    epochs, body_potentials, total_tries = get_samples(list_cdf(dfb_location),
                                                       samples=samples)
    rs = np.zeros(0)
    for epoch in epochs:
        r, v, v_rad, v_phi, v_theta = get_state(epoch, psp_ephemeris_file)
        rs = np.append(rs, r)
    print(total_tries)
    dates = [tt2000_to_date(epoch) for epoch in epochs]

    body_potentials *= -1 # to get to -sum(V)

    r = np.linspace(min(rs),max(rs),25)
    lowers = [np.quantile(body_potentials[(r_lo<rs)*(rs<r_hi)],(1-quantile)/2)
              for r_lo,r_hi in zip(r[:-1],r[1:])]
    uppers = [np.quantile(body_potentials[(r_lo<rs)*(rs<r_hi)],1-(1-quantile)/2)
              for r_lo,r_hi in zip(r[:-1],r[1:])]
    means = [np.mean(body_potentials[(r_lo<rs)*(rs<r_hi)])
             for r_lo,r_hi in zip(r[:-1],r[1:])]
    mid_r = (r[:-1]+r[1:])/2
    zeros = np.zeros(len(mid_r))

    fig, ax = plt.subplots()
    ax.set_xlabel(r"Heliocentric distance $[AU]$")
    ax.set_ylabel(r"$V_{sc} [V]$")
    ax.scatter(rs[::],body_potentials[::],c="darkgrey",s=0.5,alpha=1,
               edgecolor="none",label="Potential")
    ax.plot(mid_r,means,c="k",label="Mean")
    ax.plot([0.1,1],[0,0],c="k",ls="dotted")
    ax.plot(mid_r,lowers,c="k",ls="dashed",
            label=f"{int(100*quantile)}\%")
    ax.plot(mid_r,uppers,c="k",ls="dashed")
    ax.legend(fontsize="small",ncol=3,loc=4)
    ax.set_xlim(0.1,0.95)
    ax.set_ylim(-5,5)
    fig.tight_layout()
    if filename is not None:
        fig.savefig(figures_location+filename+".pdf",format="pdf")
    fig.show()

    return epochs, rs, body_potentials

#%%
if __name__ == "__main__":

    epochs, rs, body_potentials = main(samples=100000)





