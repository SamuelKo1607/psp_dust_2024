import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from numba import jit
import datetime as dt
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from conversions import jd2date
from paths import figures_location

import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi'] = 600

from conversions import jd2date, date2jd

def load_data(solo_file=os.path.join("data_synced","solo_flux_readable.csv"),
              psp_file=os.path.join("data_synced","psp_flux_readable.csv")):
    """
    Loads the observational data. 
    
    Parameters
    ----------
    solo_file : str, optional
        The path to the Solar Orbiter file. 
        The default is os.path.join("data_synced","solo_flux_readable.csv").
    psp_file : str, optional
        The path to the PSP file. 
        The default is os.path.join("data_synced","psp_flux_readable.csv").
    
    Returns
    -------
    solo_df : pandas.DataFrame
        The observational data from SolO.
    psp_df : pandas.DataFrame
        The observational data from SolO.
    
    """
    solo_df = pd.read_csv(solo_file)
    solo_df = solo_df[solo_df["Detection time [hours]"]>0]
    solo_df.insert(len(solo_df.columns),"Area front [m^2]",
                   10.34 * np.ones(len(solo_df.index)),
                   allow_duplicates=True)
    solo_df.insert(len(solo_df.columns),"Area side [m^2]",
                   8.24 * np.ones(len(solo_df.index)),
                   allow_duplicates=True)
    psp_df = pd.read_csv(psp_file)
    psp_df = psp_df[psp_df["Detection time [hours]"]>0]

    return solo_df, psp_df


def find_matches(solo_df, psp_df,
                 solo_dtmin=dt.datetime(2010,1,1),
                 solo_dtmax=dt.datetime(2030,1,1),
                 n_matches=6,
                 drop_margin=30,
                 mute=True):

    # filter the date
    sod = solo_df[(solo_df['Julian date']<date2jd(solo_dtmax))
                 *(date2jd(solo_dtmin)<solo_df['Julian date'])]
    psd = psp_df[(psp_df['Julian date']<date2jd(dt.datetime(2030,1,1)))
                *(date2jd(dt.datetime(2010,1,1))<psp_df['Julian date'])]

    matches = []
    while len(matches)<n_matches:

        # 1st dimension is as in solo jd, 2nd dimension as in psp jd
        solo_r = np.tile(sod['Radial distance [au]'].to_numpy(),
                         (len(psd.index),1)).transpose()
        solo_vr = np.tile(sod['Radial velocity [km/s]'].to_numpy(),
                          (len(psd.index),1)).transpose()
        solo_vt = np.tile(sod['Tangential velocity [km/s]'].to_numpy(),
                          (len(psd.index),1)).transpose()
        psp_r = np.tile(psd['Radial distance [au]'].to_numpy(),
                        (len(sod.index),1))
        psp_vr = np.tile(psd['Radial velocity [km/s]'].to_numpy(),
                        (len(sod.index),1))
        psp_vt = np.tile(psd['Tangential velocity [km/s]'].to_numpy(),
                        (len(sod.index),1))

        # find the best and store
        badness = ((np.abs(solo_vr-psp_vr)/3)
                    + (np.abs(solo_vt-psp_vt)/3)
                    + (np.abs(solo_r-psp_r)/.1))
        best = np.argwhere(badness==np.min(badness))[0]
        solo_jd = sod['Julian date'].to_numpy()[best[0]]
        psp_jd = psd['Julian date'].to_numpy()[best[1]]
        matches.append((solo_jd, psp_jd))

        # drop everything near to the match
        sod = sod.drop(sod[np.isclose(sod['Julian date'],solo_jd,
                                      rtol=0,
                                      atol=drop_margin)].index)
        psd = psd.drop(psd[np.isclose(psd['Julian date'],psp_jd,
                                      rtol=0,
                                      atol=drop_margin)].index)

        if not mute:
            print("solo: ",solo_jd,jd2date(solo_jd),
                  sod['Radial distance [au]'].to_numpy()[best[0]],
                  sod['Radial velocity [km/s]'].to_numpy()[best[0]],
                  sod['Tangential velocity [km/s]'].to_numpy()[best[0]])
            print("psp: ",psp_jd,jd2date(psp_jd),
                  psd['Radial distance [au]'].to_numpy()[best[1]],
                  psd['Radial velocity [km/s]'].to_numpy()[best[1]],
                  psd['Tangential velocity [km/s]'].to_numpy()[best[1]])
            print("dr = ",
                  (sod['Radial distance [au]'].to_numpy()[best[0]]-
                   psd['Radial distance [au]'].to_numpy()[best[1]]))
            print("dv_rad = ",
                  (sod['Radial velocity [km/s]'].to_numpy()[best[0]]-
                   psd['Radial velocity [km/s]'].to_numpy()[best[1]]))
            print("dv_azim = ",
                  (sod['Tangential velocity [km/s]'].to_numpy()[best[0]]-
                   psd['Tangential velocity [km/s]'].to_numpy()[best[1]]))

    return matches


def get_near(df,jd,days=7):
    return df[np.abs(df['Julian date']-jd)<days]


def evaluate_match(s_n,s_e,s_r,s_vr,s_vt,
                   p_n,p_e,p_r,p_vr,p_vt):
    s_lo_rate = stats.poisson.ppf(0.05, mu=s_n)/s_e
    s_hi_rate = stats.poisson.ppf(0.95, mu=s_n)/s_e

    # if all is beta
    p_n_beta = p_n * ((50-s_vr)/(50-p_vr))**2.3 * (s_r**(-2))/(p_r**(-2))
    # if all is bound
    p_n_bound = p_n * (((30-s_vt)**2+s_vr**2)**0.5
                       /((30-p_vt)**2+p_vr**2)**0.5) * ((s_r**(-1.3))
                                                        /(p_r**(-1.3)))
    p_rate_beta = p_n_beta / p_e
    p_rate_bound = p_n_bound / p_e

    beta_lo = p_rate_beta / s_hi_rate
    beta_hi = p_rate_beta / s_lo_rate

    bound_lo = p_rate_bound / s_hi_rate
    bound_hi = p_rate_bound / s_lo_rate

    distance = (s_r+p_r)/2
    v_rad = (s_vr+p_vr)/2

    return distance,v_rad,beta_lo,beta_hi,bound_lo,bound_hi


def main(solo_df, psp_df, matches):


    solo_phase = get_phase_angle(solo_ephemeris_file)
    psp_phase = get_phase_angle(psp_sun_ephemeris_file)

    distances = np.zeros(0)
    v_rads = np.zeros(0)
    raws = np.zeros(0)
    raws_stds = np.zeros(0)
    beta_los = np.zeros(0)
    beta_his = np.zeros(0)
    bound_los = np.zeros(0)
    bound_his = np.zeros(0)

    solo_jds = np.zeros(0)
    psp_jds = np.zeros(0)

    for m in matches:
        s_hit_df = get_near(solo_df,m[0],days=7)
        solo_jds = np.append(solo_jds,m[0])
        s_flux = (np.sum(s_hit_df['Fluxes [/day]'])
                  /np.sum(s_hit_df['Detection time [hours]']
                          *s_hit_df['Area front [m^2]']))
        print(np.sum(s_hit_df['Fluxes [/day]']))
        s_r = np.mean(s_hit_df['Radial distance [au]'])
        s_v_rad = np.mean(s_hit_df['Radial velocity [km/s]'])
        s_v_tan = np.mean(s_hit_df['Tangential velocity [km/s]'])
        print("solo: ",jd2date(m[0]).date(),s_r,s_v_rad,s_v_tan,s_flux)
        print("longitude: ",solo_phase(m[0])+(solo_phase(m[0])<0)*360)

        p_hit_df = get_near(psp_df,m[1],days=7)
        psp_jds = np.append(psp_jds,m[1])
        p_flux = (np.sum(p_hit_df['Count corrected [/day]'])
                  /np.sum(p_hit_df['Detection time [hours]']
                          *p_hit_df['Area front [m^2]']))
        print(np.sum(p_hit_df['Count corrected [/day]']))
        p_r = np.mean(p_hit_df['Radial distance [au]'])
        p_v_rad = np.mean(p_hit_df['Radial velocity [km/s]'])
        p_v_tan = np.mean(p_hit_df['Tangential velocity [km/s]'])
        print("psp: ",jd2date(m[1]).date(),p_r,p_v_rad,p_v_tan,p_flux)
        print("longitude: ",psp_phase(m[1])+(psp_phase(m[1])<0)*360)

        s_flux_bootstrapped = np.random.poisson(
            np.sum(s_hit_df['Fluxes [/day]']),
            size=1000) /np.sum(s_hit_df['Detection time [hours]']
                               *s_hit_df['Area front [m^2]'])
        p_flux_bootstrapped = np.random.poisson(
            np.sum(p_hit_df['Count corrected [/day]']),
            size=1000) /np.sum(p_hit_df['Detection time [hours]']
                               *p_hit_df['Area front [m^2]'])

        raws_bootstrapped = p_flux_bootstrapped / s_flux_bootstrapped
        std_bootstrepped = np.std(raws_bootstrapped)

        distance,v_rad,beta_lo,beta_hi,bound_lo,bound_hi = evaluate_match(
            np.sum(s_hit_df['Fluxes [/day]']),
            np.sum(s_hit_df['Detection time [hours]']
                    *s_hit_df['Area front [m^2]']),
            s_r, s_v_rad, s_v_tan,
            np.sum(p_hit_df['Count corrected [/day]']),
            np.sum(p_hit_df['Detection time [hours]']
                    *p_hit_df['Area front [m^2]']),
            p_r, p_v_rad, p_v_tan)

        distances = np.append(distances,distance)
        v_rads = np.append(v_rads,v_rad)
        raws = np.append(raws,p_flux/s_flux)
        raws_stds = np.append(raws_stds,std_bootstrepped)
        beta_los = np.append(beta_los,beta_lo)
        beta_his = np.append(beta_his,beta_hi)
        bound_los = np.append(bound_los,bound_lo)
        bound_his = np.append(bound_his,bound_hi)

    plt.scatter(distances,raws,color="k",label="raw")
    plt.vlines(distances,beta_los,beta_his,
               color="red",alpha=0.5,label="if all beta")
    plt.vlines(distances,bound_los,bound_his,
               color="blue",alpha=0.5,label="if all bound")
    plt.xlabel("Heliocentric distance [AU]")
    plt.ylabel("PSP vs SolO sensitivity")
    plt.legend(loc=9,fontsize="small")
    plt.show()

    plt.scatter(v_rads,raws,color="k",label="raw")
    offset = np.zeros(6)
    offset[0] = -5 #-3
    offset[1] = 5  #1.5
    offset[2] = 5
    offset[3] = -5
    offset[4] = 5
    offset[5] = 5

    for i in range(len(psp_jds)):
        plt.annotate(jd2date(psp_jds[i]).date().strftime("%d/%m/%y"),
                     (v_rads[i]+offset[i],raws[i]+0.01),
                     c="k",ha="center",fontsize="x-small")
        plt.annotate(jd2date(solo_jds[i]).date().strftime("%d/%m/%y"),
                     (v_rads[i]+offset[i],raws[i]-0.02),
                     c="grey",ha="center",fontsize="x-small")
        plt.vlines(v_rads[i],raws[i]-raws_stds[i],raws[i]+raws_stds[i],
                   colors="k")

    plt.text(0.05, 0.92, 'PSP date',
             horizontalalignment='left',
             verticalalignment='top',
             c="k", fontsize="small",
             transform = plt.gca().transAxes)
    plt.text(0.05, 0.85, 'SolO date',
             horizontalalignment='left',
             verticalalignment='top',
             c="grey", fontsize="small",
             transform = plt.gca().transAxes)
    plt.xlabel(r"Radial speed [$km/s$]")
    plt.ylabel("14 days flux PSP/SolO [$1$]")
    plt.xlim(-26,26)
    plt.ylim(0.26,0.61)
    plt.tight_layout()
    plt.savefig(figures_location+"case_study"+".pdf",format="pdf")
    plt.show()

    return matches



#%%
if __name__ == "__main__":

    solo_df, psp_df = load_data()
    matches = find_matches(solo_df, psp_df)

    print(main(solo_df, psp_df, matches))
