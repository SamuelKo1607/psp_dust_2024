import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
import datetime as dt
import functools
from numba import jit
from tqdm.auto import tqdm
from scipy.interpolate import CubicSpline
from scipy.stats import poisson
from scipy.signal import argrelextrema
from scipy.optimize import fmin

from eccentricity_core import bound_flux_vectorized
from eccentricity_core import r_smearing
from eccentricity_core import velocity_verlet
from ephemeris import get_approaches
from ephemeris import load_ephemeris
from load_data import encounter_group
from conversions import jd2date
from overplot_with_solo_result import get_detection_errors
from conversions import GM, AU
from conversions import date2jd, jd2date
from load_data import Observation
from load_data import load_all_obs

from paths import psp_sun_ephemeris_file
from paths import figures_location
from paths import all_obs_location

import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi'] = 600



def plot_maxima_zoom(data,
                     perihelia,
                     flux,
                     e,
                     max_perihelia=16,
                     aspect=2,
                     zoom=1.2,
                     days=7,
                     pointcolor="darkorange",
                     linecolor="black",
                     filename=None):

    # Getting the approaches
    approaches = np.linspace(1,16,
                             16,
                             dtype=int)

    approach_dates = np.array([jd2date(a)
                                  for a
                                  in perihelia])
    approach_groups = np.array([encounter_group(a)
                                   for a
                                   in approaches])

    dates = np.array([jd2date(jd) for jd in data["jd"]])
    post_approach_threshold_passages = np.array([
        np.min(dates[(dates>approach_date)
                     *(data["r_sc"]>0.4)])
        for approach_date in approach_dates])

    # Calculate the scatter plot
    point_errors = get_detection_errors(data["measured"])
    duty_hours = data["obs_time"]
    detecteds = data["measured"]
    scatter_point_errors = np.vstack((point_errors[0]
                                         / data["obs_time"],
                                      point_errors[1]
                                         / data["obs_time"]))

    # Caluclate the model line
    eff_rate_bound = flux

    # Plot
    fig = plt.figure(figsize=(4*aspect/zoom, 4/zoom))
    ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2, fig=fig)
    ax2 = plt.subplot2grid((2,6), (0,2), colspan=2, fig=fig)
    ax3 = plt.subplot2grid((2,6), (0,4), colspan=2, fig=fig)
    ax4 = plt.subplot2grid((2,6), (1,1), colspan=2, fig=fig)
    ax5 = plt.subplot2grid((2,6), (1,3), colspan=2, fig=fig)
    axes = np.array([ax1,ax2,ax3,ax4,ax5])

    for a in axes[:]:
        a.set_ylabel("Rate [/h equiv.]")
    for a in axes[:]:
        a.set_xlabel("Time after perihelion [h]")

    # Iterate the groups
    for i,ax in enumerate(axes):  #np.ndenumerate(axes):
        group = i+1
        if group in set(approach_groups):

            ax.set_title(f"Enc. group {group}")

            line_hourdiff = np.zeros(0)
            line_rate = np.zeros(0)

            for approach_date in approach_dates[approach_groups==group]:
                filtered_indices = np.abs(dates-approach_date
                                          )<dt.timedelta(days=days)
                datediff = dates[filtered_indices]-approach_date
                hourdiff = [24*d.days + d.seconds/3600
                            for d in datediff]
                passage_days = (np.max(post_approach_threshold_passages[
                                        approach_groups==group])
                                - approach_date)
                passage_hours = (24*passage_days.days
                                 + passage_days.seconds/3600)
                ax.scatter(hourdiff,
                           (detecteds[filtered_indices]
                            /duty_hours[filtered_indices]),
                          c=pointcolor,s=0.,zorder=100)
                ax.errorbar(hourdiff,
                            (detecteds[filtered_indices]
                             /duty_hours[filtered_indices]),
                            scatter_point_errors[:,filtered_indices],
                           c=pointcolor, lw=0., elinewidth=1,alpha=0.5)

                line_hourdiff = np.append(line_hourdiff,hourdiff)
                line_rate = np.append(line_rate,flux[filtered_indices])
            sortmask = line_hourdiff.argsort()

            ax.plot(line_hourdiff[sortmask],
                    line_rate[sortmask],
                    c=linecolor,lw=1,zorder=101,label=f"e = {e}")
            max_y = ax.get_ylim()[1]
            ax.vlines([-passage_hours,passage_hours],
                      0,10000,
                      color="gray")

            ax.set_ylim(0,max_y)
            ax.set_xlim(-days*24,days*24)

    fig.tight_layout()

    if filename is not None:
        fig.savefig(figures_location+filename+".png",dpi=1200)

    fig.show()


def load_ephem_data(ephemeris_file,
                    r_min=0,
                    r_max=0.35,
                    decimation=1):
    """
    Reads the ephemerides, returns distance, speeds while disregarding 
    the "z" component.

    Parameters
    ----------
    ephemeris_file : str
        Location of the ephemerides file.
    r_min : float, optional
        The minimum (exclusive) heliocentric distance. The default is 0.
    r_max : float, optional
        The maximum (exclusive) heliocentric distance. The default is 0.5.
    decimation : int, optional
        How much to decimate the data. The default is 4 = every 4th is kept.

    Returns
    -------
    data : pd.df
        All the data which "main" needs.

    """

    # ssb_ephem="psp_ssb_noheader.txt"
    # sun_ephem="psp_sun_noheader.txt"

    # ssb_ephem_file = os.path.join("data_synced",ssb_ephem)
    # sun_ephem_file = os.path.join("data_synced",sun_ephem)
    # sol_ephem_file = os.path.join("data_synced","sun_ssb_noheader.txt")

    # ephem_sun = load_ephemeris(sun_ephem_file)
    # hae_sun = ephem_sun[1]
    # x_sun = hae_sun[:,0]/(AU/1000) #AU
    # y_sun = hae_sun[:,1]/(AU/1000)
    # z_sun = hae_sun[:,2]/(AU/1000)
    # hae_v_sun = ephem_sun[2]
    # vx_sun = hae_v_sun[:,0] #km/s
    # vy_sun = hae_v_sun[:,1]
    # vz_sun = hae_v_sun[:,2]

    # ephem_sol = load_ephemeris(sol_ephem_file)
    # hae_sol = ephem_sol[1]
    # x_sol = hae_sol[:,0]/(AU/1000) #AU
    # y_sol = hae_sol[:,1]/(AU/1000)
    # z_sol = hae_sol[:,2]/(AU/1000)
    # hae_v_sol = ephem_sol[2]
    # vx_sol = hae_v_sol[:,0] #km/s
    # vy_sol = hae_v_sol[:,1]
    # vz_sol = hae_v_sol[:,2]

    (jd,
     hae,
     hae_v,
     hae_phi,
     radial_v,
     tangential_v,
     hae_theta,
     v_phi,
     v_theta) = load_ephemeris(ephemeris_file)

    x = hae[:,0]/(AU/1000) #AU
    y = hae[:,1]/(AU/1000)
    z = hae[:,2]/(AU/1000)

    vx = hae_v[:,0] #km/s
    vy = hae_v[:,1]
    vz = hae_v[:,2]

    # for i in [3,4]:
    #     per = perihelia[i]
    #     indices = indices = np.abs(jd-per)<5
    #     plt.plot(r_sc[indices],v_tot[indices])

    v_sc_r = np.zeros(len(x))
    v_sc_t = np.zeros(len(x))
    for i in range(len(x)):
        unit_radial = hae[i,0:2]/np.linalg.norm(hae[i,0:2])
        v_sc_r[i] = np.inner(unit_radial,hae_v[i,0:2])
        v_sc_t[i] = np.linalg.norm(hae_v[i,0:2]-v_sc_r[i]*unit_radial)
    r_sc = np.sqrt(x**2+y**2)
    area_front = np.ones(len(x))*6.11
    area_side = np.ones(len(x))*4.62

    data = pd.DataFrame(data=np.array([v_sc_r,
                                       v_sc_t,
                                       r_sc,
                                       area_front,
                                       area_side,
                                       jd]).transpose(),
                        index=np.arange(len(r_sc),dtype=int),
                        columns=["v_sc_r",
                                 "v_sc_t",
                                 "r_sc",
                                 "area_front",
                                 "area_side",
                                 "jd"])

    data = data[data["r_sc"]>r_min]
    data = data[data["r_sc"]<r_max]
    data = data[data.index % decimation == 0]

    return data


def get_detection_errors(counts,
                         prob_coverage = 0.9):
    """
    The function to calculate the errorbars for flux 
    assuming Poisson distribution and taking into account
    the number of detections.

    Parameters
    ----------
    counts : array of float
        Counts per day.
    prob_coverage : float, optional
        The coverage of the errobar interval. The default is 0.9.

    Returns
    -------
    err_plusminus_flux : np.array of float
        The errorbars, lower and upper bound, shape (2, n). 
        The unit is [1] (count).

    """

    counts_err_minus = -poisson.ppf(0.5-prob_coverage/2,
                                          mu=counts)+counts
    counts_err_plus  = +poisson.ppf(0.5+prob_coverage/2,
                                          mu=counts)-counts
    err_plusminus_counts = np.array([counts_err_minus,
                                   counts_err_plus])

    return err_plusminus_counts


def find_approaches(jd,
                    r,
                    threshold=0.2):
    """
    Finds the jds of approaches (locally lowest r).

    Parameters
    ----------
    jd : np.array 1D of float
        The array of julian dates.
    r : np.array 1D of float
        The array of radial distances.
    threshold : float, optional
        The maximum r to consider. The default is 0.2.

    Returns
    -------
    approaches : TYPE
        DESCRIPTION.

    """

    local_minima = argrelextrema(r, np.less)[0]
    approaches = jd[local_minima[r[local_minima]<threshold]]
    return approaches


def most_likely_prefactor(n,
                          e,
                          m,
                          c_min=0.5,
                          c_max=2,
                          thershold=1e-5):
    """
    Optimization algorithom to find the most likely c. 
    Assumes positive c.

    Parameters
    ----------
    n : np.array 1D of float
        Detected count.
    e : np.array 1D of float
        Exposure.
    m : np.array 1D of float
        Model count.
    c_min : float, optional
        the seed low c. The default is 0.5.
    c_max : float, optional
        the seed high c. The default is 2.
    thershold : float, optional
        how precise do we need it. The default is 1e-2.

    Returns
    -------
    c : float
        The most likely multiplicative prefactor.

    """
    cmax_loglik = np.sum(poisson.logpmf(n,c_max*m*e))
    cmin_loglik = np.sum(poisson.logpmf(n,c_min*m*e))
    while (c_max-c_min)>thershold:
        print(c_min,c_max)
        c = np.exp((np.log(c_min)+np.log(c_max))/2)
        c_loglik = np.sum(poisson.logpmf(n,c*m*e))

        if c_loglik > max(cmax_loglik,cmin_loglik):
            if cmax_loglik > cmin_loglik:
                c_min = c
                cmin_loglik = c_loglik
            else:
                c_max = c
                cmax_loglik = c_loglik
        else:
            if cmax_loglik > cmin_loglik:
                c_max = np.exp(np.log(c) + (np.log(c_max) - np.log(c_min)))
                cmax_loglik = np.sum(poisson.logpmf(n,c_max*m*e))
                c_min = c
                cmin_loglik = c_loglik
            else:
                c_min = np.exp(np.log(c) - (np.log(c_max) - np.log(c_min)))
                cmin_loglik = np.sum(poisson.logpmf(n,c_min*m*e))
                c_max = c
                cmax_loglik = c_loglik
    c = (c_min+c_max)/2
    return c


def plot_compare(r_model,
                 f_model,
                 r_obs,
                 f_obs,
                 errors_obs,
                 loc,
                 filename=None):
    """
    A simple plot of the data and the model at once.

    Parameters
    ----------
    r_model : np.array 1D of float
        Heliocentric distance [AU] in the model.
    f_model : np.array 1D of float
        Flux [/s] in the model.
    r_obs : np.array 1D of float
        Heliocentric distance [AU] in the data.
    f_obs : np.array 1D of float
        Flux [/s] in the data.
    errors_obs : np.array 2D shape (2,N) of float
        The errors +- unsigned of the data.

    Returns
    -------
    None.

    """
    fig,ax = plt.subplots()
    ax.plot(r_model,f_model)
    ax.errorbar(r_obs,f_obs,
                errors_obs)
    ax.vlines([0.02325,0.08836],0,0.1,"k")
    ax.set_xlim(0,0.2)
    ax.set_ylim(0,0.1)
    ax.set_xlabel("Heliocentric distance [AU]")
    ax.set_ylabel("Dust flux [/s]")
    if filename is not None:
        fig.savefig(loc+filename+".png",dpi=1200)
    plt.show()


def split_at_jumps(array,
                   condition_array=None,
                   step=10):
    """
    To split an array with missing data into a list of shorter arrays.

    Parameters
    ----------
    array : np.array 1D
        The array to be split. If "condition_array" is provided, this one 
        does not have to be of type float.
    condition_array : np.array 1D of float or None, optional
        The array to base the split on. If None, then "array" is used. 
        The default is None.
    step : float, optional
        How big a gap suffices for a cut. The default is 10.

    Returns
    -------
    list_of_chunks : list of np.array 1D of type as "array"
        The "array" after it was cut into shorter pieces.

    """
    if condition_array is None:
        condition_array = array
    if max(np.diff(condition_array))<=step:
        list_of_chunks = [array]
        return list_of_chunks
    else:
        jumps = np.where(np.diff(condition_array)>step)[0]
        chunk_sizes = np.append(jumps[0]+1,np.diff(jumps))
        last_chunk_size = len(array)-np.sum(chunk_sizes)
        chunk_sizes = np.append(chunk_sizes,last_chunk_size)
        list_of_chunks = []
        for chunk_size in chunk_sizes:
            list_of_chunks.append(array[:chunk_size])
            array = array[chunk_size:]
        return list_of_chunks


def join_with_nans(list_of_arrays):
    """
    Joins a list of 1D arrays back into a list, 
    but with nans at the seams.

    Parameters
    ----------
    list_of_arrays : list of 1D np.array
        DESCRIPTION.

    Returns
    -------
    arr : np.array
        DESCRIPTION.

    """
    arr = list_of_arrays[0]
    for a in list_of_arrays[1:]:
        arr = np.append(arr,np.nan)
        arr = np.append(arr,a)
    return arr


def powerlaw(c,slope,x):
    return c*(x**slope)


def loglik_slope_f(counts,
                   exposures,
                   f,
                   x):
    """
    Calculates the log-likelihood of a realization of n measurements 
    in a Poisson random process.

    Parameters
    ----------
    counts : np.array of int
        Data.
    exposures : np.array of float
        Exposures.
    f : function: 1D array -> 1D array
        Rate function.
    x : np.array of float
        Rate indep. variable.

    Returns
    -------
    loglik : float
        log likelihood.

    """
    loglik = np.sum(poisson.logpmf(counts,exposures*f(x)))
    return loglik


def best_constant_counts(counts,
                         exposures,
                         x,
                         slope):
    """
    Finds the most likely constant prefactor for the rate,
    given the slope.

    Parameters
    ----------
    counts : np.array of int
        Data.
    exposures : np.array of float
        Exposures.
    x : np.array of float
        Rate indep. variable.
    slope : float
        The slope of the powerlaw function.

    Returns
    -------
    max_c : float
        The best constant prefactor.

    """

    max_c = fmin(lambda c: -loglik_slope_f(counts,
                                           exposures,
                                           functools.partial(powerlaw,
                                                             c,slope),
                                           x), 1, disp=0)[0]
    return max_c


def loglik_slope(count_chunks,
                 duty_seconds_chunks,
                 r_chunks,
                 slope):
    """
    The loglik of the slope, given the data and given 
    the freedom in the multiplicative index of powerlaw 
    for each approach.

    Parameters
    ----------
    count_chunks : list of np.array 1D
        Chunks of the counts (a chunk per approach).
    duty_seconds_chunks : list of np.array 1D
        Chunks of the duty cycle (a chunk per approach).
    r_chunks : list of np.array 1D
        Chunks of the heliocentric distance (a chunk per approach).
    slope : float
        The slope of the powerlaw.

    Returns
    -------
    loglik : float
        The log likelihood of the slope.

    """
    constants = np.zeros(len(count_chunks))
    for i in range(len(count_chunks)):
        constants[i] = best_constant_counts(count_chunks[i],
                                            duty_seconds_chunks[i],
                                            r_chunks[i],
                                            slope)
    logliks = np.zeros(len(count_chunks))
    for i in range(len(count_chunks)):
        logliks[i] = loglik_slope_f(count_chunks[i],
                                    duty_seconds_chunks[i],
                                    functools.partial(powerlaw,
                                                      constants[i],slope),
                                    r_chunks[i])
    loglik = np.sum(logliks)
    return loglik, constants


def scaling_estimate(obs,
                     rmin=0.125,
                     rmax=0.25,
                     guess=-1.3):
    """
    Estimates the spatial scaling for the density, based on post-perihelia.

    Parameters
    ----------
    obs : list of load_data.Observation
        All the observations we have for PSP.
    rmin : float, optional
        What min distance to include. The default is 0.125.
    rmax : float, optional
        What max distance to include. The default is 0.25.

    Returns
    -------
    most_likely_slope : float
        The most likely slope, given all the observations.
    constants : np.array of float, 1D
        The prefactors for the individual approaches.

    """
    obs = [ob for ob in obs if (ob.duty_hours > 0.1
                                and ob.inbound == 1 # this means outbound
                                and ob.heliocentric_distance >= rmin
                                and ob.heliocentric_distance <= rmax
                                )]
    duty_seconds = np.array([ob.duty_hours for ob in obs])*3600
    count = np.array([ob.count_corrected for ob in obs])
    jd = np.array([ob.jd_center for ob in obs])
    r = np.array([ob.heliocentric_distance for ob in obs])

    duty_seconds_chunks = split_at_jumps(duty_seconds,jd)
    count_chunks = split_at_jumps(count,jd)
    r_chunks = split_at_jumps(r,jd)

    most_likely_slope = fmin(lambda s: -loglik_slope(count_chunks,
                                                     duty_seconds_chunks,
                                                     r_chunks,
                                                     s)[0],guess,disp=0)[0]
    constants = loglik_slope(count_chunks,
                             duty_seconds_chunks,
                             r_chunks,
                             most_likely_slope)[1]

    return most_likely_slope, constants


def estimate_powerlaws_individually(obs,
                                    guess=-1.3,
                                    max_r=0.25,
                                    max_encounter=16,
                                    maxima_indices = [5,4,4,7,9,2,5,4,
                                                      4,4,4,4,4,4,4,6],
                                    inspect=True):
    """
    Estiamtes the power-law slope of the flux 
    as a function of heliocentric distance
    for each encounter individually. 

    Parameters
    ----------
    obs : list of load_data.Observation
        All the observations we have for PSP.
    guess : float, optional
        The initial guess. The default is -1.3.
    max_r : float, optional
        Upper limit in R where the fight is done. The default is 0.25.
    max_encounter : int, optional
        How many encounters to include. The default is 16.
    maxima_indices : list of int, optional
        A manually defined indices at which to start the fit 
        for individual approaches. 
        The default is [5,4,4,7,9,2,5,4,4,4,4,4,4,4,4,6].                                                   4,4,4,4,4,4,4,6].
    inspect : bool, optional
        Whether to show the individual plots. The default is True.

    Returns
    -------
    None.

    """

    obs = [ob for ob in obs if (ob.duty_hours > 0.1
                                and ob.inbound == 1 # this means outbound
                                and ob.heliocentric_distance < max_r)]

    partial_obs = [[ob for ob in obs if (ob.encounter == e+1)]
                   for e in range(max_encounter)]

    slope_estimates = []
    encounters = []
    for i,obs in enumerate(partial_obs):
        maximum_index = maxima_indices[i]
        min_r = obs[maximum_index].heliocentric_distance

        duty_seconds = np.array([ob.duty_hours for ob in obs])*3600
        count = np.array([ob.count_corrected for ob in obs])
        jd = np.array([ob.jd_center for ob in obs])
        r = np.array([ob.heliocentric_distance for ob in obs])

        estimate = scaling_estimate(obs,rmin=min_r)
        slope_estimates.append(estimate[0])
        encounters.append(obs[0].encounter)
        if inspect:
            plt.plot(r,count/duty_seconds)
            plt.plot(r,powerlaw(estimate[1][0],estimate[0],r))
            plt.annotate(str(estimate[0])[:5], xy=(0.5, 0.9),
                         xycoords='axes fraction')
            plt.suptitle(f"encounter {obs[0].encounter}")
            plt.xlabel("R [AU]")
            plt.ylabel("flux [/s]")
            plt.show()

    plt.scatter(encounters,slope_estimates,c="r")
    plt.suptitle("slopes")
    plt.xlabel("encounter")
    plt.ylabel("flux slope")
    plt.show()


def estimate_powerlaws_all(obs,
                           compensation=-2.5,
                           dslope=0.3,
                           max_r=0.5,
                           encounter=None,
                           loc=None):
    """
    A paper plot. A compensated plot of the outbound PSP flux between 0.15
    and 0.5 AU. 

    Parameters
    ----------
    obs : list of object of class Observation
        The PSP observations.
    compensation : float, optional
        What to compensate by. The default is -2.5.
    dslope : float, optional
        The compensation +- value of this is shown in dashed lines. 
        The default is 0.3.
    max_r : float, optional
        Maximum heliocentric distance. The default is 0.5.
    encounter : int of None, optional
        If not None, only this encounter is shown. The default is None.                                        4,4,4,4,4,4,4,6].
    loc : str, optional
        Whete to save the plot. The default is None.

    Returns
    -------
    None.

    """

    obs = [ob for ob in obs if (ob.duty_hours > 0.1
                                and ob.inbound == 1 # this means outbound
                                and ob.heliocentric_distance < max_r
                                )]
    if encounter is not None:
        obs = [ob for ob in obs if (ob.encounter == encounter)]

    encounters = [ob.encounter for ob in obs]
    encounter_groups = [ob.encounter_group for ob in obs]
    jds_obs = np.array([ob.jd_center for ob in obs])

    norm = mpl.colors.Normalize(vmin=1,
                                vmax=10)
    cmap = cm.tab10
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    for enc in set(encounter_groups):
        r = np.array([ob.heliocentric_distance for ob in obs
                      if (ob.encounter_group == enc)])
        flux_obs = np.array([ob.count_corrected
                             /ob.duty_hours for ob in obs
                             if (ob.encounter_group == enc)])/3600
        plt.scatter(r,flux_obs,c=m.to_rgba(enc),label=f"{enc}")
    plt.yscale('log',base=10)
    plt.suptitle("All the fluxes")
    plt.xlabel("R [AU]")
    plt.ylabel("Flux [/s]")
    plt.xlim(0.15,0.5)
    plt.legend(ncol=3)
    plt.show()

    fig,ax = plt.subplots(5,figsize=(4,4))
    markers = ["s","s","x","x","p","p","v","v"]
    colors = 4*["k","darkgrey"]
    for i,gr in enumerate(set(encounter_groups)):

        #TBD differentiate by encounters
        encs = set([ob.encounter for ob in obs
                    if (ob.encounter_group == gr)
                    and (ob.heliocentric_distance < 0.5)
                    and (ob.heliocentric_distance > 0.15)])

        for j,enc in enumerate(encs):
            r = np.array([ob.heliocentric_distance for ob in obs
                          if (ob.encounter == enc)
                          and (ob.heliocentric_distance < 0.5)
                          and (ob.heliocentric_distance > 0.15)])
            flux_obs = np.array([ob.count_corrected
                                 /ob.duty_hours for ob in obs
                                 if (ob.encounter == enc)
                                 and (ob.heliocentric_distance < 0.5)
                                 and (ob.heliocentric_distance > 0.15)])/3600
            ax[i].scatter(r,flux_obs/(r**compensation)*10**4,
                        c=colors[j],label=f"{enc}",s=5,marker=markers[j],
                        zorder=100)

        r = np.array([ob.heliocentric_distance for ob in obs
                      if (ob.encounter_group == gr)
                      and (ob.heliocentric_distance < 0.5)
                      and (ob.heliocentric_distance > 0.15)])
        flux_obs = np.array([ob.count_corrected
                             /ob.duty_hours for ob in obs
                             if (ob.encounter_group == gr)
                             and (ob.heliocentric_distance < 0.5)
                             and (ob.heliocentric_distance > 0.15)])/3600

        ax[i].set_xlim(0.15,0.5)
        fake_r = np.linspace(0.15,0.5,20)
        #fit = np.poly1d(np.polyfit(r, flux_obs/(r**compensation)*10**4, 1))
        #midval = fit((min(r)+max(r))/2)
        #ax[i].plot([min(r),max(r)],[fit(min(r)),fit(max(r))],
        #           c="k")
        mean_r = 0.325#(min(r)+max(r))/2
        coeffs = np.polyfit(np.log(r), np.log(flux_obs), 1)
        logfit = np.poly1d(coeffs)
        midval = np.exp(logfit(np.log(mean_r)))/(
            mean_r**compensation)*10**4
        ax[i].plot(fake_r,
                   np.exp(logfit(np.log(fake_r)))/(fake_r**compensation)*10**4,
                   c="k")
        ax[i].add_patch(mpl.patches.Rectangle((0.155, 3.6),
                                              0.04, 1,
                                              fc = 'white',
                                              linewidth = 0, zorder = 49))
        ax[i].text(0.175, 4.1, f"{np.around(coeffs[0],2)}",
                   va = "center", ha = "center", zorder = 50)
        for slope in [-dslope,dslope]:
            new_slope = coeffs[0]+slope
            new_logfit = lambda x : x**(new_slope-compensation)
            new_midval = new_logfit(mean_r)
            prefactor = midval / new_midval
            #fit = np.poly1d(np.polyfit(r, flux_obs/
            #                           (r**(compensation+slope))*10**4, 1))
            ax[i].plot(fake_r,(prefactor * np.array(new_logfit(fake_r))),
                                c="k",ls="dashed")
        ax[i].set_ylabel(f"Group {gr}")
        ax[i].yaxis.set_ticks(np.array([0,2,4]))
        ax[i].grid(color='lightgrey', linestyle='-',
                   linewidth=1, axis="y", which = "both", alpha=0.5)
        if i!=4:
            ax[i].xaxis.set_ticklabels([])
        ax[i].set_ylim(bottom=0,top=5)
    ax[4].set_xlabel("R [AU]")
    ax[2].set_ylabel(f"Flux [$10^{{-4}} s^{{-1}} AU^{{{-compensation}}}$] \n"+
                     "Group 3",linespacing=2)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    if loc is not None:
        plt.savefig(loc+"compensated_flux.pdf",format="pdf")
    plt.show()


def maxima_plot(obs,
                days=7,
                marker_dist=[0.2,0.16,0.14,0.11,0.1],
                marker_time=[74,50.4,48.2,33.1,30.5],
                loc=None):

    obs = [ob for ob in obs if (ob.duty_hours > 0.1
                                and ob.heliocentric_distance < 0.3
                                and ob.encounter > 0)]

    encounters = np.array([ob.encounter for ob in obs])
    encounter_groups = np.array([ob.encounter_group for ob in obs])
    jds_obs = np.array([ob.jd_center for ob in obs])

    fig,ax = plt.subplots(5,figsize=(4,4))

    markers = ["s","s","x","x","p","p","v","v"]
    colors = 4*["k","darkgrey"]

    for i,gr in enumerate(set(encounter_groups)):
        for j,enc in enumerate(set(encounters[encounter_groups==gr])):

            r = np.array([ob.heliocentric_distance for ob in obs
                          if (ob.encounter == enc)])
            jd = np.array([ob.jd_center for ob in obs
                          if (ob.encounter == enc)])

            r_interp = CubicSpline(jd, r)
            index_peri = np.argmin(r)
            jd_fine = np.linspace(jd[index_peri-2],jd[index_peri+2],100)
            r_fine = r_interp(jd_fine)
            peri_jd = jd_fine[r_fine==np.min(r_fine)][0]
            compensated_hours = (jd - peri_jd)*24

            #mark ---- AU crossing
            jd_threshold = np.abs(jd[np.argmin(np.abs(r-marker_dist[i]))]
                            - peri_jd)*24

            print(jd_threshold)

            flux_obs = np.array([ob.count_corrected
                                 /ob.duty_hours for ob in obs
                                 if (ob.encounter == enc)])/3600
    
            ax[i].scatter(compensated_hours,flux_obs,
                          marker=markers[j],c=colors[j],s=4)

        ax[i].set_xlim(-days*24,days*24)
        ax[i].set_ylabel(f"Group {gr}")
        ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        toplim = 1.1*ax[i].get_ylim()[1]
        ax[i].vlines([marker_time[i],-marker_time[i]],0,toplim,
                     ls="dashed",color="k")
        ax[i].text(marker_time[i]+15, 0.85*toplim,
                   f"${marker_dist[i]} \, AU$",
                   horizontalalignment='left',
                   verticalalignment='top')
        if i!=4:
            ax[i].xaxis.set_ticklabels([])
        ax[i].set_ylim(bottom=0,top=toplim)

    ax[4].set_xlabel("Time since perihelia [h]")
    ax[2].set_ylabel("Flux "+"[$s^{-1}$]"+" \n"+
                     "Group 3",linespacing=2)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    if loc is not None:
        plt.savefig(loc+"maxima.pdf",format="pdf")
    plt.show()


def main(ephem,
         fig_loc,
         psp_obs,
         velocity_exponent=1,
         adaptive_align=True,
         maxima_indices = [5,4,4,7,4,2,5,4,4,4,4,4,4,4,4,6]):
    # get what we need from the data
    psp_obs = [ob for ob in psp_obs if ob.duty_hours > 0.1]
    jds_obs = np.array([ob.jd_center for ob in psp_obs])
    duty_seconds_obs = np.array([ob.duty_hours for ob in psp_obs])*3600
    flux_obs = np.array([ob.count_corrected
                         /ob.duty_hours for ob in psp_obs])/3600
    flux_obs_errors = np.array([get_detection_errors(ob.count_corrected)/
                                ob.duty_hours for ob in psp_obs])/3600

    # find the slope and the prefactor
    slope, prefactors = scaling_estimate(load_all_obs(all_obs_location),
                                         rmin=0.15,rmax=0.2)

    # evaluate model
    gamma = slope
    ex = 1e-3
    incl = 1e-3
    retro = 1e-5
    beta = 1e-5
    flux_model = bound_flux_vectorized(
        r_vector = ephem["r_sc"].to_numpy(),
        v_r_vector = ephem["v_sc_r"].to_numpy(),
        v_phi_vector = ephem["v_sc_t"].to_numpy(),
        S_front_vector = ephem["area_front"].to_numpy(),
        S_side_vector = ephem["area_side"].to_numpy(),
        ex = ex,
        incl = incl,
        retro = retro,
        beta = beta,
        gamma = gamma,
        velocity_exponent = velocity_exponent,
        n = 7e-9)
    jd = ephem["jd"].to_numpy()
    r_sc = ephem["r_sc"].to_numpy()
    flux_model_spline = CubicSpline(jd, flux_model)
    r_sc_spline = CubicSpline(jd, r_sc)
    approaches = find_approaches(jd,r_sc)

    # show the model vs. the data
    relative_rs = []
    relative_fs = []

    for i,approach in enumerate(approaches[approaches<max(jds_obs)]):
        approach_end = min(jd[(jd>approach)*(r_sc>0.2)])
        flux_obs_part = flux_obs[(jds_obs>approach)
                                  *(jds_obs<approach_end)]
        duty_seconds_obs_part = duty_seconds_obs[(jds_obs>approach)
                                                  *(jds_obs<approach_end)]
        flux_obs_errors_part = flux_obs_errors[(jds_obs>approach)
                                                *(jds_obs<approach_end),:]
        jds_obs_part = jds_obs[(jds_obs>approach)
                                *(jds_obs<approach_end)]
        r_obs_part = r_sc_spline(jds_obs_part)
        f_model_part = flux_model_spline(jds_obs_part)
        j = jd[(jd>approach)*(jd<approach_end)]
        r = r_sc[(jd>approach)*(jd<approach_end)]
        f = flux_model[(jd>approach)*(jd<approach_end)]
        flux_model_spline_r = CubicSpline(r,f)

        # TBD find the maximum and set align based on the maximum for the data
        if adaptive_align:
            align = r_obs_part>=r_obs_part[maxima_indices[i]]
        else:
            align = r_obs_part>0.16
        pref = ( np.mean(flux_obs_part[align])
                /np.mean(flux_model_spline_r(r_obs_part[align])) )

        # compare model and the data
        plot_compare(r,pref*f,
                     r_obs_part,flux_obs_part,
                     flux_obs_errors_part.transpose(),
                     loc=fig_loc,
                     filename="comparison_approach_"+str(i))

        relative_rs.append(r_obs_part)
        relative_fs.append(flux_obs_part
                           /(pref*flux_model_spline_r(r_obs_part)))

    for r,f in zip(relative_rs,relative_fs):
        plt.semilogy(r,f)
    plt.xlabel("Heliocentric distance [AU]")
    plt.ylabel("Relative flux [1]")
    plt.show()

    for r,f in zip(relative_rs,relative_fs):
        plt.plot(r,f/max(f))
    plt.xlabel("Heliocentric distance [AU]")
    plt.ylabel("Relative flux [1]")
    plt.show()



#%%
if __name__ == "__main__":
    psp_obs = load_all_obs(all_obs_location)
    loc = os.path.join(figures_location,"perihelia","ddz_profile","")
    estimate_powerlaws_all(psp_obs,compensation=-2.5,loc=loc)

    ephem = load_ephem_data(psp_sun_ephemeris_file)
    estimate_powerlaws_individually(psp_obs)
    main(ephem,
         loc,
         psp_obs,
         velocity_exponent=1)

    psp_obs = load_all_obs(all_obs_location)
    loc = os.path.join(figures_location,"perihelia","ddz_profile","")
    maxima_plot(psp_obs,loc=loc)

















