import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import datetime as dt
from numba import jit
from tqdm.auto import tqdm
from matplotlib.ticker import FormatStrFormatter

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

from paths import psp_ephemeris_file
from paths import figures_location

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


def density_scaling(gamma=-1.3,
                    ec=0.001,
                    r_min=0.08,
                    r_max=1.22,
                    size=200000,
                    mu=GM,
                    loc=figures_location,
                    mute=False):
    """
    Analyzes if the slope of heliocentric distance distribution 
    has changed or not.

    Parameters
    ----------
    gamma : float, optional
        The slope. The default is -1.3.
    ec : float, optional
        Eccentricity. The default is 0.001.
    r_min : float, optional
        min perihelion. The default is 0.04.
    r_max : float, optional
        max perihelion. The default is 1.1.
    size : int, optional
        number of sampled orbits. The default is 50000.
    mu : float, optional
        gravitational parameter. The default is GM.
    loc : str, optional
        Figure target directory. The default is figures_location.

    Returns
    -------
    None.

    """

    r_peri_proposed = np.random.uniform(r_min/10,r_max,size)
    thresholds = ((r_peri_proposed)**(gamma+1))/max(r_peri_proposed**(gamma+1))
    r_peri = r_peri_proposed[thresholds > np.random.uniform(0,1,size)]
    spreaded = np.zeros(0)
    all_sampled = []
    for r in tqdm(r_peri):
        samples = r_smearing(r,ec,size=500,burnin=100)
        all_sampled.append(samples)
    spreaded = np.reshape(all_sampled,newshape=(500*len(r_peri)))

    bins = np.linspace(r_min,r_max,int((size/50)**0.5))
    bincenters = (bins[1:]+bins[:-1])/2

    hist_orig = np.histogram(r_peri,bins,weights=r_peri**(-gamma-1))
    hist_mod = np.histogram(spreaded,bins,weights=spreaded**(-gamma-1))

    if mute:
        return bincenters, hist_orig, hist_mod
    else:
        fig,ax = plt.subplots()
        ax.hlines(1,r_min,r_max,"k",ls="dotted")
        ax.step(bincenters, hist_orig[0]/np.mean(hist_orig[0]),
                where='mid', label="$f(r_{peri})$", c="grey")
        ax.step(bincenters, hist_mod[0]/np.mean(hist_mod[0]),
                where='mid', label="$f(r)$", c="k")
        ax.legend(fontsize="small",loc=2)
        ax.text(0.15,0.83,rf"$e$ = {ec}, $\gamma$ = {gamma}")
        ax.set_xlabel("Heliocentric distance [AU]")
        ax.set_ylabel(r"$\gamma$-compensated pdf"+"\n"
                      +r"[$C \cdot m^{-2}AU^{-\gamma-1}$]",linespacing=1.2)
        ax.set_ylim(0.8,1.2)
        ax.set_xlim(0.1,1.2)
        ax.set_aspect(1.5)
        fig.tight_layout()
        if loc is not None:
            plt.savefig(loc+f"spread_{gamma}_{ec}"+".pdf",format="pdf")
        plt.show()


def density_scaling_multiple(ec=[0.2,0.8],
                             gamma=-1.3,
                             r_min=0.08,
                             r_max=1.22,
                             loc=figures_location,
                             *kwargs):
    """
    Analyzes if the slope of heliocentric distance distribution 
    has changed or not. Shows multiple plots.

    Parameters
    ----------
    ec : list of float, optional
        Eccentricity. The default is [0.2,0.8].

    Returns
    -------
    None.

    """

    fig,axs = plt.subplots(1,len(ec),sharey=True)
    fig.subplots_adjust(wspace=0.05)
    for i,ax in enumerate(axs):
        bincenters, hist_orig, hist_mod = density_scaling(gamma=gamma,
                                                          ec=ec[i],
                                                          r_min=r_min,
                                                          r_max=r_max,
                                                          mute=True,
                                                          *kwargs)
        ax.hlines(1,r_min,r_max,"k",ls="dotted")
        ax.step(bincenters, hist_orig[0]/np.mean(hist_orig[0]),
                where='mid', label="$f(r_{peri})$", c="grey")
        ax.step(bincenters, hist_mod[0]/np.mean(hist_mod[0]),
                where='mid', label="$f(r)$", c="k")
        ax.text(0.15,0.83,rf"$e$ = {ec[i]}"+"\n"+rf"$\gamma$ = {gamma}",
                linespacing=1.2)
        ax.set_xlabel("Heliocentric \n distance "+"[AU]",linespacing=1.2)
        ax.set_ylim(0.8,1.2)
        ax.set_xlim(0.1,1.2)
    axs[0].set_ylabel(r"$\gamma$-compensated pdf"+"\n"
                      +r"[$C \cdot m^{-2}AU^{-\gamma-1}$]",linespacing=1.2)
    axs[0].legend(fontsize="small",loc=2)
    fig.tight_layout()
    if loc is not None:
        plt.savefig(loc+f"spread_{gamma}_{ec}"+".pdf",format="pdf")
    plt.show()


def load_ephem_data(ephemeris_file,
                    r_min=0,
                    r_max=0.4,
                    decimation=4):
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


def construct_perihel(jd_peri,
                      n_peri,
                      r_peri,
                      v_peri,
                      days=14,
                      step_hours=1,
                      outbound_only=False):
    """
    Cosntructs an artificial part of an orbit.

    Parameters
    ----------
    jd_peri : TYPE
        DESCRIPTION.
    n_peri : TYPE
        DESCRIPTION.
    r_peri : TYPE
        DESCRIPTION.
    v_peri : TYPE
        DESCRIPTION.
    days : TYPE, optional
        DESCRIPTION. The default is 14.

    Returns
    -------
    data : pd.dataframe
        The same as load_ephem_data() provides (AU, km/s).

    """
    r,v = velocity_verlet(r_peri,v_peri,days=days,step_hours=step_hours)
    r = np.vstack((np.array([ 1,-1,-1])*np.flip(r,axis=0)[:-1,:],r))
    v = np.vstack((np.array([-1, 1, 1])*np.flip(v,axis=0)[:-1,:],v))
    jd = np.arange(0,days*24/step_hours+1)/(24/step_hours)
    jd = np.hstack((-np.flip(jd)[:-1],jd))+jd_peri
    r_sc = np.linalg.norm(r,axis=1)
    v_sc_r = np.zeros(len(jd))
    v_sc_t = np.zeros(len(jd))
    for i in range(len(jd)):
        unit_radial = r[i,:]/np.linalg.norm(r[i,:])
        v_sc_r[i] = np.inner(unit_radial,v[i,:])
        v_sc_t[i] = np.linalg.norm(v[i,:]-v_sc_r[i]*unit_radial)
    area_front = np.ones(len(jd))*6.11
    area_side = np.ones(len(jd))*4.62

    data = pd.DataFrame(data=np.array([v_sc_r/1000,
                                       v_sc_t/1000,
                                       r_sc/AU,
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

    if outbound_only:
        data = data.loc[data['jd'] >= jd_peri]

    return data


def construct_perihel_n(n,days=10,step_hours=2,outbound_only=False):
    """
    A function to construct syntetic perihel given the encounter number, 
    based on wikipedia numbers.A wrapper for "construct_perihel".

    Parameters
    ----------
    n : int
        Encounter number, 1 to 16.
    days : int, optional
        How many days after the perihel do we want. The default is 10.

    Raises
    ------
    Exception
        If the n is bad.

    Returns
    -------
    data : pd.dataframe
        The same as load_ephem_data() provides (AU, km/s).

    """
    if n>16 or n<1 or type(n) is not int:
        raise Exception("bad n, int n: 1<=n<=16 needed")
    for (jd_peri,
         n_peri,
         r_peri,
         v_peri) in zip([date2jd(dt.date(2018,11, 6)),
                         date2jd(dt.date(2019, 4, 4)),
                         date2jd(dt.date(2019, 9, 1)),
                         date2jd(dt.date(2020, 1,29)),
                         date2jd(dt.date(2020, 6, 7)),
                         date2jd(dt.date(2020, 9,27)),
                         date2jd(dt.date(2021, 1,17)),
                         date2jd(dt.date(2021, 4,28)),
                         date2jd(dt.date(2021, 8, 9)),
                         date2jd(dt.date(2021,11,21)),
                         date2jd(dt.date(2022, 2,25)),
                         date2jd(dt.date(2022, 6, 1)),
                         date2jd(dt.date(2022, 9, 6)),
                         date2jd(dt.date(2022,12,11)),
                         date2jd(dt.date(2023, 3,17)),
                         date2jd(dt.date(2023, 6,22))],
                       [1,2,3,
                        4,5,
                        6,7,
                        8,9,
                        10,11,12,13,14,15,16],
                       [2.48e10,2.48e10,2.48e10,
                        1.94e10,1.94e10,
                        1.42e10,1.42e10,
                        1.11e10,1.11e10,
                        9.2e9,9.2e9,9.2e9,9.2e9,9.2e9,9.2e9,9.2e9],
                       [9.5e4,9.5e4,9.5e4,
                        1.09e5,1.09e5,
                        1.29e5,1.29e5,
                        1.47e5,1.47e5,
                        1.63e5,1.63e5,1.63e5,1.63e5,1.63e5,1.63e5,1.63e5]):
        if n_peri == n:
            data = construct_perihel(jd_peri,
                                     n_peri,
                                     r_peri,
                                     v_peri,
                                     days=days,
                                     step_hours=step_hours,
                                     outbound_only=outbound_only)
    return data


def plot_single(data,
                ec=1e-3,
                incl=1e-3,
                retro=1e-5,
                beta=0,
                gamma=-1.3,
                vexp=1,
                loc=figures_location,
                peri=0):

    r_vector = data["r_sc"].to_numpy()
    v_r_vector = data["v_sc_r"].to_numpy()
    v_phi_vector = (data["v_sc_t"].to_numpy())
    S_front_vector = data["area_front"].to_numpy()
    S_side_vector = data["area_side"].to_numpy()
    jd = data["jd"].to_numpy()

    flux_front = bound_flux_vectorized(
        r_vector = r_vector,
        v_r_vector = v_r_vector,
        v_phi_vector = v_phi_vector,
        S_front_vector = S_front_vector,
        S_side_vector = S_side_vector*0,
        ex = ec,
        incl = incl,
        retro = retro,
        beta = beta,
        gamma = gamma,
        velocity_exponent = vexp,
        n = 7e-9)
    flux_side = bound_flux_vectorized(
        r_vector = r_vector,
        v_r_vector = v_r_vector,
        v_phi_vector = v_phi_vector,
        S_front_vector = S_front_vector*0,
        S_side_vector = S_side_vector,
        ex = ec,
        incl = incl,
        retro = retro,
        beta = beta,
        gamma = gamma,
        velocity_exponent = vexp,
        n = 7e-9)

    day_delta = np.array([jd2date(j) for j in jd]) - jd2date(np.mean(jd))
    days = np.array([d.days + d.seconds/(24*3600) for d in day_delta])

    flux = flux_front+flux_side
    dip = 1-(flux[len(flux)//2]/max(flux))

    fig, ax = plt.subplots()

    ax.plot(days,flux_front,label="Radial")
    ax.plot(days,flux_side,label="Azimuthal")
    ax.plot(days,flux,label="Total")
    ax.legend(facecolor='white',framealpha=1,loc=3,fontsize="small")
    ax.text(min(days)+1,0.9*max(flux),f"dip = {dip:.3}")
    ax.set_xlabel("Time since perihelion [d]")
    ax.set_ylabel("Dust detection rate [/s]")
    ax.set_ylim(bottom=0)
    ax.set_xlim(min(days),max(days))
    ax.set_title(f"peri: {peri};\necc={ec}; incl={incl}; "
                 +f"retro={retro}; beta={beta}; vexp={vexp}")
    fig.tight_layout()
    if loc is not None:
        plt.savefig(loc+f"peri_{peri}_ecc_{ec}_incl_{incl}"
                    +f"_retro_{retro}_beta_{beta}_vexp_{vexp}"
                    +".png",dpi=1200)
    plt.show()


def show_slopes(ec=1e-4,
                incl=1e-4,
                retro=1e-4,
                beta=0,
                gamma=-1.3,
                vexp=1,
                loc=None,
                rmin=0.15,
                rmax=0.5,
                compensation=-2.5,
                peris=[4,6,8,10],
                log=True):
    """
    The intention is to see what we have to really do to explain the slope
    in flux(r) of -2,5. 

    Parameters
    ----------
    ec : TYPE, optional
        DESCRIPTION. The default is 1e-4.
    incl : TYPE, optional
        DESCRIPTION. The default is 1e-4.
    retro : TYPE, optional
        DESCRIPTION. The default is 1e-4.
    beta : TYPE, optional
        DESCRIPTION. The default is 0.
    gamma : TYPE, optional
        DESCRIPTION. The default is -1.3.
    vexp : TYPE, optional
        DESCRIPTION. The default is 1.
    loc : TYPE, optional
        DESCRIPTION. The default is None.
    rmin : TYPE, optional
        DESCRIPTION. The default is 0.15.
    rmax : TYPE, optional
        DESCRIPTION. The default is 0.5.
    compensation : TYPE, optional
        DESCRIPTION. The default is -2.5.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots()
    for peri in peris:
        data = construct_perihel_n(peri,days=20,step_hours=8)

        r_vector = data["r_sc"].to_numpy()
        v_r_vector = data["v_sc_r"].to_numpy()
        v_phi_vector = (data["v_sc_t"].to_numpy())
        S_front_vector = data["area_front"].to_numpy()
        S_side_vector = data["area_side"].to_numpy()
        jd = data["jd"].to_numpy()

        flux = bound_flux_vectorized(
            r_vector = r_vector,
            v_r_vector = v_r_vector,
            v_phi_vector = v_phi_vector,
            S_front_vector = S_front_vector,
            S_side_vector = S_side_vector,
            ex = ec,
            incl = incl,
            retro = retro,
            beta = beta,
            gamma = gamma,
            velocity_exponent = vexp,
            n = 7e-9)

        ax.plot(r_vector,flux/(r_vector**compensation))

    ax.set_xlim(rmin,rmax)
    #ax.set_xscale("log")
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("Heliocentric distance [AU]")
    ax.set_ylabel("Compensated model flux [arb.u.]")
    ax.annotate(f"vexp = {vexp}", xy=(0.05, 0.1),
                 xycoords='axes fraction')
    ax.annotate(f"gamma = {gamma}", xy=(0.35, 0.1),
                 xycoords='axes fraction')
    ax.annotate(f"beta = {beta}", xy=(0.65, 0.1),
                 xycoords='axes fraction')
    fig.suptitle(f"All the fluxes - compensated by {compensation}")
    plt.show()


def plot_compare(ec=1e-4,
                 incl=1e-4,
                 retro=1e-4,
                 beta=0,
                 gamma=-1.3,
                 vexp=1,
                 loc=None,
                 peri=4,
                 ymax=None,
                 att="ec",
                 att_values=[1e-5,0.1,0.2,0.3,0.4]):
    """
    Compares profiles given by different values of dust orbital parameters.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    ec : TYPE, optional
        DESCRIPTION. The default is 1e-5.
    incl : TYPE, optional
        DESCRIPTION. The default is 1e-5.
    retro : TYPE, optional
        DESCRIPTION. The default is 1e-10.
    beta : TYPE, optional
        DESCRIPTION. The default is 0.
    gamma : TYPE, optional
        DESCRIPTION. The default is -1.3.
    loc : TYPE, optional
        DESCRIPTION. The default is figures_location.
    peri : TYPE, optional
        DESCRIPTION. The default is 0.
    att : str, optional
        DESCRIPTION. The default is "ec".
    att_values : list of float, optional
        The values to compare. The default is [1e-5,0.1,0.2,0.3,0.4].

    Returns
    -------
    None.

    """

    data = construct_perihel_n(peri)

    r_vector = data["r_sc"].to_numpy()
    v_r_vector = data["v_sc_r"].to_numpy()
    v_phi_vector = (data["v_sc_t"].to_numpy())
    S_front_vector = data["area_front"].to_numpy()
    S_side_vector = data["area_side"].to_numpy()
    jd = data["jd"].to_numpy()

    day_delta = np.array([jd2date(j) for j in jd]) - jd2date(np.mean(jd))
    days = np.array([d.days + d.seconds/(24*3600) for d in day_delta])

    fig, ax = plt.subplots()
    for i,value in enumerate(att_values):
        if att == "ec":
            ec = value
        elif att == "incl":
            incl = value
        elif att == "retro":
            retro = value
        elif att == "beta":
            beta = value
        elif att == "gamma":
            gamma = value
        elif att == "vexp":
            vexp = value
        else:
            raise Exception(
                'bad att, allowed: ["ec","inlc","retro","beta",'
                +'"gamma","vexp"]')
        flux = bound_flux_vectorized(
            r_vector = r_vector,
            v_r_vector = v_r_vector,
            v_phi_vector = v_phi_vector,
            S_front_vector = S_front_vector,
            S_side_vector = S_side_vector,
            ex = ec,
            incl = incl,
            retro = retro,
            beta = beta,
            gamma = gamma,
            velocity_exponent = vexp,
            n = 7e-9)
        if att == "vexp" and ymax is not None:
            compensation = np.max(flux)**(-1)*ymax/2
        else:
            compensation = 1
        ax.plot(days,flux*compensation,label=att+"="+"{:2.2f}".format(value))
        dip = 1-(flux[len(flux)//2]/max(flux))
        if ymax is None:
            ymax = ax.get_ylim()[1]
        ax.text(min(days)+1,(0.9-0.08*i)*ymax,
                "dip = "+"{:1.2f}".format(dip),color="C"+str(i))
    ax.set_xlabel(f"Time since perihelion {peri} [d]")
    ax.set_ylabel("Dust detection rate [/s]")
    ax.set_ylim(bottom=0,top=ymax)
    ax.set_xlim(min(days),max(days))
    ax.legend(facecolor='white',framealpha=0,loc=1,fontsize="small")
    fig.tight_layout()
    if loc is not None:
        plt.savefig(loc+f"peri_{peri}_att_"+att+".png",dpi=1200)
    plt.show()


def show_relative_velocity(rmin=0.,
                           rmax=0.5,
                           peris=[1,4,6,8,10]):
    """
    Creates a plot of relative speed between PSP and circular dust
    for different orbital groups.

    Parameters
    ----------
    rmin : float, optional
        lower x lim. The default is 0..
    rmax : float, optional
        upper x lim. The default is 0.5.
    peris : list of int, optional
        Which perihelia to show. The default is [1,4,6,8,10].

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots()
    for i,peri in enumerate(peris):
        data = construct_perihel_n(peri,days=30)
        r_vector = data["r_sc"].to_numpy()
        v_r_vector = data["v_sc_r"].to_numpy()
        v_phi_vector = (data["v_sc_t"].to_numpy())
        v_phi_dust = ((GM/(r_vector*AU))**0.5)/1000
        v_rel_phi = np.abs(v_phi_vector-v_phi_dust)
        v_rel_tot = np.sqrt(v_rel_phi**2 + v_r_vector**2)

        ax.plot(r_vector,v_rel_tot,
                label=f"peri {peri}",c=f"C{i}")
        # ax.plot(r_vector,v_phi_dust/1000,label=None,ls="dashed",c=f"C{i}")

    ax.set_xlim(rmin,rmax)
    #ax.set_xscale("log")
    #ax.set_yscale("log")
    ax.set_xlabel("Heliocentric distance [AU]")
    ax.set_ylabel("Relative velocity [km/s]")
    ax.legend()
    plt.show()


def show_slopes_panels(ec=1e-4,
                       incl=1e-4,
                       retro=1e-4,
                       beta=0,
                       gamma=-1.3,
                       vexp=1,
                       loc=None,
                       name="compensated_model_test",
                       rmin=0.15,
                       rmax=0.5,
                       compensation=-2.5,
                       peris=[1,4,6,8,10],
                       att=None,
                       att_values=[0],
                       overplot=None):
    """
    A plot of 5 panels, one for each orbital group, showing the 
    compensated slope of flux in the outbound leg of an orbit.

    Parameters
    ----------
    ec : float, optional
        Eccentricity. The default is 1e-4.
    incl : float, optional
        Inlicnation in degrees. The default is 1e-4.
    retro : float, optional
        Retrograde fraction of the grains. The default is 1e-4.
    beta : float, optional
        RAdiation pressure parameter. The default is 0.
    gamma : float, optional
        Number density scaling. The default is -1.3.
    vexp : float, optional
        The exponent on speed. The default is 1.
    loc : str, optional
        Figures folder. The default is None.
    name : str, optional
        File name. The default is "compensated_model_test".
    rmin : float, optional
        Min AU to include. The default is 0.15.
    rmax : float, optional
        Max AU to include. The default is 0.5.
    compensation : float, optional
        The exponent to compensate with. The default is -2.5.
    peris : list of int, optional
        The orbits to show. The default is [1,4,6,8,10].
    att : str, optional
        The name of the attribute to vary. The default is None.
    att_values : list of float, optional
        The values of att to compare. The default is [0].
    overplot : list of float, optional
        The list of parameters: [ec, incl, retro, beto, gamma, vexp]. 
        The default is None.

    Raises
    ------
    Exception
        If the att is an unknown str.

    Returns
    -------
    None.

    """

    fig,ax = plt.subplots(5,figsize=(4,4))
    for i,peri in enumerate(peris):
        enc = encounter_group(peri)
        data = construct_perihel_n(peri,days=20,step_hours=8,
                                   outbound_only=True)
        r_vector = data["r_sc"].to_numpy()
        v_r_vector = data["v_sc_r"].to_numpy()
        v_phi_vector = (data["v_sc_t"].to_numpy())
        S_front_vector = data["area_front"].to_numpy()
        S_side_vector = data["area_side"].to_numpy()
        midvals = []
        if att is None:
            att_values = [0]
        lines=["-k",":k","--k","-grey",":grey","--grey"]
        for j,value in enumerate(att_values):
            if att == "ec":
                ec = value
                legend_label = lambda v: r"$e=\,$"+f"{np.around(v,2)}"
            elif att == "incl":
                incl = value
                legend_label = lambda v: r"$\theta=\,$"+f"{np.around(v,2)}"
            elif att == "retro":
                retro = value
                legend_label = lambda v: r"$rp=\,$"+f"{np.around(v,2)}"
            elif att == "beta":
                beta = value
                legend_label = lambda v: r"$\beta=\,$"+f"{v}"
            elif att == "gamma":
                gamma = value
                legend_label = lambda v: r"$\gamma=\,$"+f"{v}"
            elif att == "vexp":
                vexp = value
                legend_label = lambda v: r"$\epsilon=\,$"+f"{v}"
            elif att is None:
                pass
                legend_label = lambda v: None
            else:
                raise Exception(
                    'bad att, allowed: ["ec","inlc","retro","beta",'
                    +'"gamma","vexp",None]')

            flux = bound_flux_vectorized(
                r_vector = r_vector,
                v_r_vector = v_r_vector,
                v_phi_vector = v_phi_vector,
                S_front_vector = S_front_vector,
                S_side_vector = S_side_vector,
                ex = ec,
                incl = incl,
                retro = retro,
                beta = beta,
                gamma = gamma,
                velocity_exponent = vexp,
                n = 7e-9)
    
            comp_flux = flux/(r_vector**compensation)*10**4
            midvals.append(comp_flux[np.argmin(np.abs(r_vector-0.325))])
            ax[i].plot(r_vector,
                       comp_flux*(2.25/midvals[-1]),
                       lines[j],label=legend_label(value))

        if overplot is not None:
            flux = bound_flux_vectorized(
                r_vector = r_vector,
                v_r_vector = v_r_vector,
                v_phi_vector = v_phi_vector,
                S_front_vector = S_front_vector,
                S_side_vector = S_side_vector,
                ex = overplot[0],
                incl = overplot[1],
                retro = overplot[2],
                beta = overplot[3],
                gamma = overplot[4],
                velocity_exponent = overplot[5],
                n = 7e-9)
            comp_flux = flux/(r_vector**compensation)*10**4
            ax[i].plot(r_vector,
                       comp_flux*(2.25
                           /comp_flux[np.argmin(np.abs(r_vector-0.325))]),
                       "--k")

        ax[i].set_xlim(0.15,0.5)
        ax[i].grid(color='lightgrey', linestyle='-',
                   linewidth=1, axis="y", which = "both", alpha=0.5)
        ax[i].set_ylabel(f"Group {enc}")
        ax[i].yaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
        if i!=4:
            ax[i].xaxis.set_ticklabels([])
        ax[i].set_ylim(0,4.8)

    if att is not None:
        ax[0].legend(loc=2,ncol=3,fontsize="small",frameon=True,edgecolor="w")
    #ax[0].set_title("All the fluxes, "
    #                +f"compensated by $R^{{{compensation}}}$")
    ax[4].set_xlabel("R [AU]")
    ax[2].set_ylabel(f"Flux [$C \cdot s^{{-1}} AU^{{{-compensation}}}$] \n"+
                     "Group 3",linespacing=2)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    if loc is not None:
        plt.savefig(loc+name+".pdf",format="pdf")
    plt.show()


def show_perihelia_panels(ec=1e-4,
                          incl=1e-4,
                          retro=1e-4,
                          beta=0,
                          gamma=-1.3,
                          vexp=1,
                          loc=None,
                          name="perihelia_model_test",
                          days=7,
                          peris=[1,4,6,8,10],
                          marker_dist=[0.2,0.16,0.14,0.11,0.1],
                          marker_time=[74,50.4,48.2,33.1,30.5],
                          att=None,
                          att_values=[0],
                          overplot=None):
    """
    A plot of 5 panels, one for each orbital group, showing the 
    near perihela flux of the orbits.

    Parameters
    ----------
    ec : float, optional
        Eccentricity. The default is 1e-4.
    incl : float, optional
        Inlicnation in degrees. The default is 1e-4.
    retro : float, optional
        Retrograde fraction of the grains. The default is 1e-4.
    beta : float, optional
        RAdiation pressure parameter. The default is 0.
    gamma : float, optional
        Number density scaling. The default is -1.3.
    vexp : float, optional
        The exponent on speed. The default is 1.
    loc : str, optional
        Figures folder. The default is None.
    name : str, optional
        File name. The default is "perihelia_model_test".
    days : int, optional
        How many days to compute and show. The default is 7.
    peris : list of int, optional
        The orbits to show. The default is [1,4,6,8,10].
    marker_dist : list of float, optional
        Where in AU to place vlines. 
        The default is [0.2,0.16,0.14,0.11,0.1].
    att : str, optional
        The name of the attribute to vary. The default is None.
    att_values : list of float, optional
        The values of att to compare. The default is [0].
    overplot : list of float, optional
        The list of parameters: [ec, incl, retro, beto, gamma, vexp]. 
        The default is None.


    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    None.

    """

    fig,ax = plt.subplots(5,figsize=(4,4))
    for i,peri in enumerate(peris):
        enc = encounter_group(peri)
        data = construct_perihel_n(peri,days=days,step_hours=4)
        r_vector = data["r_sc"].to_numpy()
        hours_vector = (data["jd"].to_numpy()
                        - construct_perihel_n(peri,0)["jd"][0])*24
        v_r_vector = data["v_sc_r"].to_numpy()
        v_phi_vector = (data["v_sc_t"].to_numpy())
        S_front_vector = data["area_front"].to_numpy()
        S_side_vector = data["area_side"].to_numpy()
        midvals = []
        if att is None:
            att_values = [0]
        lines=["-k",":k","--k","-grey",":grey","--grey"]
        for j,value in enumerate(att_values):
            if att == "ec":
                ec = value
                legend_label = lambda v: r"$e=\,$"+f"{np.around(v,2)}"
            elif att == "incl":
                incl = value
                legend_label = lambda v: r"$\theta=\,$"+f"{np.around(v,2)}"
            elif att == "retro":
                retro = value
                legend_label = lambda v: r"$rp=\,$"+f"{np.around(v,2)}"
            elif att == "beta":
                beta = value
                legend_label = lambda v: r"$\beta=\,$"+f"{v}"
            elif att == "gamma":
                gamma = value
                legend_label = lambda v: r"$\gamma=\,$"+f"{v}"
            elif att == "vexp":
                vexp = value
                legend_label = lambda v: r"$\epsilon=\,$"+f"{v}"
            elif att is None:
                pass
                legend_label = lambda v: None
            else:
                raise Exception(
                    'bad att, allowed: ["ec","inlc","retro","beta",'
                    +'"gamma","vexp",None]')

            flux = bound_flux_vectorized(
                r_vector = r_vector,
                v_r_vector = v_r_vector,
                v_phi_vector = v_phi_vector,
                S_front_vector = S_front_vector,
                S_side_vector = S_side_vector,
                ex = ec,
                incl = incl,
                retro = retro,
                beta = beta,
                gamma = gamma,
                velocity_exponent = vexp,
                n = 7e-9)

            ax[i].plot(hours_vector,flux/np.max(flux),lines[j],
                       label=legend_label(value))

        if overplot is not None:
            flux = bound_flux_vectorized(
                r_vector = r_vector,
                v_r_vector = v_r_vector,
                v_phi_vector = v_phi_vector,
                S_front_vector = S_front_vector,
                S_side_vector = S_side_vector,
                ex = overplot[0],
                incl = overplot[1],
                retro = overplot[2],
                beta = overplot[3],
                gamma = overplot[4],
                velocity_exponent = overplot[5],
                n = 7e-9)
            ax[i].plot(hours_vector,flux/np.max(flux),"--k")

        ax[i].set_xlim(-days*24,days*24)
        hour_threshold = np.abs(hours_vector[
            np.argmin(np.abs(r_vector-marker_dist[i]))])
        ax[i].set_ylabel(f"Group {enc}")
        ax[i].yaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
        if i!=4:
            ax[i].xaxis.set_ticklabels([])
        toplim = 1.25*ax[i].get_ylim()[1]
        ax[i].vlines([marker_time[i],-marker_time[i]],0,toplim,
                     ls="dashed",color="k")
        ax[i].text(marker_time[i]+25, (0.85-0.6*(i<2))*toplim,
                   f"${marker_dist[i]} \, AU$",
                   horizontalalignment='left',
                   verticalalignment='top')
        ax[i].set_ylim(bottom=0,top=toplim)
        ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    if att is not None:
        ax[0].legend(loc=2,ncol=1,fontsize="small",frameon=True,edgecolor="w")
    #ax[0].set_title("All the fluxes, "
    #                +f"compensated by $R^{{{compensation}}}$")
    ax[4].set_xlabel("Time since perihelia [h]")
    ax[2].set_ylabel("Flux [$C \cdot s^{{-1}}$] \n"+
                     "Group 3",linespacing=2)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    if loc is not None:
        plt.savefig(loc+name+".pdf",format="pdf")
    plt.show()


def estimate_slope(ec=1e-4,
                   incl=1e-4,
                   retro=1e-4,
                   beta=0,
                   gamma=-1.3,
                   vexp=1,
                   peris=[4,6,8,10],
                   r1=0.25,
                   r2=0.4):
    """
    Estimates the slope, given the parameters. Be careful, this assumes 
    that the profile is power-law, which it almost never is.

    Parameters
    ----------
    ec : float, optional
        Eccentricity. The default is 1e-4.
    incl : float, optional
        Inlicnation in degrees. The default is 1e-4.
    retro : float, optional
        Retrograde fraction of the grains. The default is 1e-4.
    beta : float, optional
        RAdiation pressure parameter. The default is 0.
    gamma : float, optional
        Number density scaling. The default is -1.3.
    vexp : float, optional
        The exponent on speed. The default is 1.
    peris : list of int, optional
        The perihelia to investigate. The default is [4,6,8,10].
    r1 : float, optional
        The fisrt reference AU. The default is 0.25.
    r2 : float, optional
        The second reference AU. The default is 0.4.

    Returns
    -------
    float
        The implied slope.

    """
    slopes = np.zeros(len(peris))
    for i,peri in enumerate(peris):
        data = construct_perihel_n(peri,days=13,step_hours=8,
                                   outbound_only=True)
        r_vector = data["r_sc"].to_numpy()
        v_r_vector = data["v_sc_r"].to_numpy()
        v_phi_vector = (data["v_sc_t"].to_numpy())
        S_front_vector = data["area_front"].to_numpy()
        S_side_vector = data["area_side"].to_numpy()

        flux = bound_flux_vectorized(
            r_vector = r_vector,
            v_r_vector = v_r_vector,
            v_phi_vector = v_phi_vector,
            S_front_vector = S_front_vector,
            S_side_vector = S_side_vector,
            ex = ec,
            incl = incl,
            retro = retro,
            beta = beta,
            gamma = gamma,
            velocity_exponent = vexp,
            n = 7e-9)
    
        i1 = np.argmin(np.abs(r_vector-r1))
        i2 = np.argmin(np.abs(r_vector-r2))
        slopes[i] = np.log(flux[i1]/flux[i2])/np.log(r_vector[i1]/r_vector[i2])
    return np.mean(slopes)


def heatmap(x, y, values,
            xlabel="xlabel",
            ylabel="ylabel",
            cmap="Greys"):
    """
    General purpose usef friendly heatmap plotting wrapper.

    Parameters
    ----------
    x : np.array 1D
        x-axis varaible values.
    y : np.array 1D
        y-axis variable values.
    values : np.array 2D
        values to show.
    xlabel : str, optional
        x-axis label. The default is "xlabel".
    ylabel : str, optional
        y-axis label. The default is "ylabel".
    cmap : str, optional
        name of the pyplot colormap. The default is "Greys".

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots()
    im = ax.imshow(np.flip(values,axis=1).transpose(),
                   extent=(min(x),max(x),
                           min(y),max(y)),
                   aspect='auto',
                   cmap=cmap)
    cbar = fig.colorbar(im)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.show()


def show_slope_map(vexps=np.linspace(1.7,2.6,24),
                   gammas=np.linspace(-2,-1.3,22),
                   **kwargs):
    slopes = np.zeros((len(vexps),len(gammas)))
    with tqdm(total=len(vexps)*len(gammas)) as pbar:
        for i,vexp in enumerate(vexps):
            for j,gamma in enumerate(gammas):
                slopes[i,j] = estimate_slope(vexp=vexp,gamma=gamma,**kwargs)
                pbar.update(1)

    heatmap(vexps,gammas,np.abs(slopes+2.5),
            xlabel=r"$\epsilon$",
            ylabel=r"$\gamma$")

    heatmap(gammas,vexps,np.abs(slopes.transpose()+2.5),
            xlabel=r"$\gamma$",
            ylabel=r"$\epsilon$")

    return slopes


def empirical_relation(gammas,vexps,
                       results1,results2,
                       target=-2.5,
                       order=2,
                       loc=os.path.join(figures_location,
                                        "perihelia","ddz_profile",""),
                       name="empirical_slope"):

    fig, ax = plt.subplots()

    for j,results in enumerate([results1,results2]):
        badness = np.abs(results-target)
        x_to_fit = []
        y_to_fit = []
        for i,ix in enumerate(gammas):
            x_to_fit.append(ix)
            y_to_fit.append(vexps[np.argmin(badness[:,i])])
        fitcoeff = np.polyfit(x_to_fit,y_to_fit,order)
        fit = np.poly1d(fitcoeff)
        if not j:
            ax.plot(x_to_fit,fit(x_to_fit),
                    c="k",ls="solid",label="Base")
        else:
            ax.plot(x_to_fit,fit(x_to_fit),
                    c="k",ls="dashed",label="Non-circular")

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\epsilon$")
    ax.legend()
    ax.set_xlim(min(gammas),max(gammas))
    fig.tight_layout()
    if loc is not None:
        plt.savefig(loc+name+".pdf",format="pdf")
    plt.show()


def analyse_slopes(vexps=np.linspace(1.5,3,16),
                   gammas=np.linspace(-2,-0.999,16)):
    """
    creates the paper plot for epsilon = f(gamma)

    Parameters
    ----------
    vexps : np.array of float, optional
        the vexps of interest. The default is np.linspace(1.7,2.6,16).
    gammas : np.array of float, optional
        the gammas of interest. The default is np.linspace(-2,-1.3,10).

    Returns
    -------
    None.

    """
    slopes_base = show_slope_map(vexps,gammas)
    slopes_upper = show_slope_map(vexps,gammas,
                                  ec=0.5,
                                  incl=45,
                                  retro=0.1,
                                  beta=0.5)
    empirical_relation(gammas,vexps,slopes_base,slopes_upper)



#%%
if __name__ == "__main__":
    density_scaling_multiple()



#%%
if __name__ == "__main__":

    analyse_slopes()



#%%
if __name__ == "__main__":

    loc = os.path.join(figures_location,"perihelia","ddz_profile","")

    #The vanilla case
    show_slopes_panels(loc=loc,
                       name="compensated_vanilla_model")
    
    #Different speed exponents
    show_slopes_panels(att="vexp",att_values=[1,2,3],
                       loc=loc,
                       name="compensated_model_vexp")

    #Different spatial scaling
    show_slopes_panels(att="gamma",att_values=[-1.3,-2,-3],
                       loc=loc,
                       name="compensated_model_gamma")
    
    #Different beta
    show_slopes_panels(att="beta",att_values=[0,0.25,0.5],
                       loc=loc,
                       name="compensated_model_beta")
    
    #Different eccentricity
    show_slopes_panels(att="ec",att_values=[1e-4,0.25,0.5],
                       loc=loc,
                       name="compensated_model_ec")
    
    #Different inclination
    show_slopes_panels(att="incl",att_values=[1e-4,15,45],
                       loc=loc,
                       name="compensated_model_incl")
    
    #Different retrograde
    show_slopes_panels(att="retro",att_values=[1e-4,0.02,0.1],
                       loc=loc,
                       name="compensated_model_retro")

    #Vanilla "-" and the best without adjustment to the exponents "--"
    show_slopes_panels(overplot=[0.5,45,0.1,0.5,-1.3,1],
                       loc=loc,
                       name="compensated_insufficient_model")

    #Vanilla "-" and Szalay "--"
    show_slopes_panels(overplot=[1e-4,1e-4,1e-4,0,-1.3,4.15],
                       loc=loc,
                       name="compensated_szalay")

    #Vanilla "-" and the best without adjustment to the exponents "--"
    show_slopes_panels(overplot=[0.5,45,0.1,0.5,-1.5,2],
                       loc=loc,
                       name="compensated_sufficient_model")

    #A viable option
    show_slopes_panels(overplot=[0.3,25,0.05,0.25,-1.8,2],
                       loc=loc,
                       name="compensated_viable_model")

    #Another viable option
    show_slopes_panels(overplot=[0.1,10,0.03,0.05,-1.9,2],
                       loc=loc,
                       name="compensated_viable_model")




#%%
if __name__ == "__main__":

    loc = os.path.join(figures_location,"perihelia","ddz_profile","")

    #The vanilla case
    show_perihelia_panels(loc=loc,
                          name="perihelia_model_vanilla")

    #Different speed exponents
    show_perihelia_panels(att="vexp",att_values=[1,2,3],
                       loc=loc,
                       name="perihelia_model_vexp")

    #Different spatial scaling
    show_perihelia_panels(att="gamma",att_values=[-1.3,-2,-3],
                       loc=loc,
                       name="perihelia_model_gamma")
    
    #Different beta
    show_perihelia_panels(att="beta",att_values=[0,0.25,0.5],
                       loc=loc,
                       name="perihelia_model_beta")
    
    #Different eccentricity
    show_perihelia_panels(att="ec",att_values=[1e-4,0.25,0.5],
                       loc=loc,
                       name="perihelia_model_ec")

    #Different inclination
    show_perihelia_panels(att="incl",att_values=[1e-4,15,45],
                       loc=loc,
                       name="perihelia_model_incl")
    
    #Different retrograde
    show_perihelia_panels(att="retro",att_values=[1e-4,0.02,0.1],
                       loc=loc,
                       name="perihelia_model_retro")

    #The original viable option
    show_perihelia_panels(overplot=[0.3,25,0.05,0.25,-1.8,2],
                       loc=loc,
                       name="perihelia_viable_model")

    #Another viable option
    show_perihelia_panels(overplot=[0.1,10,0.03,0.05,-1.9,2],
                       loc=loc,
                       name="perihelia_viable_model")


#%%
if __name__ == "__main__":

    loc = os.path.join(figures_location,"perihelia","eccentricity","")
    for ec in [1e-4]:
        for incl in [1e-4]:
            for retro in [1e-4]:
                for beta in [0]:
                    for vexp in [1]:
                        #density_scaling(ex=ex,size=500000,loc=loc)
                        for n_peri in [1,4,6,8,10]:
                            plot_single(data=construct_perihel_n(n_peri),
                                        ec=ec,
                                        incl=incl,
                                        retro=retro,
                                        beta=beta,
                                        vexp=vexp,
                                        loc=loc,
                                        peri=n_peri)

    loc = os.path.join(figures_location,"perihelia","comparison","")
    for peri,ymax in zip([4,10],[0.035,0.12]):
        plot_compare(att="ec",att_values=[1e-4,0.1,0.2,0.3,0.4,0.5],
                     peri=peri,loc=loc,ymax=ymax)
        plot_compare(att="incl",att_values=[1e-4,15,30,45],
                     peri=peri,loc=loc,ymax=ymax)
        plot_compare(att="retro",att_values=[1e-4,0.03,0.1,0.2],
                     peri=peri,loc=loc,ymax=ymax)
        plot_compare(att="beta",att_values=[0,0.1,0.3,0.5],
                     peri=peri,loc=loc,ymax=ymax)
        plot_compare(att="vexp",att_values=[1,1.1,2,4.15],
                     peri=peri,loc=loc,ymax=ymax)


    show_relative_velocity(peris=[1,4,6,8,10])



