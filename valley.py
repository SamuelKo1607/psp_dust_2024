import os
import sys
import numpy as np
from multiprocessing import Pool
from functools import partial
import matplotlib as mpl
import matplotlib.pyplot as plt
from mcmc import load_data
from mcmc import log_prior
from mcmc import log_likelihood
from mcmc import proposal
from tqdm.auto import tqdm

from paths import figures_location

import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi'] = 600



def combinations(arr1,arr2):
    """
    Makes a list of all the possible combinations.

    Parameters
    ----------
    arr1 : iterable
        DESCRIPTION.
    arr2 : iterable
        DESCRIPTION.

    Returns
    -------
    pairs : list of tuples
        DESCRIPTION.

    """

    combs = []
    for item1 in arr1:
        for item2 in arr2:
            combs.append([item1,item2])
    return combs


def dummy(a,b,c,d):
    return a*b*c*d


def task(x): # x is a list of len 2
    return (x,dummy(x[0],x[1],c=0.5,d=2))


def step(theta,
         data,
         scale=0.075,
         family="normal",
         shield=1,
         vary_l_a=True,
         vary_l_b=True,
         vary_v_b_r=False,
         vary_e_v=False,
         vary_e_b_r=True,
         vary_shield_miss_rate=True,
         flat_prior=True):
    """
    Performs a step, returns either the old or the new theta.

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    npdata : TYPE
        DESCRIPTION.
    scale : float, optional
        The scale of the proposed change.
    shield : bool, optional
        Whether to assume PSP's heat shield different 
        from the rest of the SC. The default is True.
    vary_l_a : bool, optional
        Whether to vary this parameter. The default is True.
    vary_l_b : bool, optional
        Whether to vary this parameter. The default is True.
    vary_v_b_r : bool, optional
        Whether to vary this parameter. The default is False.
    vary_e_v : bool, optional
        Whether to vary this parameter. The default is False.
    vary_e_b_r : bool, optional
        Whether to vary this parameter. The default is True.
    vary_shield_miss_rate : bool, optional
        Whether to vary this parameter. The default is True.
    flat_prior : bool, optional
        whether to disregrad prior. The default is True.

    Returns
    -------
    theta : TYPE
        DESCRIPTION.
    change : bool
        Whether the new theta is actually new, or just the old one.

    """
    old_goodness = (log_likelihood(theta, data, shield)
                    + log_prior(theta)*(1-flat_prior))

    proposed_theta = proposal(theta,scale=scale,family=family,
                              vary_l_a=vary_l_a,
                              vary_l_b=vary_l_b,
                              vary_v_b_r=vary_v_b_r,
                              vary_e_v=vary_e_v,
                              vary_e_b_r=vary_e_b_r,
                              vary_shield_miss_rate=vary_shield_miss_rate)
    proposed_goodness = (log_likelihood(proposed_theta, data, shield)
                         + log_prior(proposed_theta)*(1-flat_prior))

    log_acc_ratio = proposed_goodness - old_goodness
    threshold = np.random.uniform(0,1)

    if np.exp(log_acc_ratio) > threshold:
        theta = proposed_theta

    return theta, proposed_goodness


def walk(nsteps,
         theta_start,
         data,
         stepscale=0.075,
         stepfamily="normal",
         stepshield=1):
    theta = theta_start
    record_goodness = -1e12
    for i in range(nsteps):
        theta, goodness = step(theta,
                               data,
                               scale=stepscale,
                               family=stepfamily,
                               shield=stepshield)
        record_goodness = max(goodness,record_goodness)
    return record_goodness


def trail(x,data,nsteps,stepscale): # x is a list of len 2, x[0] is v_b_r, x[1] is e_v
    record_goodness = walk(nsteps,
                           [7.79e-05, 5.88e-05, x[0], x[1], 0.075, 0.742],
                           data,
                           stepscale=stepscale)
    return (x,record_goodness)


def main(v_b_r_grid,
         e_v_grid,
         cores=8,
         nsteps=1000,
         stepscale=0.075):

    pairs = combinations(v_b_r_grid,e_v_grid)
    canvas = np.zeros((len(v_b_r_grid),len(e_v_grid)))

    data = load_data()
    local_trail = partial(trail,
                          data = data,
                          nsteps = nsteps,
                          stepscale = stepscale)

    with Pool(cores) as p:
        poolresults = list(tqdm(p.imap(local_trail, pairs),
                                total=len(pairs)))

    for item in poolresults:
        x = item[0]
        result = item[1]
        v_b_r_grid_index = np.argmin(np.abs(v_b_r_grid-x[0]))
        e_v_grid_index = np.argmin(np.abs(e_v_grid-x[1]))
        canvas[v_b_r_grid_index,e_v_grid_index] = result

    fig,ax = plt.subplots()
    cs = ax.contourf(canvas,cmap="cividis",
                     extent=[min(e_v_grid),max(e_v_grid),
                             min(v_b_r_grid),max(v_b_r_grid)],
                     levels=30)
    cbar = fig.colorbar(cs)
    cbar.set_label('MaxLogLik')
    ax.set_xlabel(r"$\epsilon_v [1]$")
    ax.set_ylabel(r"$v_{\beta,r} [km/s]$")
    fig.tight_layout()
    fig.savefig(os.path.join("data_synced","")+"valley"+".png",dpi=1200)
    fig.savefig(os.path.join("data_synced","")+"valley"+".pdf",dpi=1200)
    plt.close()
    return poolresults

#%%
if __name__ == "__main__":

    v_b_r_res = int(sys.argv[1])
    e_v_res = int(sys.argv[2])
    nsteps = int(sys.argv[3])

    results = main(v_b_r_grid = np.linspace(10, 200, num=v_b_r_res),
                   e_v_grid = np.linspace(1e-3, 3.5, num=e_v_res),
                   nsteps = nsteps)

