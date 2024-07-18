#import numpy as np
import cupy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi'] = 600

def mean_charge(# km, km/s:
                m,v,
                # as in Shen's PhD thesis for PSP/TPS:
                C=4.3e-2,
                a=1,
                b=3.46):
    Q = C*np.outer((m**a),(v**b))
    return Q #shape (len(m),len(v))


def cleanup():
    if np.__name__=="cupy":
        np._default_memory_pool.free_all_blocks()
        mempool = np.get_default_memory_pool()
        print(mempool.used_bytes()/(1024**3))
    else:
        pass


def compare(randomness=0):
    v = np.linspace(V_min,V_max,V_steps)
    m = (np.random.pareto(M_shape, M_size) + 1 ) * M_lower
    if randomness==0:
        over_threshold = mean_charge(m,v,b=B) > Q_min
    else:
        over_threshold = 10**(np.log10(mean_charge(m,v,b=B))+
                              np.random.normal(0,np.log10(randomness),
                                               size=(len(m),len(v)))
                              ) > Q_min
    detected = np.sum(over_threshold,axis=0)/M_size
    m = None
    over_threshold = None

    fig, ax = plt.subplots()
    if np.__name__=="cupy":
        ax.plot(np.asnumpy(v),np.asnumpy(detected),label="sim")
        ax.plot(np.asnumpy(v),
                np.asnumpy(np.min(detected)*(v/np.min(v))**(M_shape*B)),
                ls="dashed",label="ab={:.2f} slope ".format(M_shape*B))
    else:
        ax.plot(v,detected,label="sim")
        ax.plot(v,
                np.min(detected)*(v/np.min(v))**(M_shape*B),
                ls="dashed",label="ab={:.2f} slope ".format(M_shape*B))
    ax.set_xlabel("Impact speed [km/s]")
    ax.set_ylabel("Detected [1/1]")
    if randomness==0:
        fig.suptitle("No randomness")
    else:
        fig.suptitle(f"+- factor of {randomness}")
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(10,109)
    ax.set_xticks([10, 20, 30, 40, 50, 70, 100])
    ax.set_xticklabels([10, 20, 30, 40, 50, 70, 100])
    plt.show()


#%%
if __name__ == "__main__":

    # detection threshold
    Q_min = 1 * (1.602e-19 * 3e8) # in C, as a * of as in Szalay et al. 2021
    # charge generation eq.
    B = 3.46
    # dust speeds to consider
    V_min = 16
    V_max = 96
    V_steps = 32

    # reasonable masses
    M_lower = 1e-14 # r = 1 \mu m, if in kg
    M_shape = 5/6 # or 0.9, whatever. This is the slope of CDF
    M_size = 4 *1000000

    # if there is no randomness in the amount of generated charge
    compare(randomness=0)

    # if there is randomness of a factor of 2
    compare(randomness=2)

    # if there is randomness of a factor of 5
    compare(randomness=5)









