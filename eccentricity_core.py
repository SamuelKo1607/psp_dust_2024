import numpy as np
from numba import jit

from conversions import GM
from conversions import AU

Size=100000

@jit
def azimuthal_flux(r_si,v_phi_si,
                   ex,mu,gamma):
    """
    The azimuthal component of the bound dust flux.

    Parameters
    ----------
    r : float
        SC heliocentric distance [m].
    v_phi : float
        SC heliocentric azimuthal speed [m/s].
    e : float
        Dust eccentricity.
    mu : float
        Effective gravitational parameter, 
        acocunting for beta value.
    gamma : float
        Bound dust radial spatial density exponent. 

    Returns
    -------
    j_tot : float
        The total azimuthal flux, stupid unit, see bound_flux.

    """
    def indefinite(x):
        a = (x**(gamma+2))/(gamma+2)
        b = v_phi_si*(x**(gamma+1))/(gamma+1)
        return a-b

    j_plus = ( indefinite(np.max(np.array([v_phi_si,
                                  (mu*(1+ex)/r_si)**0.5])))
              - indefinite(np.max(np.array([v_phi_si,
                                   (mu*(1-ex)/r_si)**0.5]))) )

    j_minus = ( indefinite(np.min(np.array([v_phi_si,
                                   (mu*(1+ex)/r_si)**0.5])))
               - indefinite(np.min(np.array([v_phi_si,
                                    (mu*(1-ex)/r_si)**0.5]))) )

    j_tot = j_plus - j_minus
    return j_tot

@jit
def azimuthal_flux_inclination(r_si,v_phi_si,
                               ex,incl,mu,gamma):
    """
    The azimuthal component of the bound dust flux, 
    assuming PSP is a cuboid.

    Parameters
    ----------
    r : float
        SC heliocentric distance [m].
    v_phi : float
        SC heliocentric azimuthal speed [m/s].
    ex : float
        Dust eccentricity.
    incl : float
        Dust inclination [deg].
    mu : float
        Effective gravitational parameter, 
        acocunting for beta value.
    gamma : float
        Bound dust radial spatial density exponent. 

    Returns
    -------
    j_tot : float
        The total azimuthal flux, stupid unit, see bound_flux.

    """
    def indefinite_we(x):
        a = (x**(gamma+2))/(gamma+2)*np.cos(np.deg2rad(incl))
        b = v_phi_si*(x**(gamma+1))/(gamma+1)
        return a-b

    def indefinite_n(x):
        a = (x**(gamma+2))/(gamma+2)*np.sin(np.deg2rad(incl))
        return a

    j_w = ( indefinite_we(np.max(np.array([v_phi_si/np.cos(np.deg2rad(incl)),
                                 (mu*(1+ex)/r_si)**0.5])) )
          - indefinite_we(np.max(np.array([v_phi_si/np.cos(np.deg2rad(incl)),
                                 (mu*(1-ex)/r_si)**0.5])) ) )

    j_e = ( indefinite_we(np.min(np.array([v_phi_si/np.cos(np.deg2rad(incl)),
                                 (mu*(1+ex)/r_si)**0.5])) )
          - indefinite_we(np.min(np.array([v_phi_si/np.cos(np.deg2rad(incl)),
                                 (mu*(1-ex)/r_si)**0.5])) ) )

    j_n = ( indefinite_n((mu*(1+ex)/r_si)**0.5)
          - indefinite_n((mu*(1-ex)/r_si)**0.5) )

    j_tot = j_w - j_e + j_n
    return j_tot

@jit
def azimuthal_flux_inclination_cyllinder(r_si,v_phi_si,
                                         ex,incl,retro,
                                         mu,gamma,
                                         velocity_exponent=1,
                                         size=Size):
    """
    The azimuthal component of the bound dust flux, 
    assuming PSP is a cylinder.

    Parameters
    ----------
    r : float
        SC heliocentric distance [m].
    v_phi : float
        SC heliocentric azimuthal speed [m/s].
    ex : float
        Dust eccentricity.
    incl : float
        Dust inclination [deg].
    retro : float
        Retrograde fraction (0-1).
    mu : float
        Effective gravitational parameter, 
        acocunting for beta value.
    gamma : float
        Bound dust radial spatial density exponent.
    velocity_exponent : float, optional
        The exponent on the relative speed between dust and SC. 
        If 1, then velocity does not play a role, and only detection metters. 
        The default is 1. Szalay et al 2021 used 4.15.
    size : int, optional
        The number of MC integration points.
        The default is 100000.

    Returns
    -------
    j_azim_cyl : float
        The total azimuthal flux, stupid unit, see bound_flux.

    """
    lo = (mu*(1-ex)/r_si)**0.5
    hi = (mu*(1+ex)/r_si)**0.5

    V = hi - lo
    x = np.random.uniform(lo,hi,size)
    x *= (-1)**((np.arange(size)/size)<retro)

    v_cyl = (  (     v_phi_si * np.sin(np.deg2rad(incl)) )**2
             + ( x - v_phi_si * np.cos(np.deg2rad(incl)) )**2 )**0.5

    j_azim_cyl = V * np.average((v_cyl**velocity_exponent)
                                * (np.abs(x))**gamma)

    return j_azim_cyl

@jit
def radial_flux(r_si,v_r_si,
                ex,mu,gamma,
                velocity_exponent=1,
                size=Size):
    """
    The radial component of the bound dust flux.

    Parameters
    ----------
    r_si : float
        SC heliocentric distance [m].
    v_r_si : float
        SC heliocentric radial speed [m/s].
    e : float
        Dust eccentricity.
    mu : float
        Effective gravitational parameter, 
        acocunting for beta value.
    gamma : float
        Bound dust radial spatial density exponent. 
    velocity_exponent : float, optional
        The exponent on the relative speed between dust and SC. 
        If 1, then velocity does not play a role, and only detection metters. 
        The default is 1. Szalay et al 2021 used 4.15.
    size : int, optional
        The number of MC integration points.
        The default is 100000.

    Returns
    -------
    j_tot : float
        The total radial flux, stupid unit, see bound_flux.

    """
    lo = (mu*(1-ex)/r_si)**0.5
    hi = (mu*(1+ex)/r_si)**0.5

    V = hi - lo
    x = np.random.uniform(lo,hi,size) # v_dust
    good_chunk = ( ((ex**2-1)*mu**2)
                   +(2*mu*(x**2)*r_si)
                   -((x**4)*(r_si**2))
                  )**0.5/(x*r_si)

    j_plus_pre = V * np.average(
         (x**(gamma))*((np.abs( - good_chunk - v_r_si ))**velocity_exponent)
         * ( ( - good_chunk - v_r_si)>0 ) )
    j_plus_post = V * np.average(
         (x**(gamma))*((np.abs( + good_chunk - v_r_si ))**velocity_exponent)
         * ( ( + good_chunk - v_r_si)>0 ) )
    j_minus_pre = V * np.average(
         (x**(gamma))*((np.abs( + good_chunk + v_r_si ))**velocity_exponent)
         * ( ( + good_chunk + v_r_si)>0 ) )
    j_minus_post = V * np.average(
         (x**(gamma))*((np.abs( - good_chunk + v_r_si ))**velocity_exponent)
         * ( ( - good_chunk + v_r_si)>0 ) )

    j_tot = 0.5 * (j_plus_pre
                   + j_plus_post
                   + j_minus_pre
                   + j_minus_post)
    return j_tot

@jit
def bound_flux(r,v_r,v_phi,
               S_front,
               S_side,
               ex=1e-5,
               incl=1e-5,
               retro=1e-10,
               beta=0,
               gamma=-1.3,
               velocity_exponent=1,
               n=1e-8):
    """
    The wrapper for the total bound flux observed, 
    given the dust parameters and the sc state.

    Parameters
    ----------
    r : float
        SC heliocentric distance [AU].
    v_r : float
        SC heliocentric radial speed [km/s].
    v_phi : float
        SC heliocentric azimuthal speed [km/s].
    S_front : float
        SC front-side cross section [m^2].
    S_side : float
        SC lateral cross section [m^2].
    ex : float, optional
        Dust eccentricity. The default is 1e-5.
    incl : float, optional
        Inclination. The default is 1e-5.
    retro : float, optional
        Retrograde fraction (0-1). 1e-10.
    beta : float, optional
        Dust beta parameter. The default is 0.
    gamma : float, optional
        Bound dust radial spatial density exponent. 
        The default is -1.3.
    velocity_exponent : float, optional
        The exponent on the relative speed between dust and SC. 
        If 1, then velocity does not play a role, and only detection metters. 
        The default is 1. Szalay et al 2021 used 4.15.
    n : float, optional
        Dust number density [m^-3] at 1AU. 
        The default is 1e-8.

    Returns
    -------
    total_flux : float
        The total bound dust flux, [s^-1].
        
    """

    r_si = r * AU #[m]
    v_r_si = v_r * 1000 #[m/s]
    v_phi_si = v_phi * 1000 #[m/s]

    mu = (1-beta)*GM
    C = n * ((r_si/AU)**gamma) * ((AU/mu)**((gamma+1)/2)) * ((gamma+1)/
            ((1+ex)**((gamma+1)/2) - (1-ex)**((gamma+1)/2)))

    normalize = (50000**(velocity_exponent-1))**(-1)

    total_flux = C * normalize * (
                  S_side * azimuthal_flux_inclination_cyllinder(
                      r_si,v_phi_si,
                      ex,incl,retro,
                      mu,gamma,
                      velocity_exponent=velocity_exponent)
                + S_front * radial_flux(
                    r_si,v_r_si,
                    ex,
                    mu,gamma,
                    velocity_exponent=velocity_exponent) )

    return total_flux

@jit
def bound_flux_vectorized(r_vector,v_r_vector,v_phi_vector,
                          S_front_vector,
                          S_side_vector,
                          ex=1e-5,
                          incl=1e-5,
                          retro=1e-10,
                          beta=0,
                          gamma=-1.3,
                          velocity_exponent=1,
                          n=1e-8):
    """
    A vectorizer for bound_flux function.

    Parameters
    ----------
    r : np.array of float
        SC heliocentric distance [AU].
    v_r : np.array of float
        SC heliocentric radial speed [km/s].
    v_phi : np.array of float
        SC heliocentric azimuthal speed [km/s].
    S_front : np.array of float
        SC front-side cross section [m^2].
    S_side : np.array of float
        SC lateral cross section [m^2].
    ex : float, optional
        Dust eccentricity. The default is 0.
    incl : float, optional
        Inclination. The default is 1e-5.
    retro : float, optional
        Retrograde fraction (0-1). 1e-10.
    beta : float, optional
        Dust beta parameter. The default is 0.
    gamma : float, optional
        Bound dust radial spatial density exponent. 
        The default is -1.3.
    velocity_exponent : float, optional
        The exponent on the relative speed between dust and SC. 
        If 1, then velocity does not play a role, and only detection metters. 
        The default is 1. Szalay et al 2021 used 4.15.
    n : float, optional
        Dust number density [m^-3] at 1AU. 
        The default is 1e-8.

    Returns
    -------
    flux_vector : np.array of float
        The bound dust vector, as encountered along the ephemeris.
    
    """

    flux_vector = np.zeros(0)
    for r,v_r,v_phi,S_front,S_side in zip(r_vector,
                                          v_r_vector,
                                          v_phi_vector,
                                          S_front_vector,
                                          S_side_vector):
        flux_vector = np.append(flux_vector,
                                bound_flux(r,
                                           v_r,
                                           v_phi,
                                           S_front,
                                           S_side,
                                           ex,incl,retro,beta,gamma,
                                           float(velocity_exponent),n))
    return flux_vector


@jit
def r_smear_prob(r,r_peri,ex):
    """
    The pdf of where the grain will be found (r) given its perihel (r_peri)
    and eccentricity (ex).

    Parameters
    ----------
    r : float
        Heliocentric distance [AU], the indep. variable of the pdf.
    r_peri : float
        Perihelion distance [AU].
    ex : float
        Eccentricity.

    Returns
    -------
    float
        The pdf at the ditance r.

    """
    r_aph = ((1+ex)/(1-ex))*r_peri
    if r<r_peri or r>r_aph:
        return 0
    else:
        return np.sqrt((2/r-(1-ex)/r_peri-((1+ex)*r_peri)/(r**2)))**(-1)


@jit
def r_smearing(r_peri,
               ex,
               size=10,
               burnin=10):
    """
    An MCMC procedure to provide a sample of actual immediate 
    heliocentric distances, given the perihelion and eccentricity.

    Parameters
    ----------
    r_peri : float
        Perihelion distance [AU].
    ex : float
        Eccentricity [1].
    size : int, optional
        The number of samples needed. The default is 10.
    burnin : int, optional
        The burnin length. The default is 10.

    Returns
    -------
    sampled : np.array of float
        The sampled heliocentric distances.

    """
    r_aph = ((1+ex)/(1-ex))*r_peri
    r = np.random.uniform(r_peri,r_aph)
    proposal_width = (r_aph - r_peri)/5
    sampled=np.zeros(0)
    for i in range(size+burnin):
        sampled = np.append(sampled,r)
        r_proposal = r+np.random.uniform(-proposal_width/2,proposal_width/2)
        goodness = (r_smear_prob(r_proposal,r_peri,ex)
                    /r_smear_prob(r,r_peri,ex))
        if goodness > np.random.random():
            r = r_proposal
    return sampled[burnin:]

@jit
def acceleration(r):
    """
    The Sun is assumed in [0,0,0], stationary. 

    Parameters
    ----------
    r : np.array, 1D of length 3
        postion [m]

    Returns
    -------
    a : np.array, 1D of length 3
        acceleration [m/s**2]

    """
    unit_r = r/np.sqrt(np.sum(r**2))
    a = -unit_r * GM/np.sum(r**2)
    return a

@jit
def verlet_step(r,
                v,
                a,
                dt):
    """
    One inegration step, speed Verlet alg.

    Parameters
    ----------
    r : np.array, 1D of length 3
        postion [m]
    v : np.array, 1D of length 3
        velocity [m/s]
    a : np.array, 1D of length 3
        acceleration [m/s**2]
    dt : float
        time step [s]

    Returns
    -------
    r_new : np.array, 1D of length 3
        new postion [m]
    v_new : np.array, 1D of length 3
        new velocity [m/s]
    a_new : np.array, 1D of length 3
        new acceleration [m/s**2]

    """
    r_new = r + v*dt + 0.5*a*dt**2
    a_new = acceleration(r_new)
    v_new = v + 0.5*(a + a_new)*dt
    return r_new, v_new, a_new


def velocity_verlet(r_peri=2.48e10,
                    v_peri=9.5e4,
                    days=140,
                    step_hours=1):
    """
    A simple implementation of velocity Verlet alg. The sun is assumed 
    in [0,0,0] point, the perihelion aligned with the x-axis.

    Parameters
    ----------
    r_peri : float, optional
        The perihelion distance, [m]. The default is 2.48e10.
    v_peri : float, optional
        The perihelion speed, [m/s]. The default is 9.5e4.
    days : float, optional
        The number of days after the perihelion. The default is 140.
    step_hours : float, optional
        The time step [h]. The default is 1.

    Returns
    -------
    r : np.array of float, shape (n,3)
        The calculated position vectors.
    v : np.array of float, shape (n,3)
        The calculated positions.

    """
    r = np.zeros(shape=(0,3))
    v = np.zeros(shape=(0,3))
    r = np.vstack((r,np.array([r_peri,0,0])))
    v = np.vstack((v,np.array([0,v_peri,0])))
    a = acceleration(r[-1,:])
    for i in range(int(days*24/step_hours)):
        r_new, v_new, a = verlet_step(r[-1,:],v[-1,:],a,step_hours*3600)
        r = np.vstack((r,r_new))
        v = np.vstack((v,v_new))
    return r,v










"""
i = 150

r = r_vector[i]
v_r = v_r_vector[i]
v_phi = v_phi_vector[i]
S_front = S_front_vector[i]
S_side = S_side_vector[i]

bound_flux(r,
           v_r,
           v_phi,
           S_front,
           S_side,
           e,beta,gamma,A)

"""








