import numpy as np
import csv
from scipy import interpolate
from scipy.signal import argrelextrema
import pickle
import os

from conversions import tt2000_to_jd

au = 149597870.7 #astronomical unit, km
r_sun = 695700   #solar radius, km


def load_ephemeris(ephemeris_file):
    """
    Parameters
    ----------
    ephemeris_file : str
        The ephemeris file to access.

    Raises
    ------
    LookupError
        in case the file is not loaded correctly.

    Returns
    -------
    time : numpy.ndarray(1,:) of float
        Julian date of an ephemeris table record.
    hae_r : numpy.ndarray(3,:) of float
        3D vector of HAE coordinates for every record in the epehemeris file.
    hae_v : numpy.ndarray(3,:) of float
        3D vector of HAE velocities for every record in the epehemeris file.
    hae_phi : numpy.ndarray(1,:) of float
        Phase angle for every ephemeris table record.
    radial_v : numpy.ndarray(1,:) of float
        Radial velocity in km/s for every ephemeris table record.
    tangential_v : numpy.ndarray(1,:) of float
        Tangential velocity in km/s for every ephemeris table record.
    hae_theta : numpy.ndarray(1,:) of float
        Inclination angle of the position for every ephemeris table record. 
        Measred from the North pole, hence ecpliptics being 90 deg.
    v_phi : numpy.ndarray(1,:) of float
        Phase angle of the velocity for every ephemeris table record.
    v_theta : numpy.ndarray(1,:) of float
        Inclination angle of the velocity for every ephemeris table record. 
        Measured from the ecliptics, with negative meaning North.
    """

    time = np.zeros(0)
    hae_r = np.zeros(0)
    hae_v = np.zeros(0)

    try:
        with open(ephemeris_file) as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                time = np.append(time,float(row[0]))
                hae_r = np.append(hae_r,[float(row[2]),float(row[3]),float(row[4])])
                hae_v = np.append(hae_v,[float(row[5]),float(row[6]),float(row[7])])
            hae_r = np.reshape(hae_r,((len(hae_r)//3,3)))
            hae_v = np.reshape(hae_v,((len(hae_v)//3,3)))
            r = (hae_r[:,0]**2 + hae_r[:,1]**2 + hae_r[:,2]**2)**0.5
            v = (hae_v[:,0]**2 + hae_v[:,1]**2 + hae_v[:,2]**2)**0.5
            hae_phi = np.degrees(np.arctan2(hae_r[:,1],hae_r[:,0]))
            hae_theta = np.degrees(np.arccos((hae_r[:,2]/r[:])))
            v_phi = np.degrees(np.arctan2(hae_v[:,1],hae_v[:,0]))
            v_theta = np.degrees(np.arccos((hae_v[:,2]/v[:])))-90

    except:
        raise LookupError("Unable to load file "+ephemeris_file)

    else:
        #compute radial and tangential velocities
        radial_v = np.zeros(len(hae_r[:,0]))
        tangential_v = np.zeros(len(hae_r[:,0]))
        for i in range(len(hae_r[:,0])):
            unit_radial = hae_r[i,:]/np.linalg.norm(hae_r[i,:])
            radial_v[i] = np.inner(unit_radial,hae_v[i,:])
            tangential_v[i] = np.linalg.norm(hae_v[i,:]-radial_v[i]*unit_radial)     
        return (time,
                hae_r,
                hae_v,
                hae_phi,
                radial_v,
                tangential_v,
                hae_theta,
                v_phi,
                v_theta)


def load_hae(ephemeris_file):
    """
    A wrapper to return a table of tiems and HAEs for the given ephemeris file.
    Uses load_ephemeris().

    Parameters
    ----------
    ephemeris_file : str
        The ephemeris file to access.

    Returns
    -------
    jd : np.array of float
        Times in jd.
    hae/au : np.array of float
        HAE positions in AU.

    """
    (jd,
     hae,
     hae_v,
     hae_phi,
     radial_v,
     tangential_v,
     hae_theta,
     v_phi,
     v_theta) = load_ephemeris(ephemeris_file)

    return jd, hae/au


def fetch_heliocentric(file,
                       cache_psp = True,
                       location = os.path.join("998_generated",
                                               "assets",
                                               "")):
    """
    This function returns helicoentric distance & phase, in addition to 
    radial and tangential velocities in a form of 1D functions. Be careful, 
    there is extrapolation.

    Parameters
    ----------
    file : str
       The ephemeris file to access.
    cache_psp : bool, optional
        Whether to use the chached files for PSP. 
        This is a legacy and compatiblity thing, it will end up wrong if used
        with a body different from PSP, use with caution.
        The default is True for legacy reasons. 

    Returns
    -------
    f_hel_r : 1D function: float -> float
        heliocentric distance in AU, function of JD.
    f_hel_phi : 1D function: float -> float
        heliocentric phase angle, measured from the first point of Aries.
    f_rad_v : 1D function: float -> float
        heliocentric radial velocity in km/s.
    f_tan_v : 1D function: float -> float
        heliocentric tangential veloctiy in km/s.
    """
    if "psp" not in file:
        cache_psp = False
    if cache_psp:
        try:
            with open(location+"f_hel_r.pkl", "rb") as f:
                f_hel_r = pickle.load(f)
            with open(location+"f_hel_phi.pkl", "rb") as f:
                f_hel_phi = pickle.load(f)
            with open(location+"f_rad_v.pkl", "rb") as f:
                f_rad_v = pickle.load(f)
            with open(location+"f_tan_v.pkl", "rb") as f:
                f_tan_v = pickle.load(f)
            with open(location+"f_v_phi.pkl", "rb") as f:
                f_v_phi = pickle.load(f)
            with open(location+"f_v_theta.pkl", "rb") as f:
                f_v_theta = pickle.load(f)
        except:
            print("assets missing, loading "+file)
            (jd_ephem,
             hae_r,
             hae_v,
             hae_phi,
             radial_v,
             tangential_v,
             hae_theta,
             v_phi,
             v_theta) = load_ephemeris(file)
            heliocentric_distance = np.sqrt(  hae_r[:,0]**2
                                            + hae_r[:,1]**2
                                            + hae_r[:,2]**2 )/au #in au
            f_hel_r = interpolate.interp1d(jd_ephem,heliocentric_distance,
                                           fill_value="extrapolate",kind=3)
            f_hel_phi = interpolate.interp1d(jd_ephem,hae_phi,
                                             fill_value="extrapolate",kind=3)
            f_rad_v = interpolate.interp1d(jd_ephem,radial_v,
                                           fill_value="extrapolate",kind=3)
            f_tan_v = interpolate.interp1d(jd_ephem,tangential_v,
                                           fill_value="extrapolate",kind=3)
            f_v_phi = interpolate.interp1d(jd_ephem,v_phi,
                                           fill_value="extrapolate",kind=3)
            f_v_theta = interpolate.interp1d(jd_ephem,v_theta,
                                             fill_value="extrapolate",kind=3)
    
            print("constructing assets for PSP")
            os.makedirs(location, exist_ok=True)
            with open(location+"f_hel_r.pkl", "wb") as f:
                pickle.dump(f_hel_r, f)
            with open(location+"f_hel_phi.pkl", "wb") as f:
                pickle.dump(f_hel_phi, f)
            with open(location+"f_rad_v.pkl", "wb") as f:
                pickle.dump(f_rad_v, f)
            with open(location+"f_tan_v.pkl", "wb") as f:
                pickle.dump(f_tan_v, f)
            with open(location+"f_v_phi.pkl", "wb") as f:
                pickle.dump(f_v_phi, f)
            with open(location+"f_v_theta.pkl", "wb") as f:
                pickle.dump(f_v_theta, f)
        else:
            pass
        finally:
            return f_hel_r, f_hel_phi, f_rad_v, f_tan_v, f_v_phi, f_v_theta
    else:
        (jd_ephem,
         hae_r,
         hae_v,
         hae_phi,
         radial_v,
         tangential_v,
         hae_theta,
         v_phi,
         v_theta) = load_ephemeris(file)
        heliocentric_distance = np.sqrt(  hae_r[:,0]**2
                                        + hae_r[:,1]**2
                                        + hae_r[:,2]**2 )/au #in au
        f_hel_r = interpolate.interp1d(jd_ephem,heliocentric_distance,
                                       fill_value="extrapolate",kind=3)
        f_hel_phi = interpolate.interp1d(jd_ephem,hae_phi,
                                         fill_value="extrapolate",kind=3)
        f_rad_v = interpolate.interp1d(jd_ephem,radial_v,
                                       fill_value="extrapolate",kind=3)
        f_tan_v = interpolate.interp1d(jd_ephem,tangential_v,
                                       fill_value="extrapolate",kind=3)
        f_v_phi = interpolate.interp1d(jd_ephem,v_phi,
                                       fill_value="extrapolate",kind=3)
        f_v_theta = interpolate.interp1d(jd_ephem,v_theta,
                                         fill_value="extrapolate",kind=3)
        return f_hel_r, f_hel_phi, f_rad_v, f_tan_v, f_v_phi, f_v_theta


def get_state(epoch, ephem_file):
    """
    The function to get the heliocentric distance and the velocity of SolO.

    Parameters
    ----------
    epoch : int
        The time of interest, tt2000.
    ephem_file : str
        The location of the ephemereis file.

    Returns
    -------
    r : float
        heliocentric distance in AU.
    v : float
        total speed in km/s.
    v_rad : float
        radial speed in km/s.
    v_phi : float
        The velocity angle in XY plane in degrees, measured from the first
        point of Airies (HAE).
    v_theta : float
        The velocity angle inclination from the XY plane 
        in degrees as in (HAE).

    """
    (f_hel_r,
     f_hel_phi,
     f_rad_v,
     f_tan_v,
     f_v_phi,
     f_v_theta) = fetch_heliocentric(ephem_file)

    jd = tt2000_to_jd(epoch)
    r = float(f_hel_r(jd))
    v_rad = float(f_rad_v(jd))
    v_tan = float(f_tan_v(jd))
    v = float(np.sqrt(v_rad**2+v_tan**2))
    v_phi = float(f_v_phi(jd))
    v_theta = float(f_v_theta(jd))

    return r, v, v_rad, v_phi, v_theta


def get_approaches(ephem_file,
                   distance=0.2):
    """
    Gets an array of all the approaches at which the spacecraft 
    got closer than the specified distance.

    Parameters
    ----------
    distance : float, optional
        The approach defining distance in AU. The default is 0.2.

    Returns
    -------
    approaches : np.array of floats
        Julian dates of the approaches.

    """

    (jd,
     hae,
     hae_v,
     hae_phi,
     radial_v,
     tangential_v,
     hae_theta,
     v_phi,
     v_theta) = load_ephemeris(ephem_file)

    (f_hel_r,
     f_hel_phi,
     f_rad_v,
     f_tan_v,
     f_v_phi,
     f_v_theta) = fetch_heliocentric(ephem_file)

    jd_resampled = np.arange(min(jd),max(jd),1/2880)
    r_resampled = f_hel_r(jd_resampled)
    local_minima = argrelextrema(r_resampled, np.less)[0]
    approaches = jd_resampled[local_minima[r_resampled[local_minima]<distance]]
    return approaches


def get_phase_angle(ephem_file):
    """
    A function to get a function for an object's phase angle as
    a function of julian date. 

    Parameters
    ----------
    ephem_file : str
        The location of the ephemeris file for the body of interest.

    Returns
    -------
    phase_angle : function : float -> float
        The phase angle as a function of julian date.

    """
    jds, hae = load_hae(ephem_file)
    x = interpolate.interp1d(jds,hae[:,0],fill_value="extrapolate",kind=3)
    y = interpolate.interp1d(jds,hae[:,1],fill_value="extrapolate",kind=3)
    def phase_angle(jd):
        """
        The HAE phase angle as a function of julian date. 
        Heliocentric phase angle is measured from the first point of Aries.

        Parameters
        ----------
        jd : float or np.array of float
            Julian date of interest.

        Returns
        -------
        angle : float of np.array of float
            HAE phase angle, measured from the first point of Aries,
            in degrees. The shape matches the input.

        """
        angle = np.degrees(np.arctan2(y(jd),x(jd)))
        return angle
    return phase_angle













