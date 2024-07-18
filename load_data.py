import os
import glob
import cdflib
import pickle
import datetime as dt
import numpy as np
from scipy import io
from tqdm.auto import tqdm

from conversions import tt2000_to_date
from conversions import tt2000_to_jd
from ephemeris import get_state
from orientation import fetch_orientation

from paths import l3_dust_location
from paths import all_obs_location
from paths import exposure_location
from paths import dfb_location
from paths import psp_ephemeris_file
from paths import zero_time_csv
from ephemeris import r_sun
from ephemeris import au



class Observation:
    """
    Aggregate results of the measurement period.
    """
    def __init__(self,
                 date,
                 epoch_center,
                 epochs_on_day,
                 encounter,
                 rate_wav, #[s^-1], wave-effects corrected
                 rate_ucc, #[s^-1], undercounting corrected
                 count_raw, #[1], raw count
                 t_obs_eff, #[]
                 inbound,
                 ej2000,
                 heliocentric_distance,
                 spacecraft_speed,
                 heliocentric_radial_speed,
                 velocity_phase,
                 velocity_inclination):
        self.date = date
        self.YYYYMMDD = date.strftime('%Y%m%d')
        self.epoch_center = epoch_center
        self.jd_center = tt2000_to_jd(epoch_center)
        self.epochs_on_day = epochs_on_day
        self.encounter = encounter
        self.encounter_group = encounter_group(encounter)
        self.rate_wav = rate_wav
        self.rate_ucc = rate_ucc
        self.count_raw = count_raw
        self.t_obs_eff = t_obs_eff #[s]
        if count_raw == 0:
            # if no dust was observed in the period of interest
            self.count_corrected = 0
        else:
            self.count_corrected = np.round(count_raw*(rate_ucc/rate_wav))
        self.duty_hours = self.t_obs_eff / 3600
        self.inbound = inbound
        self.ej2000 = ej2000
        self.heliocentric_distance = heliocentric_distance  #[AU]
        self.spacecraft_speed = spacecraft_speed            #[km/s]
        self.heliocentric_radial_speed = heliocentric_radial_speed #[km/s]
        self.heliocentric_tangential_speed = ( spacecraft_speed**2
                                               - heliocentric_radial_speed**2
                                             )**0.5         #[km/s]
        self.velocity_phase = velocity_phase                #[deg]
        self.velocity_inclination = velocity_inclination    #[deg]
        self.velocity_HAE_x = ( spacecraft_speed
                                * np.sin(np.deg2rad(90-velocity_inclination))
                                * np.cos(np.deg2rad(velocity_phase)) )
        self.velocity_HAE_y = ( spacecraft_speed
                                * np.sin(np.deg2rad(90-velocity_inclination))
                                * np.sin(np.deg2rad(velocity_phase)) )
        self.velocity_HAE_z = ( spacecraft_speed
                                * np.cos(np.deg2rad(90-velocity_inclination)) )
        self.los_deviation = fetch_orientation(epoch_center) #degrees
        self.produced = dt.datetime.now()


def encounter_group(enc):
    """
    A lookup table to translate from encounter number to encounter group.

    Parameters
    ----------
    enc : int
        Encounter number strating at 1.

    Returns
    -------
    group : int
        Encounter group .

    Raises
    ------
    Exception : "enc: {enc} is {type(enc)} ; should be int>=0"
        In case the input is not int>0

    """

    if type(enc) not in [int,np.int32] or not enc>=0:
        raise Exception(f"enc: {enc} is {type(enc)} ; should be int>=0")
    elif enc == 0:
        group = 0
    elif enc < 4:
        group = 1
    elif enc < 6:
        group = 2
    elif enc < 8:
        group = 3
    elif enc < 10:
        group = 4
    elif enc < 17:
        group = 5
    elif enc < 22:
        group = 6
    else:
        group = 7

    return group


def save_list(data,
              name,
              location=""):
    """
    A simple function to save a given list to a specific location using pickle.

    Parameters
    ----------
    data : list
        The data to be saved. In our context: mostly a list of objects.
    
    name : str
        The name of the file to be written. May be an absolute path, 
        if the location is not used.
        
    location : str, optional
        The relative path to the data folder. May not be used if the name
        contains the folder as well. Default is therefore empty.

    Returns
    -------
    none
    """

    location = os.path.join(os.path.normpath( location ), '')
    os.makedirs(os.path.dirname(location+name), exist_ok=True)
    with open(location+name, "wb") as f:  
        pickle.dump(data, f)
        
        
def load_list(name,
              location):
    """
    A simple function to load a saved list from a specific location 
    using pickle.

    Parameters
    ----------    
    name : str
        The name of the file to load. 
        
    location : str, optional
        The relative path to the data folder.

    Returns
    -------
    data : list
        The data to be loaded. In our context: mostly a list of objects.
    """

    if location != None:
        location = os.path.join(os.path.normpath( location ), '')
        with open(location+name, "rb") as f:
            data = pickle.load(f)
        return data
    else:
        with open(name, "rb") as f:
            data = pickle.load(f)
        return data


def load_all_obs(location=all_obs_location):
    """
    A tiny wrapper to get all the observations.

    Parameters
    ----------
    location : str, optional
        the location of the all_obs file. 
        The default is all_obs_location from paths.

    Returns
    -------
    all_obs : list of Observation
        All the observations.

    """
    all_obs = load_list("all_obs.pkl",
                        location = location)
    return all_obs


def list_cdf(location=l3_dust_location):
    """
    The function to list all the l3 dust datafiles.

    Parameters
    ----------
    location : str, optional
        The data directory. Default is paths.l3_dust_location.

    Returns
    -------
    files : list of str 
        The available datafiles.
    """

    files = glob.glob(os.path.join(location,"psp_fld_*.cdf"))
    return files


def list_mat(location=exposure_location):
    """
    The function to list all the Matlab dust datafiles.

    Parameters
    ----------
    location : str, optional
        The data directory. Default is paths.exposure_location.

    Returns
    -------
    files : list of str 
        The available datafiles.
    """

    files = glob.glob(os.path.join(location,
                                   "psp_fld_l3_dust_rates_events_*.mat"))
    return files


def build_obs_from_cdf(cdf_file,
                       exposure_success=False,
                       twin=np.zeros(3),
                       twav=np.zeros(3)):
    """
    A function to build a list of observations extracted from one cdf file.

    Parameters
    ----------
    cdf_file : cdflib.cdfread.CDF
        The PSP L3 dust cdf file to extract data from. 
    exposure_success : bool, optional
        Whether the exposure info is provided. The default is False.
    twin : 3-elements np.array of float, 1D
        The observation time in seconds. Zeroes if success == False.
    twav : 3-elements np.array of float, 1D
        The waveform time in seconds. During this time, potential dust 
        impacts are obscured by waveforms. Zeroes if success == False.

    Returns
    -------
    observations : list of Observation object
        One entry for each observation, typically once per 8 hours.

    """

    # Some initial loads.
    epochs = cdf_file.varget("psp_fld_l3_dust_V2_rate_epoch")
    event_epochs = cdf_file.varget("psp_fld_l3_dust_V2_event_epoch")
    YYYYMMDD = [str(cdf_file.cdf_info().CDF)[-16:-8]]*len(epochs)
    dates = tt2000_to_date(epochs)
    encounter = cdf_file.varget("psp_fld_l3_dust_V2_rate_encounter")
    rate_wav = cdf_file.varget("psp_fld_l3_dust_V2_rate_wav")
    rate_ucc = cdf_file.varget("psp_fld_l3_dust_V2_rate_ucc")
    inbound = cdf_file.varget("psp_fld_l3_dust_V2_rate_inoutbound")
    ej2000_x = cdf_file.varget("psp_fld_l3_dust_V2_rate_ej2000_x")
    ej2000_y = cdf_file.varget("psp_fld_l3_dust_V2_rate_ej2000_y")
    ej2000_z = cdf_file.varget("psp_fld_l3_dust_V2_rate_ej2000_z")

    # Check the fromat.
    if len(epochs) != 3:
        raise Exception(f"unexp. struct. @ {YYYYMMDD[0]}: {len(epochs)} != 3")

    # Make the Observations.
    rate_seg_starts = epochs - (epochs[1]-epochs[0])//2
    rate_seg_ends    = epochs + (epochs[1]-epochs[0])//2
    observations = []
    for i,epoch in enumerate(epochs):

        # Get the epehemeris.
        r, v, v_rad, v_phi, v_theta = get_state(epoch, psp_ephemeris_file)

        # Get the counts.
        count_observed = len(event_epochs[
                                     (event_epochs > rate_seg_starts[i])
                                    *(event_epochs < rate_seg_ends[i]) ])

        # Get the observation time.
        if exposure_success:
            t_obs_eff = twin[i] - twav[i]
        elif rate_wav[i]:
            t_obs_eff = count_observed / rate_wav[i]
        else:
            raise Exception(f"no exposure info @ {YYYYMMDD[0]}")

        # Make the Observation, if we have good enough data.
        observations.append(Observation(date = dates[i],
                                        epoch_center = epoch,
                                        epochs_on_day = len(epochs),
                                        encounter = encounter[i],
                                        rate_wav = rate_wav[i],
                                        rate_ucc = rate_ucc[i],
                                        count_raw = count_observed,
                                        t_obs_eff = t_obs_eff,
                                        inbound = inbound[i],
                                        ej2000 = [ej2000_x[i],
                                                  ej2000_y[i],
                                                  ej2000_z[i]],
                                        heliocentric_distance = r,
                                        spacecraft_speed = v,
                                        heliocentric_radial_speed = v_rad,
                                        velocity_phase = v_phi,
                                        velocity_inclination = v_theta
                                        ))

    return observations


def get_exposure_info(YYYYMMDD,
                      exposure_location = exposure_location):
    """
    A function to access and provide the exposure times for a given 
    PSP measurement day.

    Parameters
    ----------
    YYYYMMDD : str
        The date of interest.
    exposure_location : str, optional
        The location of the .mat files. The default is exposure_location.

    Returns
    -------
    success : bool
        Whether we have a hit on the date of interest.
    twin : 3-elements np.array of float, 1D
        The observation time in seconds. Zeroes if success == False.
    twav : 3-elements np.array of float, 1D
        The waveform time in seconds. During this time, potential dust 
        impacts are obscured by waveforms. Zeroes if success == False.
    """
    exposure_file = glob.glob(os.path.join(exposure_location,
                                           "psp_fld_l3_dust_rates_events_*"
                                           +YYYYMMDD
                                           +"*.mat"))
    if len(exposure_file):
        mat = io.loadmat(exposure_file[0])
        twin = mat['psp_fld_l3_dust_V2_rate_Twin'][0]
        twav = mat['psp_fld_l3_dust_V2_rate_Twav'][0]
        success = True
    else:
        twin = np.zeros(3)
        twav = np.zeros(3)
        success = False

    return success, twin, twav


def get_missing_data(dust_location=l3_dust_location,
                     mat_location=exposure_location,
                     output_csv_location=zero_time_csv,
                     save=False):
    """
    A procedure to list all the rate data points based on an empty interval.

    Parameters
    ----------
    dust_location : str, optional
        The path to the data. The default is paths.l3_dust_location.
    mat_location : str, optional
        The location of the .mat files. 
        The default is paths.exposure_location.
    output_csv_location : str, optional
        The target location of the .csv output. 
        The default is paths.zero_time_csv.
    save : bool, optional
        Thether to save the output as a .csv. The default is False.

    Raises
    ------
    Exception
        if too few or too many .cdf matches for a single .mat file.

    Returns
    -------
    None.

    """

    missing_cdfs = []

    if save:
        f = open(os.path.join(output_csv_location,'zero_times.csv'),'w')
        f.write(("YYYYMMDD,"
                 "twin_0, twin_1, twin_2,"
                 "rate_raw_0, rate_raw_1, rate_raw_2\n"))

    for file in list_mat(mat_location):
        YYYYMMDD = file[-12:-4]
        success, twin, twav = get_exposure_info(YYYYMMDD)
        if np.isclose(min(twin),0):
            cdf_match = glob.glob(os.path.join(dust_location,
                                               "psp_fld_l3_dust_*"
                                               +YYYYMMDD
                                               +"*.cdf"))
            if len(cdf_match)==0:
                missing_cdfs.append(YYYYMMDD)
                print(success, twin)
            else:
                cdf_file = cdflib.CDF(cdf_match[0])
                rate_raw = cdf_file.varget("psp_fld_l3_dust_V2_rate_raw")
                line = (f"{YYYYMMDD},{twin[0]},{twin[1]},{twin[2]},"
                        f"{rate_raw[0]},{rate_raw[1]},{rate_raw[2]}")
                if save:
                    f.write(line+"\n")
                else:
                    print(line)

    if save:
        f.close()

    return missing_cdfs


def main(dust_location=l3_dust_location,
         target_directory=all_obs_location,
         save=True):
    """
    A function to aggregate all the observations as per PSP L3 dust. A folder
    is browsed nad a list of Observation type is created and optionally saved.

    Parameters
    ----------
    dust_location : str, optional
        The path to the data. The default is l3_dust_location.
    target_directory : str, optional
        The path where to put the final list. 
        The default is os.path.join("998_generated","observations","").
    save : bool, optional
        Whether to save the data. The default is True.

    Returns
    -------
    observation : list of Observation
        The agregated data.
    """
    observations = []
    for file in tqdm(list_cdf(dust_location)):
        cdf_file = cdflib.CDF(file)
        cdf_short_name = str(cdf_file.cdf_info().CDF)[
                             str(cdf_file.cdf_info().CDF).find("psp_fld_l3_")
                             :-4]

        YYYYMMDD = file[-16:-8]

        success, twin, twav = get_exposure_info(YYYYMMDD)

        try:
            observation = build_obs_from_cdf(cdf_file,success,twin,twav)
        except Exception as err:
            print(f"{cdf_short_name}: {err}")
        else:
            observations.extend(observation)

    if save:
        save_list(observations,
                  "all_obs.pkl",
                  target_directory)

    return observations


#%%
if __name__ == "__main__":
    main()






