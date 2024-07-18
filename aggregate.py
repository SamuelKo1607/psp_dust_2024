import csv
import os

from load_data import load_all_obs
from load_data import Observation

from paths import all_obs_location


def make_flux_to_fit_inla(observations,
                          name="psp_flux_readable.csv",
                          location=os.path.join("data_synced","")):
    """
    Creates a CSV in a reasonably readable format, to use in an external
    application, such as RStudio. 

    Parameters
    ----------
    observations : list of Observation
        Impact data to include.
    name : str, optional
        the name of the file. The default is "flux_readable.csv".
    location : str, optional
        The path where to put the result. 
        The default is os.path.join("data_synced","").

    Returns
    -------
    None.

    """

    with open(location+name, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["Julian date",
                         "Count corrected [/day]",
                         "Radial velocity [km/s]",
                         "Tangential velocity [km/s]",
                         "Radial distance [au]",
                         "Detection time [hours]",
                         "Velocity phase angle [deg]",
                         "Velocity inclination [deg]",
                         "V_X (HAE) [km/s]",
                         "V_Y (HAE) [km/s]",
                         "V_Z (HAE) [km/s]",
                         "Deviation angle [deg]",
                         "Area front [m^2]",
                         "Area side [m^2]"])
        for obs in observations:
            writer.writerow([str(obs.jd_center),
                             str(obs.count_corrected),
                             str(obs.heliocentric_radial_speed),
                             str(obs.heliocentric_tangential_speed),
                             str(obs.heliocentric_distance),
                             str(obs.duty_hours),
                             str(obs.velocity_phase),
                             str(obs.velocity_inclination),
                             str(obs.velocity_HAE_x),
                             str(obs.velocity_HAE_y),
                             str(obs.velocity_HAE_z),
                             str(obs.los_deviation),
                             6.11,
                             4.62])



def aggregate_flux_readable(location=os.path.join("998_generated",
                                                    "observations",""),
                            target_location=os.path.join("data_synced",""),
                            force=False):
    """
    A wrapper function to aggregate all Observation objects creted with 
    load_data.py into one human redable file, this checks if it exists 
    and if not, then saves it.

    Parameters
    ----------
    location : str, optional
        The location of the source data. 
        The default is os.path.join("998_generated","obseervations","").
    target_location : str, optional
        The location where the aggregated data will be saved. 
        The default is os.path.join("data_synced","").
    force : bool, optional
        Whether to overwrite any existing file. The default is False.

    Returns
    -------
    None.

    """
    if force:
        print("saving readable file forcedly")
        make_flux_to_fit_inla(load_all_obs(all_obs_location),
                              location = target_location)
    else:
        try:
            with open(target_location+"psp_flux_readable.csv") as f:
                print("readable file OK:")
                print(f)
        except:
            print("saving readable file")
        make_flux_to_fit_inla(load_all_obs(all_obs_location),
                              location = target_location)


#%%
if __name__ == "__main__":
    aggregate_flux_readable()

