import os

l3_dust_location = os.path.join("..","data","psp","fields","l3_dust","")
exposure_location = os.path.join("..","data","psp","fields","exposure","")
dfb_location = os.path.join("..","data","psp","fields","dfb_wf_vdc","")
psp_ephemeris_file = os.path.join("data_synced","psp_ephemeris_noheader.txt")
solo_ephemeris_file = os.path.join("data_synced","solo_ephemeris_noheader.txt")
psp_sun_ephemeris_file = os.path.join("data_synced","psp_sun_noheader.txt")
figures_location = os.path.join("998_generated","figures","")
all_obs_location = os.path.join("998_generated","observations","")
readable_data = os.path.join("data_synced","psp_flux_readable.csv")

psp_model_location = os.path.join("data_synced","parkersolarprobe.stl")
solo_model_nopanels_location = os.path.join("data_synced","solarorbiter_nopanels.stl")
solo_model_location = os.path.join("data_synced","solarorbiter.stl")

if __name__ == "__main__":
    print("data paths in your system:")
    print(l3_dust_location)
    print("---------------------------------------")
    print("checking ephemeris files:")
    for file in [psp_ephemeris_file]:
        try:
            with open(file, 'r') as f:
                print(file+" OK in place")
        except:
            print(file+" not OK")