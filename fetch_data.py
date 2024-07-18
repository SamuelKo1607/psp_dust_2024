import requests
import os
import datetime as dt
import cdflib
import numpy as np

from conversions import date2YYYYMMDD

from paths import l3_dust_location
from paths import dfb_location

def build_url(YYYYMMDD,
              version="v01",
              product="dust",
              hour=None):
    """
    A function to prepare the URL where the data file is found. 
    The URL goes like:  https://research.ssl.berkeley.edu/data/psp/data/
                        sci/fields/l3/dust/2019/08/
                        psp_fld_l3_dust_20190806_v01.cdf

    Parameters
    ----------
    YYYYMMDD : str
        The date of interest.
    version : str, optional
        The suffix indicating the version. The default is "v01", which is 
        sufficient as of 11/2023 for dust, but we need "v02" for dfb.
    product : str, optional
        The nickname of the product requested, for now one of "dust" or 
        "dfb_wf_vdc". Dhe default is "dust".
    hour : str, optional
        The hour of interest, needed for dfb products. Usually one of 
        "00", "06", "12", "18". The default is None.

    Returns
    -------
    url : str
        The URL to the datafile for the day.

    filename : str
        Just the filename.

    """
    YYYY = YYYYMMDD[:4]
    MM = YYYYMMDD[4:6]
    if product=="dust":
        url = ("https://research.ssl.berkeley.edu/data/psp/data/sci/fields/l3/"
            + f"dust/{YYYY}/{MM}/psp_fld_l3_dust_{YYYYMMDD}_{version}.cdf")
    elif product=="dfb_wf_vdc":
        if hour is None:
            raise Exception(f"hour not provided")
        url = ("https://research.ssl.berkeley.edu/data/psp/data/sci/fields/l2/"
          + f"dfb_wf_vdc/{YYYY}/{MM}/"
          + f"psp_fld_l2_dfb_wf_vdc_{YYYYMMDD}{hour}_{version}.cdf")
    else:
        raise Exception(f"an unknown data product: {product}")
    filename = url[url.find("psp_fld_l"):]
    return url, filename


def fields_download(YYYYMMDD,
                    target_folder=l3_dust_location,
                    product="dust",
                    version="v01",
                    hour=None):
    """
    The function that downloads the .cdf L3 dust file for the requested date.

    Parameters
    ----------
    YYYYMMDD : str
        The date of interest.
    target_folder : str, optional
        The target folder for the download. 
        The default is l3_dust_location.
    product : str, optional
        The product nickname, passed down to build_url. The default is "dust"
        for legacy reasons.
    version : str, optional
        The version of the product, passed down to build_url. The default 
        is "v01" for legacy reasons.
    hour : str, optional
        The hour of interest, as in build_url, needed for dfb products. 
        The default is None.

    Returns
    -------
    target : str
        The filepath to the downloaded file.

    Raises
    ------
    Exception
        In case the download failed. 

    """
    url, filename = build_url(YYYYMMDD,version,product,hour)
    target = os.path.join(target_folder,filename)

    a = dt.datetime.now()
    r = requests.get(url, allow_redirects=True)
    if not str(r)=="<Response [404]>":
        open(target, 'wb').write(r.content)
        print(str(round(os.path.getsize(target)/(1024**2),ndigits=2))
              +" MiB dowloaded in "+str(dt.datetime.now()-a))
    else:
        print(filename+" N/A; <Response [404]>")
        raise Exception(f"Download unseccsessful @ {YYYYMMDD}")
    return target


def fields_fetch(YYYYMMDD,
                 target_folder=l3_dust_location,
                 product="dust",
                 version="v01",
                 hour=None):
    """
    The function to hand in a file for the requests date. If not present,
    then calls fields_download. Either returns the target file that 
    can be reached or rasies an Exception through fields_download.

    Parameters
    ----------
    YYYYMMDD : str
        The date of interest.
    target_folder : str, optional
        The target folder for the download. 
        The default is l3_dust_location.
    product : str, optional
        The product nickname, passed down to build_url. The default is "dust"
        for legacy reasons.
    version : str, optional
        The version of the product, passed down to build_url. The default 
        is "v01" for legacy reasons.
    hour : str, optional
        The hour of interest, as in build_url, needed for dfb products. 
        The default is None.

    Returns
    -------
    target : str
        The filepath to the downloaded file that can be reached.

    """

    url, filename = build_url(YYYYMMDD,version,product,hour)
    target = os.path.join(target_folder,filename)
    try:
        f = open(target)
    except:
        target = fields_download(YYYYMMDD,target_folder,product,version,hour)
    else:
        f.close()
    return target


def fields_load(YYYYMMDD,
                target_folder=l3_dust_location,
                product="dust",
                version="v01",
                hour=None):
    """
    A wrapped to load the correct psp dust file as a cdf file using cdflib.

    Parameters
    ----------
    YYYYMMDD : str
        The date of interest.
    target_folder : str, optional
        The target folder for the download. 
        The default is l3_dust_location.
    product : str, optional
        The product nickname, passed down to build_url. The default is "dust"
        for legacy reasons.
    version : str, optional
        The version of the product, passed down to build_url. The default 
        is "v01" for legacy reasons.
    hour : str, optional
        The hour of interest, as in build_url, needed for dfb products. 
        The default is None.

    Returns
    -------
    cdf_file : cdflib.cdfread.CDF
        The cdf datafile of interest.

    """
    target = fields_fetch(YYYYMMDD,target_folder,product,version,hour)
    cdf_file = cdflib.CDF(target)
    return cdf_file


def get_list_of_days(date_min = dt.date(2018,10,2),
                     date_max = dt.date(2023,12,31)):

    days = np.arange(date_min,
                     date_max,
                     step=dt.timedelta(days=1)
                     ).astype(dt.datetime)
    YYYYMMDDs = [date2YYYYMMDD(day) for day in days]
    return YYYYMMDDs


#%%
product = "dfb_wf_vdc" #"dust

if __name__ == "__main__":
    for YYYYMMDD in get_list_of_days():
        try:
            if product=="dust":
                fields_fetch(YYYYMMDD)
            elif product=="dfb_wf_vdc":
                for hour in ["00","06","12","18"]:
                    try:
                        fields_fetch(YYYYMMDD,
                                     target_folder=dfb_location,
                                     product="dfb_wf_vdc",
                                     version="v02",
                                     hour=hour)
                    except:
                        pass
        except:
            pass








