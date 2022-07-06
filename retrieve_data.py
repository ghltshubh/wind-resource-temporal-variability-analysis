import h5pyd
import pandas as pd
import numpy as np


HDF5_FILE_URL = "/nrel/wtk/conus/wtk_conus_2012.h5"


# -- read file handle object
def get_data():
    """Connects to the file and create a file object

    Returns:
        f: File object
    """     

    hdf5_file = h5pyd.File(HDF5_FILE_URL, mode='r')
    return hdf5_file


# -- read meta data
def get_metadata(f):
    """Get meta data

    Args:
        f (file object): hdf5 file handle object

    Returns:
        meta: meta dataframe
    """

    meta = pd.DataFrame(f['meta'][...])
    for col, dtype in meta.dtypes.items():
        if dtype == np.object:  # Only process byte object columns.
            meta[col] = meta[col].apply(lambda x: x.decode("utf-8"))
    return meta