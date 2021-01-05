import json
import h5py
from tqdm import tqdm_notebook as tqdm
import time
import os
import numpy as np


def get_package_data(path):
    _ROOT = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(_ROOT, 'data', path)

def save_json_file(data, filename, overwrite=False):
    """
    Creates json file from dictionary

    Parameters
    ----------
    data: the data to be stored
    filename: the filename in local storage

    Returns
    -------
    filename of local storage
    """
    if (os.path.exists(filename) and overwrite):
        print('Overwriting...')
    elif (os.path.exists(filename) and not overwrite):
        print('File already exists, set overwrite to True')
        return filename
    with open(filename, 'w') as ff:
        json.dump(data, ff)
    return filename

def load_json_file(filename):
    """
    Loads a json_file

    Parameters
    ----------
    filename: the filename in local storage

    Returns
    -------
    data: the contents of the json file
    """
    with open(filename, 'r') as ff:
        data = json.load(ff)
    return data

def open_h5py_file(file, fs):
    """
    Gets the h5py file from online storage. The file must include only one datafile.

    Parameters
    ----------
    file: the location of the file to be opened.
    fs: the file system object

    Returns
    -------
    Data file
    """
    with fs.open(file) as f_chan:
        f_chan = h5py.File(f_chan, 'r')
        keys = list(f_chan.keys())
        datakeys = [i for i in keys if '__' not in i]
        mwt_chan = f_chan[datakeys[0]]
        mwt_chan = mwt_chan[:].squeeze()
    return mwt_chan

def batch_process(func, params, client, upload_mods=[]):
    """ 
    Map `params` onto `func` and submit to a dask kube cluster.
    
    Parameters
    ----------
    func : callable
        For now, accepts a single tuple as input and unpacks that tuple internally. 
        See below in `params`.
    
    params : sequence 
        Has the form [(a1, b1, c1, ...), (a2, b2, c2, ...), ...., (an, bn, cn, ...)], 
        where each tuple is the inputs to one process.
        
    client : a dask Client to an initialized cluster, optional. 
        Defaults to start a new client.
    """
    for m in upload_mods:
        client.upload_file(m)
    results = client.map(func, params)
    
    all_done = False 
    pbar = tqdm(total=len(params))
    n_done = 0
    while not all_done:
        time.sleep(1)
        n_done_now = sum([r.done() for r in results])
        if n_done_now > n_done:
            pbar.update(n_done_now - n_done)
            n_done = n_done_now

        all_done = n_done == len(params)
    
    exceptions = {}
    outputs = {}
    for ii, rr in enumerate(results): 
        if rr.status == 'error':
            exceptions[ii] = rr.exception()
        else:
            outputs[ii] = rr
            
    return outputs, exceptions

def get_results(log):
    """
    Retrieve results from tools.batch_process.
    If errors are present, will return those and good results
    
    Parameters
    ----------
    log : list of dictionaries of results.
        
    Returns
    -------
    results, list of errors
    """
    results = []
    for i in list(log[0].values()):
        results.append(i.result())
    if(not bool(log[1])):
        return(results, log[1])
    else:
        print('Errors Present')
        return(results, log[1])
    
def merge_json_dicts(master_file, branch_file, remove_duplicates=True):
    """
    Merges two json dictionaries, and discard branch_file
    
    Parameters
    ----------------
    master_file: filename of json file that you want to keep
    branch_file: filename of json file that you want to merge into master_file
    remove_duplicates: Flag to tell whether to keep unique values
    
    Returns
    ----------------
    Message with filenames
    """
    if (not os.path.exists(master_file)):
        print(master_file + ' doesn\'t exist, check name')
    if (not os.path.exists(branch_file)):
        print(branch_file + ' doesn\'t exist, check name')
    if (not os.path.exists(master_file) or not os.path.exists(branch_file)):
        return('Files not merged')
    master_dict = load_json_file(master_file)
    branch_dict = load_json_file(branch_file)
    master_keys = list(master_dict.keys())
    branch_keys = list(branch_dict.keys())
    for b in branch_keys:
        if (b in master_keys):
            master_dict[b] = list(np.unique(np.hstack(master_dict[b] + branch_dict[b])))
        else:
            master_dict[b] = branch_dict[b]
    save_json_file(master_dict, master_file, True)
    os.remove(branch_file)
    return('Files merged: ' + master_file + ' + ' + branch_file)

def remove_json_dicts(master_file, branch_file):
    """
    Removes values and subsequent empty keys from master_file, and discards branch_file
    
    Parameters
    ---------------
    master_file: filename of json dictionary to keep
    branch_file: filename of json dictionary of values to discard
    
    Return
    ---------------
    Message of success with filenames
    """
    if (not os.path.exists(master_file)):
        print(master_file + ' doesn\'t exist, check name')
    if (not os.path.exists(branch_file)):
        print(branch_file + ' doesn\'t exist, check name')
    if (not os.path.exists(master_file) or not os.path.exists(branch_file)):
        return('Values not removed')
    master_dict = load_json_file(master_file)
    branch_dict = load_json_file(branch_file)
    master_keys = list(master_dict.keys())
    branch_keys = list(branch_dict.keys())
    for b in branch_keys:
        if (b in master_keys):
            master_dict[b] = [m for m in master_dict[b] if m not in branch_dict[b]]
            if (len(master_dict[b])==0):
                master_dict.pop(b, None)
    save_json_file(master_dict, master_file, True)
    os.remove(branch_file)
    return('Dictionary values removed: ' + master_file + ' + ' + branch_file)