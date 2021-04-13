import json
import h5py
from tqdm import tqdm_notebook as tqdm
import time
import os
import numpy as np



def get_filename_by_chan(filenames, chan):
    """
    Finds the filename by the channel name
    
    Parameters
    ----------
    filenames : list of filenames
    chan : channel to find
    
    Returns
    -------
    file : the filename of the file requested
    """
    file = [f for f in filenames if '_chan-'+chan in f][0]
    return(file)


def new_derivative_name(filename, derivative):
    """
    Finds the new file location name for the derivative

    Parameters
    ----------
    filename: the location of the file to be changed.
    derivative: the name of the new derivative. E.g. mwt-4

    Returns
    -------
    filename for datafile, filename for json file
    """
    if ('U19Data' in filename):
        filename = filename.replace('U19Data','derivatives')
    folders = filename.rsplit('/', 1)
    new = folders[0] + '/' + derivative + '/' + folders[1].split('.')[0] + '_' + derivative + '.' + folders[1].split('.')[1]
    new_json = new.rsplit('.', 1)[0] + '.json'
    return new, new_json

def get_package_data(path):
    """
    Finds the path to the packaged data
    
    Parameters
    ----------
    path : filename to check
    
    Returns
    -------
    path to filename
    """
    _ROOT = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(_ROOT, 'data', path)

def save_json_file(data, filename, overwrite=False, local=False):
    """
    Creates json file from dictionary.
    json files are stored in lfp_tools/data

    Parameters
    ----------
    data : the data to be stored
    filename : the filename in local storage
    overwrite : Flag to tell whether to overwrite existing file (if exists)
    local : Flag to tell whether to save json file locally or in packaged data

    Returns
    -------
    filename of local storage
    """
    if (local):
        filename = filename
    else:
        filename = get_package_data(filename)
        
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
    json files are stored in lfp_tools/data

    Parameters
    ----------
    filename: the filename in local storage

    Returns
    -------
    data: the contents of the json file
    """
    with open(get_package_data(filename), 'r') as ff:
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

def save_h5py_file(data, col_name, filename='new_file'):
    """
    Creates h5py file with data to store

    Parameters
    ----------
    data: the data to be saved.
    col_name: the kind of data to be saved
    filename: the filename in local storage

    Returns
    -------
    filename of local storage
    """
    file = h5py.File(filename, 'w')
    file.create_dataset(col_name, data=data)
    file.close()
    return filename


def batch_process(func, params, client, mini_batch_size=None, return_futures=True):
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
        
    mini_batch_size : int
        size of batches to compute at once
        
    client : a dask Client to an initialized cluster, optional. 
        Defaults to start a new client.
    """
    if (mini_batch_size==None):
        mini_batch_size = len(params)
    
    pbar = tqdm(total=int(np.ceil(len(params)/mini_batch_size)))
    exceptions = {}
    outputs = {}
    
    for i in range(int(np.ceil(len(params)/mini_batch_size))):
        mini_done = False
        mini_pbar = tqdm(total=mini_batch_size)
        n_done = 0
        
        mini_results = client.map(func, params[i*mini_batch_size:(i+1)*mini_batch_size])
        
        while not mini_done:
            time.sleep(1)
            n_done_now = sum([r.done() for r in mini_results])
            if n_done_now > n_done:
                mini_pbar.update(n_done_now - n_done)
                n_done = n_done_now
            mini_done = n_done == mini_batch_size
        
        
        for ii, rr in enumerate(mini_results): 
            if rr.status == 'error':
                exceptions[ii+i*mini_batch_size] = rr.exception()
            else:
                if return_futures:
                    outputs[ii+i*mini_batch_size] = rr
                else:
                    outputs[ii+i*mini_batch_size] = rr.result()
        pbar.update(1)
            
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
    json files are stored in lfp_tools/data
    Assumes merging across lists, arrays, or dictionaries
    
    Parameters
    ----------------
    master_file: filename of json file that you want to keep
    branch_file: filename of json file that you want to merge into master_file
    remove_duplicates: Flag to tell whether to keep unique values
    
    Returns
    ----------------
    Message with filenames
    """
    if (not os.path.exists(get_package_data(master_file))):
        print(master_file + ' doesn\'t exist, check name')
    if (not os.path.exists(get_package_data(branch_file))):
        print(branch_file + ' doesn\'t exist, check name')
    if (not os.path.exists(get_package_data(master_file)) or not os.path.exists(get_package_data(branch_file))):
        return('Files not merged')
    master_dict = load_json_file(get_package_data(master_file))
    branch_dict = load_json_file(get_package_data(branch_file))
    master_keys = list(master_dict.keys())
    branch_keys = list(branch_dict.keys())
    for b in branch_keys:
        if (b in master_keys):
            if (isinstance(master_dict[b], dict)):
                master_dict[b].update(branch_dict[b])
            else:
                master_dict[b] = list(np.unique(np.hstack(master_dict[b] + branch_dict[b])))
        else:
            master_dict[b] = branch_dict[b]
    save_json_file(master_dict, master_file, True)
    os.remove(get_package_data(branch_file))
    return('Files merged: ' + get_package_data(master_file) + ' + ' + get_package_data(branch_file))

def remove_json_dicts(master_file, branch_file):
    """
    Removes values and subsequent empty keys from master_file, and discards branch_file
    json files are stored in lfp_tools/data
    
    Parameters
    ---------------
    master_file: filename of json dictionary to keep
    branch_file: filename of json dictionary of values to discard
    
    Return
    ---------------
    Message of success with filenames
    """
    if (not os.path.exists(get_package_data(master_file))):
        print(master_file + ' doesn\'t exist, check name')
    if (not os.path.exists(get_package_data(branch_file))):
        print(branch_file + ' doesn\'t exist, check name')
    if (not os.path.exists(get_package_data(master_file)) or not os.path.exists(get_package_data(branch_file))):
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
    os.remove(get_package_data(branch_file))
    return('Dictionary values removed: ' + get_package_data(master_file) + ' + ' + get_package_data(branch_file))