import json
import h5py
from tqdm import tqdm_notebook as tqdm
import time
import os
import numpy as np


def func_over_last_dim(data, data_final, num_dim, func, **kwargs):
    '''
    Performs arbitrary function over last dimension(s)
    E.g. could use func = np.mean() and num_dim = 2 to take the mean over the last 2 dimensions
    
    Parameters
    -----------------
    data : the input data
    data_final : the output data
    num_dim : the number of dimensions that are reduced
    func : arbitrary function to apply to the last dimension(s)
    '''
    if(data.shape[:len(data.shape)-num_dim] != data_final.shape):
        print('Input shape, output shape, and num_dim aren\'\'t appropriate')
        return
    
    if(len(data.shape)>1+num_dim):
        for i in range(len(data)):
            func_over_last_dim(data[i], data_final[i], num_dim, func, **kwargs)
    else:
        for i in range(len(data)):
            data_final[i] = func(data[i], **kwargs)

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


def new_derivative_name(filename, derivative, folder):
    """
    Finds the new file location name for the derivative

    Parameters
    ----------
    filename: the location of the file to be changed.
    derivative: the name of the new derivative. E.g. mwt-4
    folder: name of folder to place new derivatives into

    Returns
    -------
    filename for datafile, filename for json file
    """
    if ('rawdata' in filename):
        filename = filename.replace('rawdata', folder)
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

def load_json_file_from_S3(filename, fs):
    """
    Loads a json_file from S3

    Parameters
    ----------
    filename: the filename in S3
    fs : filesytem object

    Returns
    -------
    data: the contents of the json file
    """
    with fs.open(filename, 'r') as ff:
        data = json.load(ff)
    return data

def open_lfp_file(file, fs, filetype='hdf5', num_return=1):
    """
    Opens a mat file from online storage. The file should only include one datafile
    
    Parameters
    ----------
    file: the location of the file to be opened.
    fs: the file system object
    filetype: either hdf5 or mat, depending on the version of saved data
    num_return: number of data arguments to return. Use 'all' to return all of them

    Returns
    -------
    Data file
    """
    if filetype=='hdf5':
        files = open_h5py_file(file, fs, num_return)
    elif filetype=='mat':
        files = open_mat_file(file, fs, num_return)
    else:
        print('Unknown filetype, either hdf5 or mat')
        files = []
    return files

from scipy.io import loadmat
def open_mat_file(file, fs, num_return=1):
    """
    Opens a mat file from online storage. The file should only include one datafile
    
    Parameters
    ----------
    file: the location of the file to be opened.
    fs: the file system object
    num_return: number of data arguments to return. Use 'all' to return all of them

    Returns
    -------
    Data file
    """
    with fs.open(file) as f:
        mat = loadmat(f)
        keys = list(mat.keys())
        datakeys = [i for i in keys if '__' not in i]
        if num_return=='all':
            num_return = len(datakeys)
        if num_return==1:
            data = mat[datakeys[0]][:,0]
        else:
            data = []
            for i in range(num_return):
                data.append(mat[datakeys[i]][:,0])
            data = np.array(data)
    return(data)

def open_h5py_file(file, fs, num_return=1):
    """
    Gets the h5py file from online storage. The file must include only one datafile.

    Parameters
    ----------
    file: the location of the file to be opened.
    fs: the file system object
    num_return: number of data arguments to return. Use 'all' to return all of them

    Returns
    -------
    Data file
    """
    with fs.open(file) as f_chan:
        f_chan = h5py.File(f_chan, 'r')
        keys = list(f_chan.keys())
        datakeys = [i for i in keys if '__' not in i]
        if num_return=='all':
            num_return = len(datakeys)
        if num_return==1:
            temp = f_chan[datakeys[0]]
            data = temp[:].squeeze()
        else:
            data = []
            for i in range(num_return):
                temp = f_chan[datakeys[i]]
                data.append(temp[:].squeeze())
            data = np.array(data)
    return data

def open_local_h5py_file(f_chan, num_return=1):
    """
    Gets the h5py file from local storage. The file must include only one datafile.

    Parameters
    ----------
    f_chan: the location of the file to be opened.
    num_return: number of data arguments to return. Use 'all' to return all of them

    Returns
    -------
    Data file
    """
    f_chan = h5py.File(f_chan, 'r')
    keys = list(f_chan.keys())
    datakeys = [i for i in keys if '__' not in i]
    if num_return=='all':
        num_return = len(datakeys)
    if num_return==1:
        temp = f_chan[datakeys[0]]
        data = temp[:].squeeze()
    else:
        data = []
        for i in range(num_return):
            temp = f_chan[datakeys[i]]
            data.append(temp[:].squeeze())
        data = np.array(data)
    return data

def save_dataframe(df, metadata, location, overwrite=False):
    '''
    Saves a dataframe locally
    
    Parameters
    ----------------
    df : dataframe in question
    metadata : metadata about creation of dataframe
    location : location of data, make sure it doesn't include any ending (.csv e.g.)
    overwrite : if false, doesn't overwrite any data
    
    Returns
    ----------------
    'Files saved: ' + location
    '''
    if (not overwrite):
        assert not (os.path.isfile(location+'.csv') or os.path.isfile(location+'.json')), \
        'File already exists, check location and name'
        
    df.to_csv(location+'.csv')
    save_json_file(metadata, location+'.json', local=True, overwrite=overwrite)
    
    return('Files saved: '+location)

def save_dataframe_to_s3(fs, df, metadata, location, overwrite=False):
    '''
    Saves a dataframe to s3
    
    Parameters
    ----------------
    fs : s3 filesystem object
    df : dataframe in question
    metadata : metadata about creation of dataframe
    location : location of data, make sure it doesn't include any ending (.csv e.g.)
    overwrite : if false, doesn't overwrite any data
    
    Returns
    ----------------
    'Files saved: ' + location
    '''
    if (not overwrite):
        assert not (fs.exists(location+'.csv') or fs.exists(location+'.json')), \
        'File already exists, check location and name'
        
    local_df = 'temp.csv'
    df.to_csv(local_df)
    local_json = save_json_file(metadata, 'temp.json', local=True)
    
    fs.put(local_df, location+'.csv')
    fs.put(local_json, location+'.json')
    
    os.remove(local_df)
    os.remove(local_json)
    
    return('Files saved: '+location)

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

def save_h5py_to_S3(fs, ar, metadata, location, name, col_name='data', local_name='temp', overwrite=False):
    '''
    Saves file to location in S3 in AWS
    
    Parameters
    -----------------
    fs : filesystem object
    ar : numpy array to save as h5py file
    metadata : dictionary to save as json file
    location : bucket in S3 to save to. Make sure the string ends in '/'
    name : name of file for both metadata and ar
    col_name : type of data in ar
        defaults to 'data'
    local_name : name of file for temporary local storage. Necessary to change based on race conditions
        defaults to 'temp'
    overwrite : Condtion to overwrite data if necessary. Recommended to set to False for most cases
        defaults to False
    
    Returns
    -----------------
    message : string dictating where the data was saved
    '''
    if (not overwrite):
        assert not (fs.exists(location+name+'.mat') or fs.exists(location+name+'.json')), \
        'File already exists, check location and name'
    
    local_h5py = save_h5py_file(ar, col_name, filename=local_name+'.mat')
    local_json = save_json_file(metadata, local_name+'.json', local=True)
    
    fs.put(local_h5py, location+name+'.mat')
    fs.put(local_json, location+name+'.json')
    
    os.remove(local_h5py)
    os.remove(local_json)
    
    return('Files saved: '+location+name)


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
        mini_results = client.map(func, params[i*mini_batch_size:(i+1)*mini_batch_size])
        
        try:
            mini_done = False
            mini_pbar = tqdm(total=len(mini_results))
            n_done = 0

            while not mini_done:
                time.sleep(1)
                n_done_now = sum([r.done() for r in mini_results])
                if n_done_now > n_done:
                    mini_pbar.update(n_done_now - n_done)
                    n_done = n_done_now
                mini_done = n_done == len(mini_results)


            for ii, rr in enumerate(mini_results): 
                if rr.status == 'error':
                    exceptions[ii+i*mini_batch_size] = rr.exception()
                else:
                    if return_futures:
                        outputs[ii+i*mini_batch_size] = rr
                    else:
                        outputs[ii+i*mini_batch_size] = rr.result()
            pbar.update(1)
        except KeyboardInterrupt:
            print('Keyboard Interrupt, returning current set of futures and all collected errors')
            return mini_results, exceptions
            
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
    
def cancel_futures(log, client):
    '''
    Cancels futures created with batch_process
    Note: works with dictionaries and arrays
    
    Parameters
    ------------------
    log : list of list (or dicts) of futures
    client : client that created the futures
    '''
    for log_temp in log:
        if type(log_temp)==dict:
            for l in list(log_temp.values()):
                client.cancel(l)
        else:
            for l in log_temp:
                client.cancel(l)
    
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