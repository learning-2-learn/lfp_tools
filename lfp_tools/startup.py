from lfp_tools import general
from lfp_tools import analysis
from collections import Counter
import pandas as pd
import numpy as np
import s3fs

from dask_gateway import Gateway

def get_bipole_hilb_files(fs, species, subject, exp, session, band, dtype='abs'):
    '''
    Gets the preprocessed bipole-subtracted, Hilbert transformed data
    
    Parameters
    ----------
    fs : file system object
    species : species
    subject : the subject
    exp : the experiment
    session : the session
    band : the bandpass band
    dtype : string of 'abs' or 'ang', indicating the absolute value or phase of the Hilbert transform
    
    Returns
    -------
    files : list of files
    '''
    files = fs.ls(
        'nhp-lfp/wcst-preprocessed/derivatives_bipole/sub-'+subject+\
        '/sess-'+session+\
        '/lfp/bipole/norm-z/notch-mne/bp-'+band+\
        '/hilb-'+dtype
    )
    files = [f for f in files if '.mat' in f]
    return(files)

def get_fs():
    """
    Gets the filesystem object.
    
    Returns
    -------
    fs : filesystem object
    """
    fs = s3fs.S3FileSystem()
    return(fs)

def get_sac_dataframe(fs, species, subject, exp, session, get_json=False):
    '''
    Retrieves saccade dataframe from S3
    
    Parameters
    ------------------
    fs : s3 filesystem object
    species : species (currently only valid for nhp)
    subject : subject
    exp : experiment
    session : session identifier
    get_json : flag to indicate whether to return the json file too
    
    Returns
    ------------------
    if not get_json:
        sac : saccade dataframe
    else:
        sac_json : json file attached to dataframe
    '''
    assert species=='nhp', 'Function not currently valid for humans or any species except nhp'
    loc = 'nhp-lfp/wcst-preprocessed/rawdata/sub-'+subject+\
          '/sess-'+session+'/behavior/sub-'+subject+'_sess-'+session+'_saccades'
    
    assert (fs.exists(loc+'.csv') and fs.exists(loc+'.json')), \
    'Files do not exist, run 20220810_nd_sac_dataframe.ipynb first'
    
    if get_json:
        sac_json = general.load_json_file_from_S3(loc+'.json', fs)
        return(sac_json)
    else:
        with fs.open(loc+'.csv') as f:
            sac = pd.read_csv(f, index_col=0)
        return(sac)
    
    
def get_electrode_locations(fs, species, subject, exp, session, chans_spc=None):
    '''
    Gathers electrode locations
    
    Parameters
    -------------------
    fs : filesystem object
    species : the species
    subject : the subject
    exp : the experiment
    session : the session to observe
    chans_spc : specific channels to find xyz location
        Use 'all' to get all channels, regardless if they've been determined as 'bad'
    
    Returns
    -------------------
    locs : pandas dataframe giving the coordinates of each electrode
    '''
    full_session_name = session
    if len(session)==12: #For the cases like 201807250001
        session = session[:8]
        
    file = 'nhp-lfp/wcst-preprocessed/rawdata/sub-'+subject+\
           '/sess-'+session+\
           '/session_info/sub-'+subject+\
           '_sess-'+session+\
           '_sessioninfo.json'
    
    assert fs.exists(file)
    
    with fs.open(file) as f:
        locs = general.load_json_file_from_S3(f, fs)
        
    locs = locs['electrode_info']
    locs = pd.DataFrame.from_dict(locs)
    
    if subject=='SA':
        ref_file = 'l2l.jbferre.scratch/electrode_information/'+\
                   'sub-SA_channel_reference_and_company.csv'
        with fs.open(ref_file, 'r') as ff:
            ref = pd.read_csv(ff)
            
        chan = locs.electrode_id.values
        ref_loc = np.empty(len(chan), dtype='<U20')
        com_loc = np.empty(len(chan), dtype='<U20')
        for i,c in enumerate(chan):
            ref_loc[i] = ref[ref['channel']==c].reference.values[0]
            com_loc[i] = ref[ref['channel']==c].company.values[0]
        locs['reference'] = ref_loc
        locs['company'] = com_loc
    
    if chans_spc=='all':
        locs = locs
    elif chans_spc!=None:
        locs = locs[locs['electrode_id'].isin(chans_spc)]
    else:
        bad_chan = analysis.get_bad_channels(species, subject, exp, full_session_name)
        locs = locs[~locs['electrode_id'].isin(bad_chan)]
        
    return(locs)


def get_bad_trials(species, subject, exp, session, return_chans=False):
    """
    Gets the bands for a desired subject and experiment
    
    Parameters
    ----------
    species : the selected species
    subject : the selected subject
    exp : the selected experiment
    session : the selected session
    return_chans : flag indicating whether to return which channels caused the trial to be thrown out
    
    Returns
    -------
    bad_trials : array of bad trials for given session
        or
    bad_t_c : dictionary where keys are the trials and values are arrays of bad channels for that trial
    """
    bad_trials_all = general.load_json_file('sp-'+species+'_sub-'+subject+'_exp-'+exp+'_bad-trials.json')
    
    all_keys = np.array(list(bad_trials_all.keys()))
    session_keys = [k for k in all_keys if subject+'-'+session in k]
    
    if (len(session_keys)>0):
        bad_trials = np.sort([int(k.split('t_')[1]) for k in session_keys])
        
        if return_chans:
            bad_t_c = {}
            for t in bad_trials:
                temp = bad_trials_all[subject+'-'+session+'-t_'+str(t)]
                temp = np.array([t.strip('\'') for t in temp.strip('[]').split(', ')])
                bad_t_c[t] = temp
            return(bad_t_c)
    else:
        print('No bad trials for session '+session+' have been identified')
        bad_trials = np.array([])
    return(bad_trials)


def get_bands(species, subject, exp, idx=0):
    """
    Gets the bands for a desired subject and experiment
    
    Parameters
    ----------
    species : the selected species
    subject : the selected subject
    exp : the selected experiment
    idx : the index of the collection of bands to observe
    
    Returns
    -------
    bands : a list of bands to use in future analysis
    """
    bands = general.load_json_file('sp-'+species+'_sub-'+subject+'_exp-'+exp+'_bands.json')
    bands = bands[str(idx)]
    return(bands)
        

def start_cluster(n_workers=10):
    """
    Starts a dask cluster
    
    Parameters
    ----------
    n_workers : the number of workers you want to start the cluster with
    
    Returns
    -------
    cluster : the cluster itself
    client : the client to the cluster
    """
    gateway = Gateway()
    options = gateway.cluster_options()
    cluster = gateway.new_cluster(options)
    client = cluster.get_client()
    cluster.scale(n_workers)
    return(cluster, client)
    
def get_filenames_human(fs, species, subject, exp, session_id, datatype, params=[]):
    '''
    Finds the filenames for the given session_id and parameters.
    
    Parameters
    ----------------
    fs: file system object
    species : the selected species
    subject: the selected subject
    exp: the selected experiment
    session_id: the session identifier
    datatype: the type of data to retrieve.
        'behavior', 'trial_timestamps', 'electrode_info', 'raw', 'derivative'
    params: list of parameters interested in, in order (e.g. lfp_30)
    
    Returns
    ----------------
    list of filenames
    '''
    file_loc = general.load_json_file('sp-'+species+'_exp-'+exp+'_file_locations.json')
    files = []
    
    if (datatype == 'behavior'):
        files.append(file_loc['raw_loc'] + '/sub-' + subject + '/sess-' + session_id + '/' + file_loc['behavior'][0] +\
                     '/sub-' + subject + '-sess-' + session_id + file_loc['behavior'][1])
    elif (datatype == 'trial_timestamps'):
        files.append(file_loc['raw_loc'] + '/sub-' + subject + '/sess-' + session_id + '/' + file_loc['behavior'][0] +\
                     '/sub-' + subject + '-sess-' + session_id + file_loc['behavior'][2])
    elif (datatype == 'electrode_info'):
        files.append(file_loc['raw_loc'] + '/sub-' + subject + '/sess-' + session_id +\
                     '/sub-' + subject + '-sess-' + session_id + file_loc['electrode_info'])
    elif (datatype == 'raw'):
        temp = fs.ls(file_loc['raw_loc'] + '/sub-' + subject + '/sess-' + session_id + '/' + file_loc['ephys'] + '/')
        files = [f for f in temp if '.mat' in f]
    elif (datatype == 'der'):
        print('TODO')
    else:
        print('Wrong datatype, please input \'behavior\', \'eye\', \'raw\', or \'derivative\'')
    return(files)
    
def get_filenames(fs, species, subject, exp, session_id, datatype, params=[]):
    '''
    Finds the filenames for the given session_id and parameters. 
    If 'ic-rem' is a parameter and the file does not exist but exists without 
        'ic-rem', this function will return the non ic-removed files instead. 
        Assumes that all files for a drive (with 'a' or without 'a') have ic-rem or not
    
    Parameters
    ----------------
    fs: file system object
    species : the selected species
    subject: the selected subject
    exp: the selected experiment
    session_id: the session identifier
    datatype: the type of data to retrieve.
        'behavior', 'object_features', 'eye', 'chan_loc', 'raw', 'derivative', 'derivative_012023'
    params: list of parameters interested in, in order (e.g. lfp_30)
    
    Returns
    ----------------
    list of filenames
    '''
    if species=='human':
        files = get_filenames_human(fs, species, subject, exp, session_id, datatype, params)
        return(files)
    elif species!='nhp':
        print('Incorrect species, either nhp or human')
        return([])
    
    file_loc = general.load_json_file('sp-'+species+'_sub-'+subject+'_exp-'+exp+'_file_locations.json')
    files = []
    
    if (datatype == 'behavior'):
        files.append(file_loc['raw_loc'] + '/sub-' + subject + '/sess-' + session_id + '/' + file_loc['behavior'][0] +\
                     '/sub-' + subject + '_sess-' + session_id + file_loc['behavior'][1])
    elif (datatype == 'object_features'):
        files.append(file_loc['raw_loc'] + '/sub-' + subject + '/sess-' + session_id + '/' + file_loc['behavior'][0] +\
                     '/sub-' + subject + '_sess-' + session_id + file_loc['behavior'][2])
    elif (datatype == 'eye'):
        for eye in file_loc['eye_type']:
            files.append(file_loc['raw_loc'] + '/sub-' + subject + '/sess-' + session_id + '/' + file_loc['eye'][0] +\
                         '/sub-' + subject + '_sess-' + session_id + file_loc['eye'][1] +\
                         eye + file_loc['eye'][2])
    elif (datatype == 'chan_loc'):
        files.append(file_loc['raw_loc'] + '/sub-' + subject + '/sess-' + session_id + '/' + file_loc['chan_loc'][0] +\
                     '/sub-' + subject + '_sess-' + session_id + file_loc['chan_loc'][1])
    elif (datatype == 'raw'):
        chans = file_loc['chan']
        if (not params):
            chans = [c for c in chans if not 'GR' in c]
            chans = [c for c in chans if c not in analysis.get_bad_channels(species, subject, exp, session_id)]
        elif (params[0]=='GR'):
            chans = [c for c in chans if 'GR' in c]
        elif (params[0]=='all'):
            chans = [c for c in chans if not 'GR' in c]
        else:
            print('Parameters need to be [\'GR\'] or [\'all\'] if intended')
            chans = [c for c in chans if not 'GR' in c]
            chans = [c for c in chans if c not in analysis.get_bad_channels(specices, subject, exp, session_id)]
        for ch in chans:
            files.append(file_loc['raw_loc'] + '/sub-' + subject + '/sess-' + session_id + '/' + 'lfp' +\
                         '/sub-' + subject + '_sess-' + session_id + '_chan-' + ch +\
                         file_loc['ephys'][1])
    elif (datatype == 'derivative'):
        chans = file_loc['chan']
        chans = [c for c in chans if not 'GR' in c]
        chans = [c for c in chans if c not in analysis.get_bad_channels(species, subject, exp, session_id)]
        for ch in chans:
            files.append(file_loc['der_loc'] + '/sub-' + subject + '/sess-' + session_id + '/' + file_loc['ephys'][0] +\
                         '/' + '/'.join(params) + '/sub-' + subject + '_sess-' + session_id + '_chan-' + ch +\
                         '_' + '_'.join(params) + file_loc['ephys'][1])
    elif (datatype == 'derivative_012023'):
        chans = file_loc['chan']
        chans = [c for c in chans if not 'GR' in c]
        chans = [c for c in chans if c not in analysis.get_bad_channels(species, subject, exp, session_id)]
        for ch in chans:
            files.append(file_loc['der_loc'] + '_012023/sub-' + subject + '/sess-' + session_id + '/lfp' +\
                         '/' + '/'.join(params) + '/sub-' + subject + '_sess-' + session_id + '_chan-' + ch +\
                         '_' + '_'.join(params) + file_loc['ephys'][1])
    else:
        print('Wrong datatype, please input \'behavior\', \'eye\', \'raw\', or \'derivative\'')
     
    if (datatype == 'raw' or datatype == 'derivative' or datatype == 'derivative_012023'):
        files_1 = [f for f in files if 'a' in f.split('_chan-')[1].split('_')[0].split('.')[0]]
        files_2 = [f for f in files if 'a' not in f.split('_chan-')[1].split('_')[0].split('.')[0]]
        bad_idx = []
        for i in range(len(files_1)):
            if (not fs.exists(files_1[i])):
                f_no_ic = ''.join(''.join(files_1[i].split('/ic-rem')).split('_ic-rem'))
                if (fs.exists(f_no_ic)):
                    print('Files in Frontal drive do not have ic components removed, trying non-ic removed files...')
                    files_1 = [''.join(''.join(f.split('/ic-rem')).split('_ic-rem')) for f in files_1]
                    files_1 = [f for f in files_1 if fs.exists(f)]
                    break
                else:
                    bad_idx.append(i)
        files_1 = [files_1[i] for i in range(len(files_1)) if i not in bad_idx]
        bad_idx = []
        for i in range(len(files_2)):
            if (not fs.exists(files_2[i])):
                f_no_ic = ''.join(''.join(files_2[i].split('/ic-rem')).split('_ic-rem'))
                if (fs.exists(f_no_ic)):
                    print('Files in Temporal drive do not have ic components removed, trying non-ic removed files...')
                    files_2 = [''.join(''.join(f.split('/ic-rem')).split('_ic-rem')) for f in files_2]
                    files_2 = [f for f in files_2 if fs.exists(f)]
                    break
                else:
                    bad_idx.append(i)
        files_2 = [files_2[i] for i in range(len(files_2)) if i not in bad_idx]
        files = files_2 + files_1
    else:
        for i in range(len(files)):
            if (not fs.exists(files[i])):
                print('File doesn\'t exist: ' + files[i])
        files = [f for f in files if fs.exists(f)]
    return(files)

def get_subjects(fs, species, exp):
    '''
    Gets a list of subjects for some species/experiment
    Note that for the NHP, there is currently only one subject online, this will change later
    
    Parameters
    ----------
    fs : file system
    species : the desired species
    exp : the experiment
    
    Returns
    -------
    subjects : list of subjects
    '''
    if species=='nhp':
        file_loc = general.load_json_file('sp-'+species+'_sub-SA_exp-'+exp+'_file_locations.json')
        subjects = np.array([file_loc['sub']])
    elif species=='human':
        file_loc = general.load_json_file('sp-'+species+'_exp-'+exp+'_file_locations.json')
        subjects = np.array(file_loc['sub'])
    else:
        print('Incorrect species, either nhp or human')
        subjects = []
    return(subjects)

def get_bipole_info(fs, species, subject, exp, session, ending='csv'):
    '''
    Gets bipole information
    
    Parameters
    ----------
    fs : filesystem object
    species : the species
    subject : the subject
    exp : the experiment
    session : the session
    ending : ending of file name, can be csv or json for the respective file to retrieve
    
    Returns
    -------
    bipole : either pandas dataframe or json file with bipole information
    '''
    filename = 'nhp-lfp/wcst-preprocessed/derivatives_bipole/sub-'+subject+\
               '/sess-'+session+\
               '/bipole-info/sub-'+subject+\
               '_sess-'+session+\
               '_bipole-info.'+ending
    assert fs.exists(filename), 'filename does not exist'
    
    if ending=='csv':
        with fs.open(filename, 'r') as ff:
            bipole = pd.read_csv(ff, index_col=0)
            bipole = bipole.reset_index(drop=True)
    else:
        bipole = general.load_json_file_from_S3(filename, fs)
            
    return(bipole)

def get_raw_filetype(species, subject, exp, session):
    '''
    Retrieves the filetype of the raw data
    
    Parameters
    -----------
    species : species selected, nhp or human
    subject : subject selected
    exp : experiment selected
    all_ids : flag indicating whether to include all sessions or just \'good\' sessions
    
    Returns
    -----------
    sess_ids : session ids
    '''
    if species=='nhp':
        file_loc = general.load_json_file('sp-'+species+'_sub-SA_exp-'+exp+'_file_locations.json')
        filetype = file_loc['raw_filetype']
    elif species=='human':
        file_loc = general.load_json_file('sp-'+species+'_exp-'+exp+'_file_locations.json')
        filetype = file_loc['raw_filetype']
    else:
        print('Incorrect species, either nhp or human')
        filetype = ''
    return filetype

def get_session_ids(species, subject, exp, all_ids=False):
    '''
    Finds and returns all of the possible session ids.
    
    Parameters
    -----------
    species : species selected, nhp or human
    subject : subject selected
    exp : experiment selected
    all_ids : flag indicating whether to include all sessions or just \'good\' sessions
    
    Returns
    -----------
    sess_ids : session ids
    '''
    if species=='nhp':
        if (all_ids):
            file_loc = general.load_json_file('sp-'+species+'_sub-'+subject+'_exp-'+exp+'_file_locations.json')
            sess_ids = file_loc['sess']
        else:
            sessions = general.load_json_file('sp-'+species+'_sub-'+subject+'_exp-'+exp+'_good_sessions.json')
            sess_ids = sessions['SA']
    elif species=='human':
        file_loc = general.load_json_file('sp-'+species+'_exp-'+exp+'_file_locations.json')
        temp = [f for f in file_loc['sess'] if f.split('sub-')[1].split('_sess')[0]==subject]
        sess_ids = np.array([f.split('_sess-')[1] for f in temp])
    else:
        print('Species should either be nhp or human')
        sess_ids = []
    return sess_ids

def get_all_chans(species, subject, exp, params=None):
    '''
    Gets all of the channels possible for any day.
    
    Parameters
    ----------------
    species : species selected
    subject: subject selected
    exp: experiment selected
    Params: If \'GR\', will return the GR channel names.
            Otherwise, will return only the channels in the regular drives
            
    Returns
    ----------------
    chans: the channel names sorted
    '''
    file_loc = general.load_json_file('sp-'+species+'_sub-'+subject+'_exp-'+exp+'_file_locations.json')
    chans = file_loc['chan']
    if (not params):
        chans = [c for c in chans if not 'GR' in c]
        chans = np.array(chans)[analysis.reorder_chans(chans)]
    elif (params=='GR'):
        chans = [c for c in chans if 'GR' in c]
    else:
        print('Input needs to be \'GR\' if intended')
        chans = [c for c in chans if not 'GR' in c]
        chans = np.array(chans)[analysis.reorder_chans(chans)]
    return(list(chans))

def get_object_features(fs, species, subject, exp, session):
    """
    Gets the object features csv file and builds a dataframe
    
    Parameters
    ----------------
    fs : s3 file system object
    species : species selected
    subject : subject selected
    exp : experiment selected
    session : session selected
    
    Returns
    ----------------
    of : dataframe of object features
    """
    filename = get_filenames(fs, species, subject, exp, session, 'object_features')[0]
    if fs.exists(filename):
        with fs.open(filename) as f:
            of = pd.read_csv(f)
            return(of)
    else:
        print("Object Features file does not exist")
        return None

def get_behavior(fs, sp, sub, exp, sess_id, import_obj_features=True):
    """
    Gets the behavior file and builds a dataframe to help analyze the data.
    DOES NOT WORK ON ANYTHING BUT NHP-WCST
    
    Parameters
    ---------------
    fs: filesystem object
    sp : the species selected
    sub: the subject selected
    exp: the experiment selected
    sess_id: the session to obtain the behavior file from
    
    Returns
    ----------------
    df: dataframe of behavior
    """
    if sp=='nhp':
        file_beh = get_filenames(fs, sp, sub, exp, sess_id, 'behavior')
        if (file_beh):
            file_beh = file_beh[0]
        else:
            return (file_beh)

        with fs.open(file_beh) as f:
            df = pd.read_csv(f, header=None,names = ['time','encode'])

        df = _beh_trim(df) #removes anything before and after real trials
        if (_beh_check(df)):
            return(df) #Check if right number of data
        df = _beh_add_trial_info(df) #Adds trial, relative_trials, group, rule, rule dim, response columns
        df = _beh_add_last_cor(df) #Add last correct dim
        df = _beh_change_encode_labels(df) #rename certain encodes
        df = _beh_add_action_column(df) #adds column for actions
        _beh_check_act(df) #Check action column to make sure it has everything
        _beh_add_single_trial(df) #Add column that specifies weird things that occur in a group
        _beh_check_last_cor(df) #Check if there's incomplete groups
        df = _beh_ignore(df)
        df = _beh_bad_trials(df, sp, sub, exp, sess_id)
        if import_obj_features:
            df = _beh_add_obj_features(df, fs, sp, sub, exp, sess_id)
    elif sp=='human':
        file = get_filenames(fs, sp, sub, exp, sess_id, 'behavior')[0]
        with fs.open(file) as f:
            df1 = pd.read_csv(f)
        file = get_filenames(fs, sp, sub, exp, sess_id, 'trial_timestamps')[0]
        with fs.open(file) as f:
            df2 = pd.read_csv(f)
        df = pd.concat((df1, df2), axis=1)
    else:
        print('Species either should be nhp or human')
    return(df)

def get_eye_data(fs, sp, sub, exp, sess_id, sample=True):
    """
    Retrieves eye data for a given session and subject
    
    Parameters
    ---------------
    fs: filesystem object
    sp : species selected
    sub: subject performing task
    exp: the experiment selected
    sess_id: the session desired
    sample: if true will resample eye data to match neural data.
        Assumes 2000 Hz sampling
    
    Returns
    ---------------
    eye[0]: pupil size
    eye[1]: horizontal displacement
    eye[2]: vertical displacement
    """
    file_eye = get_filenames(fs, sp, sub, exp, sess_id, 'eye')
    if (not file_eye):
        return (file_eye, file_eye, file_eye)
    
    eye = []
    for i in range(3):
        temp = general.open_h5py_file(file_eye[i], fs)
        if (sample):
            eye.append(temp[::2])
        else:
            eye.append(temp)
    
    return(eye[0], eye[1], eye[2])

def get_channel_locations(fs, sp, sub, exp, sess_id):
    """
    Gets the channel locations
    
    Parameters
    ---------------
    fs: filesystem object
    sp: the species selected
    sub: the subject selected
    exp: the experiment selected
    sess_id: the session to obtain the behavior file from
    
    Returns
    ----------------
    cl: dataframe of channel locations. nan is used for channels with unknown locations
    """
    if sp=='nhp':
        print('TODO')
        cl = []
#         file_cl = get_filenames(fs, sp, sub, exp, sess_id, 'chan_loc')
#         if (file_cl):
#             file_cl = file_cl[0]
#         else:
#             return (file_cl)

#         with fs.open(file_cl) as f:
#             cl = pd.read_csv(f, header=None,names = ['chan','loc'])
    elif sp=='human':
        file = get_filenames(fs, sp, sub, exp, sess_id, 'electrode_info')[0]
        with fs.open(file) as f:
            cl = pd.read_csv(f)
    else:
        print('species should either be nhp or human')
        cl = []
    
    return(cl)


#Unsure if used

#import tools
#import tools_specific
#import gcsfs
#import json
#import scipy.io

#def startup_general(fs):
#    """
#    vo is not general
#    """
#    with open('startup.json') as json_file:
#        metadata = json.load(json_file)
#        
#    df = tools.get_behavior_df(metadata['behavior_file'], fs, int(metadata['behavior_t_remove_from_end']), #metadata['behavior_remove'], metadata['behavior_unique'])
    
#    obj0 = scipy.io.loadmat('obj0.mat')['trl_test0'][:960]
#    obj1 = scipy.io.loadmat('obj1.mat')['trl_test1'][:960]
#    obj2 = scipy.io.loadmat('obj2.mat')['trl_test2'][:960]
#    obj3 = scipy.io.loadmat('obj3.mat')['trl_test3'][:960]
#    vo = tools_specific.convert_obj_to_num(tools_specific.get_viewed_obj(df, obj0, obj1, obj2, obj3))
    
#    df['vo_0'] = -1
#    df['vo_1'] = -1
#    df['vo_2'] = -1
    
#    idx = df[(df['encode']==2300) | (df['encode']==2500) | (df['encode']==2700) | (df['encode']==2900)].index.values
    
#    df.loc[idx,'vo_0'] = vo[:,0]
#    df.loc[idx,'vo_1'] = vo[:,1]
#    df.loc[idx,'vo_2'] = vo[:,2]
    
#    return(df, vo)
    
    

# Helper functions
    
def _beh_trim(df):
    """
    Removes anything before and after first and last trial in behavior file.
    """
    if (sum(((df['encode']==200) | (df['encode']==206))) == 0):
        print('No trials in this file')
        return(df)
    
    time_response_0 = df[df['encode'].isin([200,202,204,206])].time.values[0]
    time_response_1 = df[df['encode'].isin([200,202,204,206])].time.values[-1]
    
    if (sum((df['encode']==150) & (df['time']<time_response_0)) == 0):
        start = df[df['encode']==150].time.values[0]
    else:
        start = df[(df['encode']==150) & (df['time']<time_response_0)].time.values[-1]
    
    if (sum((df['encode']==151) & (df['time']>time_response_1)) == 0):
        end = df[df['encode']==151].time.values[-1]
    else:
        end = df[(df['encode']==151) & (df['time']>time_response_1)].time.values[0]
    
    df = df[(df['time']>=start) & (df['time']<=end)]
    return(df)

def _beh_check(df):
    """
    Checks if there's the correct number of rules, trial starts, trial ends, and responses
    in behavior file.
    """
    num_trials = len(df[df['encode']==150])
    if(
        len(df[df['encode'].between(2000,2013)]) != num_trials or
        len(df[df['encode'].between(199,207)]) != num_trials or 
        len(df[df['encode']==151]) != num_trials
    ):
        print('Incomplete data, check 1, stopping dataframe modification...')
        return(True)
    else:
        return(False)
    
def _beh_add_trial_info(df):
    """
    Adds columns to behavior dataframe.
    Adds trials, relative trials (forward and backward), group number, rules,
    rule dimensions, and responses
    """
    trials = []
    rel_trials = []
    all_rules = []
    responses = []
    group = []
    rule = df[df['encode'].between(2000,2013)].encode.values
    response = df[df['encode'].between(199,207)].encode.values
    trial = -1
    rel_trial = -1
    g = 0
    for i in range(len(df['encode'].values)):
        if(df['encode'].values[i] == 150):
            trial = trial + 1
            rel_trial = rel_trial + 1
        trials.append(trial)
        all_rules.append(rule[trial])
        responses.append(response[trial])
        if(all_rules[i] != all_rules[i - 1]):
            g = g + 1
            rel_trial = 0
        group.append(g)
        rel_trials.append(rel_trial)
    all_rules = np.array(all_rules) - 2001
    df['group'] = group
    df['trial'] = trials
    df['trialRel'] = rel_trials
    groupNums = df[df['encode']==150].group.values
    rel_b = np.array(rel_trials) - np.hstack([Counter(groupNums)[g]*np.ones((Counter(group)[g])) for g in np.unique(groupNums)])
    df['trialRelB'] = [int(t) for t in rel_b]
    df['rule'] = all_rules
    df['ruleDim'] = [int(r) for r in all_rules / 4]
    df['response'] = responses
    return(df)

def _beh_add_last_cor(df):
    """
    Adds lastcorrect column to behavior dataframe.
    """
    df['lastCorrect']=0
    for g in np.unique(df[df['encode']==150].group.values):
        temp = np.unique(df[(df['group']==g) & (df['response']==200)].trial.values)
        if (len(temp) < 8):
            tnum = temp
        elif (np.array_equal(temp[-8:], range(temp[-8],temp[-8]+8))):
            tnum = temp[-8:]
        elif (len(temp) < 16):
            tnum = temp
        else:
            tnum = temp[-16:]
        idx = df[df['trial'].isin(tnum)].index.values
        df.loc[idx,'lastCorrect']=1
    return(df)

def _beh_change_encode_labels(df):
    """
    Changes duplicated encodes.
    23 goes to 23 and 2300 (similar for 24-30, 11, and 8)
    """
    idx = []
    for i in range(23, 31):
        a = df[df.loc[:, df.columns != 'time'].duplicated()].encode == i
        idx.append(a.index.values[a.values])
    a = df[df.loc[:, df.columns != 'time'].duplicated(keep='last')].encode == 11
    idx11 = a.index.values[a.values]
    a = df[df.loc[:, df.columns != 'time'].duplicated(keep='last')].encode == 8
    idx8 = a.index.values[a.values]
    for i in range(len(idx)):
        df.loc[idx[i], 'encode'] = int((i + 23) * 100)
    df.loc[idx11,'encode'] = 1100
    df.loc[idx8,'encode'] = 800
    return(df)

def _beh_add_action_column(df):
    """
    Adds action column that tells when certain actions occur.
    """
    df.insert(2, 'act', 0)
    idx = df[df['encode']==35].index.values
    df.loc[idx, 'act'] = 'cross_on'
    idx = df[df['encode']==36].index.values
    df.loc[idx, 'act'] = 'cross_off'
    idx = df[df['encode']==11].index.values
    df.loc[idx, 'act'] = 'wait_fix'
    idx = df[df['encode']==8].index.values
    df.loc[idx, 'act'] = 'cross_fix'
    idx = df[df['encode']==-10].index.values
    df.loc[idx, 'act'] = 'end_required_fix'
    idx = df[df['encode']==1100].index.values
    df.loc[idx, 'act'] = 'wait_fix_break'
    idx = df[df['encode']==800].index.values
    df.loc[idx, 'act'] = 'cross_fix_break'
    idx = df[df['encode']==29].index.values
    df.loc[idx, 'act'] = 'obj_on'
    idx = df[df['encode']==30].index.values
    df.loc[idx, 'act'] = 'obj_off'
    idx = df[(df['encode']==200) | (df['encode']==206)].index.values
    df.loc[idx, 'act'] = 'fb'
    idx = df[df['encode']==202].index.values
    df.loc[idx, 'act'] = 'late'
    idx = df[df['encode']==204].index.values
    df.loc[idx, 'act'] = 'no_cross_fix'
    idx = df[(df['encode']==2300) | (df['encode']==2500) | (df['encode']==2700) | (df['encode']==2900)].index.values
    df.loc[idx, 'act'] = 'obj_fix_break'
    idx = df[(df['encode']==200) | (df['encode']==206)].index.values-1
    df.loc[idx, 'act'] = 'obj_fix'
    return(df)

def _beh_check_act(df):
    """
    Performs second check which checks if the actions have appropriate numbers.
    """
    num_trials = len(df[df['encode']==150])
    num_no_fix_trials = len(df[df['act']=='no_cross_fix'])
    num_incomplete_trials = len(df[df['act']=='late']) + len(df[df['act']=='no_cross_fix'])
    if(
        len(df[df['act']=='cross_on']) != num_trials or
        len(df[df['act']=='cross_off']) + num_no_fix_trials != num_trials or #The cross isn't turned off in some cases
        len(df[df['act']=='wait_fix']) != num_trials or
        len(df[(df['act']=='cross_fix') & (df['response']!=204)]) != num_trials - num_no_fix_trials or
        len(df[df['act']=='obj_on']) + num_no_fix_trials != num_trials or
        len(df[df['act']=='obj_off']) + num_no_fix_trials != num_trials or
        len(df[df['act']=='fb']) + num_incomplete_trials != num_trials or
        len(df[df['act']=='obj_fix']) + num_incomplete_trials != num_trials
    ):
        print('Incomplete data in act column, check 2')
    return

def _beh_add_single_trial(df):
    """
    Adds column that specifies if single trials or random rule changes occur.
    0 means that the group is normal (8 of 8, or 16 of 20)
    Number means that number of trials occured in a group
    -1 means that enough at least 8 trials occured, but rule changed spontaneously
    """
    df['badGroup'] = 0
    for g in np.unique(df.group.values)[:-1]: #last group can't have problems
        num_trials = df[df['group']==g].trialRel.values[-1] + 1
        idx = df[df['group']==g].index.values
        if (num_trials < 8):
            df.loc[idx, 'badGroup'] = num_trials
        elif (sum((df['group']==g) & (df['lastCorrect']) & (df['encode']==150)) not in [8,16]):
            df.loc[idx, 'badGroup'] = -1
    return(df)

def _beh_check_last_cor(df):
    """
    Check if there are groups where there are neither regular groups or single trial groups.
    """
    breaks = analysis.beh_get_breaks(df)
    if (breaks.size > 0):
        breaks = [int(b-0.5) for b in breaks]
        bad = df[(df['badGroup'] != 0) & (df['badGroup'] != 1)]
        if (len(bad) > 0):
            for g in np.unique(bad.group.values):
                if (bad[(bad['group']==g)].trial.values[-1] not in breaks):
                    print('Incomplete groups exist')
                    return
    else:
        if (sum((df['badGroup'] != 0) & (df['badGroup'] != 1)) > 0):
            print('Incomplete groups exist')
            return
        
def _beh_ignore(df):
    """
    Adds a column of zeros with ones whenever something should be ignored.
    Currently ignores:
        Any saccade to an object with another saccade within 50 timesteps
    """
    df['ignore'] = np.zeros((len(df)), dtype=int)
    time_delay = []
    for t in np.unique(df.trial.values):
        temp = df[(df['act'].isin(['obj_fix_break', 'obj_fix'])) & (df['trial']==t)].time.values
        idx = df[df['time'].isin(temp[np.argwhere(temp[1:]-temp[:-1] < 50)[:,0]])].index.values
        df.loc[idx, 'ignore']=1
    return(df)

def _beh_bad_trials(df, species, subject, exp, session):
    """
    Adds column with bad trials to dataframe
    """
    bt = get_bad_trials(species, subject, exp, session)
    df['badTrials'] = np.zeros(len(df), dtype=int)
    idx = df[df['trial'].isin(bt)].index.values
    df.loc[idx, 'badTrials'] = 1
    return(df)

def _beh_add_obj_features(df, fs, species, subject, exp, session):
    '''
    Adds object features, including shape, pattern, and color and their locations.
    Locations are in visual degrees
    '''
    of = get_object_features(fs, species, subject, exp, session)
    if of is None:
        return(df)
    else:
        df_sub = df[df['act']=='cross_on']
        of_sub = of[of['TrialNumber'].isin(np.unique(df.trial.values))]
        
        res_dict = {200:'Correct', 206:'Incorrect', 202:'Late', 204:'NoFixation'}
        if np.all(np.array([res_dict[r] for r in df_sub.response.values])==of_sub.Response.values):
            colName = []
            for i in ['0','1','2','3']:
                for j in ['_x', '_y']:
                    df['Item'+i+j+'Pos'] = np.NaN
                    colName.append('Item'+i+j+'Pos')
            for i in ['0','1','2','3']:
                for j in ['Shape', 'Color', "Pattern"]:
                    df['Item'+i+j] = ""
                    colName.append('Item'+i+j)
                    
            for t in np.unique(df.trial.values):
                of_row = of[of['TrialNumber']==t]
                idx = df[df['trial']==t].index.values
                
                for col in colName:
                    df.loc[idx, col] = of_row[col].values[0]
        else:
            print('Responses are NOT equal between behavior and object features, ignoring object features...')
            return(df)
    return(df)