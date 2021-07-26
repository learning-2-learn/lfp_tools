from lfp_tools import general
from lfp_tools import analysis
from collections import Counter
import pandas as pd
import numpy as np
import s3fs

from dask_gateway import Gateway

def get_fs():
    """
    Gets the filesystem object.
    
    Returns
    -------
    fs : filesystem object
    """
    fs = s3fs.S3FileSystem()
    return(fs)


def get_bands(subject, exp, idx=0):
    """
    Gets the bands for a desired subject and experiment
    
    Parameters
    ----------
    subject : the selected subject
    exp : the selected experiment
    idx : the index of the collection of bands to observe
    
    Returns
    -------
    bands : a list of bands to use in future analysis
    """
    bands = general.load_json_file('sub-'+subject+'_exp-'+exp+'_bands.json')
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
    
    
def get_filenames(fs, subject, exp, session_id, datatype, params=[]):
    '''
    Finds the filenames for the given session_id and parameters. 
    If 'ic-rem' is a parameter and the file does not exist but exists without 
        'ic-rem', this function will return the non ic-removed files instead. 
        Assumes that all files for a drive (with 'a' or without 'a') have ic-rem or not
    
    Parameters
    ----------------
    fs: file system object
    subject: the selected subject
    exp: the selected experiment
    session_id: the session identifier
    datatype: the type of data to retrieve.
        'behavior', 'eye', 'chan_loc', 'raw', 'derivative'
    params: list of parameters interested in, in order (e.g. lfp_30)
    
    Returns
    ----------------
    list of filenames
    '''
    file_loc = general.load_json_file('sub-'+subject+'_exp-'+exp+'_file_locations.json')
    files = []
    
    if (datatype == 'behavior'):
        files.append(file_loc['raw_loc'] + '/sub-' + subject + '/sess-' + session_id + '/' + file_loc['behavior'][0] +\
                     '/sub-' + subject + '_sess-' + session_id + file_loc['behavior'][1])
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
            chans = [c for c in chans if c not in analysis.get_bad_channels(subject, exp, session_id)]
        elif (params[0]=='GR'):
            chans = [c for c in chans if 'GR' in c]
        elif (params[0]=='all'):
            chans = [c for c in chans if not 'GR' in c]
        else:
            print('Parameters need to be [\'GR\'] or [\'all\'] if intended')
            chans = [c for c in chans if not 'GR' in c]
            chans = [c for c in chans if c not in analysis.get_bad_channels(subject, exp, session_id)]
        for ch in chans:
            files.append(file_loc['raw_loc'] + '/sub-' + subject + '/sess-' + session_id + '/' + file_loc['ephys'][0] +\
                         '/sub-' + subject + '_sess-' + session_id + '_chan-' + ch +\
                         file_loc['ephys'][1])
    elif (datatype == 'derivative'):
        chans = file_loc['chan']
        chans = [c for c in chans if not 'GR' in c]
        chans = [c for c in chans if c not in analysis.get_bad_channels(subject, exp, session_id)]
        for ch in chans:
            files.append(file_loc['der_loc'] + '/sub-' + subject + '/sess-' + session_id + '/' + file_loc['ephys'][0] +\
                         '/' + '/'.join(params) + '/sub-' + subject + '_sess-' + session_id + '_chan-' + ch +\
                         '_' + '_'.join(params) + file_loc['ephys'][1])
    else:
        print('Wrong datatype, please input \'behavior\', \'eye\', \'raw\', or \'derivative\'')
     
    if (datatype == 'raw' or datatype == 'derivative'):
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

def get_session_ids(subject, exp, all_ids=False):
    '''
    Finds and returns all of the possible session ids.
    
    Parameters
    -----------
    subject : subject selected
    exp : experiment selected
    all_ids : flag indicating whether to include all sessions or just \'good\' sessions
    
    Returns
    -----------
    sess_ids : session ids
    '''
    if (all_ids):
        file_loc = general.load_json_file('sub-'+subject+'_exp-'+exp+'_file_locations.json')
        return(file_loc['sess'])
    else:
        sessions = general.load_json_file('sub-'+subject+'_exp-'+exp+'_good_sessions.json')
        return(sessions['SA'])

def get_all_chans(subject, exp, params=None):
    '''
    Gets all of the channels possible for any day.
    
    Parameters
    ----------------
    subject: subject selected
    exp: experiment selected
    Params: If \'GR\', will return the GR channel names.
            Otherwise, will return only the channels in the regular drives
            
    Returns
    ----------------
    chans: the channel names sorted
    '''
    file_loc = general.load_json_file('sub-'+subject+'_exp-'+exp+'_file_locations.json')
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

def get_behavior(fs, sub, exp, sess_id):
    """
    Gets the behavior file and builds a dataframe to help analyze the data.
    DOES NOT WORK ON ANYTHING BUT NHP-WCST
    
    Parameters
    ---------------
    fs: filesystem object
    sub: the subject selected
    exp: the experiment selected
    sess_id: the session to obtain the behavior file from
    
    Returns
    ----------------
    df: dataframe of behavior
    """
    file_beh = get_filenames(fs, sub, exp, sess_id, 'behavior')
    if (file_beh):
        file_beh = file_beh[0]
    else:
        return (file_beh)
    
    with fs.open(file_beh) as f:
        df = pd.read_csv(f, header=None,names = ['time','encode'])
    
    df = _beh_trim(df) #removes anything before and after real trials
    df = _beh_special(df, sess_id, sub) # Does any special case changes
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
    return(df)

def get_eye_data(fs, sub, exp, sess_id, sample=True):
    """
    Retrieves eye data for a given session and subject
    
    Parameters
    ---------------
    fs: filesystem object
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
    file_eye = get_filenames(fs, sub, exp, sess_id, 'eye')
    if (not file_eye):
        return (file_eye, file_eye, file_eye)
    
    eye = []
    for i in range(3):
        temp = general.open_h5py_file(file_eye[i], fs)
        if (sample):
            eye.append(temp[::2])
        else:
            eye.append(temp)
    
    print('Warning, no renormalization was done; still needs to be implemented')
    return(eye[0], eye[1], eye[2])

def get_channel_locations(fs, sub, exp, sess_id):
    """
    Gets the channel locations
    
    Parameters
    ---------------
    fs: filesystem object
    sub: the subject selected
    exp: the experiment selected
    sess_id: the session to obtain the behavior file from
    
    Returns
    ----------------
    cl: dataframe of channel locations. nan is used for channels with unknown locations
    """
    file_cl = get_filenames(fs, sub, exp, sess_id, 'chan_loc')
    if (file_cl):
        file_cl = file_cl[0]
    else:
        return (file_cl)
    
    with fs.open(file_cl) as f:
        cl = pd.read_csv(f, header=None,names = ['chan','loc'])
    
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
    
    time_response_0 = df[(df['encode']==200) | (df['encode']==206)].time.values[0]
    time_response_1 = df[(df['encode']==200) | (df['encode']==206)].time.values[-1]
    
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

def _beh_special(df, sess, sub):
    """
    Corrects specific sessions as special cases in behavior file.
    """
    if (sess == '20180712' and sub == 'SA'):
        idx=df[df['encode']==1800].index.values
        df.loc[idx, 'encode']=2012
    if (sess == '20180817' and sub == 'SA'):
        idx=df[df['encode']==1800].index.values
        df.loc[idx, 'encode']=2012
    if (sess == '20180928' and sub == 'SA'):
        idx=df[df['encode']==3].index.values
        df.loc[idx, 'encode']=35
    if (sess == '20181019' and sub == 'SA'):
        df = df.drop(np.arange(9060, 9071))
        df.loc[9071, 'time'] = -1
        df.loc[9072, 'time'] = -1
    if (sess == '20180803' and sub == 'SA'):
        df = df.drop(np.arange(35259, 35261))
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
    idx = df[df['encode']==1100].index.values
    df.loc[idx, 'act'] = 'wait_fix_break'
    idx = df[df['encode']==800].index.values
    df.loc[idx, 'act'] = 'cross_fix_break'
    idx = df[df['encode']==23].index.values
    df.loc[idx, 'act'] = 'obj_on'
    idx = df[df['encode']==26].index.values
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