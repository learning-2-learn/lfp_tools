from lfp_tools import general
from lfp_tools import analysis
import configparser
import os.path as op
import s3fs

def get_fs():
    CP = configparser.ConfigParser()
    CP.read_file(open(op.join(op.expanduser('~'), '.aws', 'credentials')))
    CP.sections()
    ak = CP.get('default', 'AWS_ACCESS_KEY_ID')
    sk = CP.get('default', 'AWS_SECRET_ACCESS_KEY')
    fs = s3fs.S3FileSystem(key=ak, secret=sk)
    return(fs)

def get_filenames(fs, session_id, subject, datatype, params=[]):
    '''
    Finds the filenames for the given session_id and parameters
    
    Parameters
    ----------------
    fs: file system object
    session_id: the session identifier
    params: list of parameters interested in, in order (e.g. lfp_30)
    
    Returns
    ----------------
    list of filenames
    '''
    file_loc = general.load_json_file('file_locations.json')
    files = []
    
    if (datatype == 'behavior'):
        files.append(file_loc['raw_loc'] + '/sess-' + session_id + '/' + file_loc['behavior'][0] +\
                     '/sub-' + subject + '_sess-' + session_id + file_loc['behavior'][1])
    elif (datatype == 'eye'):
        for eye in file_loc['eye_type']:
            files.append(file_loc['raw_loc'] + '/sess-' + session_id + '/' + file_loc['eye'][0] +\
                         '/sub-' + subject + '_sess-' + session_id + file_loc['eye'][1] +\
                         eye + file_loc['eye'][2])
    elif (datatype == 'raw'):
        chans = file_loc['chan']
        if (not params):
            chans = [c for c in chans if not 'GR' in c]
            chans = [c for c in chans if c not in analysis.get_bad_channels(subject, session_id)]
        elif (params[0]=='GR'):
            chans = [c for c in chans if 'GR' in c]
        elif (params[0]=='all'):
            chans = [c for c in chans if not 'GR' in c]
        else:
            print('Parameters need to be [\'GR\'] or [\'all\'] if intended')
            chans = [c for c in chans if not 'GR' in c]
            chans = [c for c in chans if c not in analysis.get_bad_channels(subject, session_id)]
        for ch in chans:
            files.append(file_loc['raw_loc'] + '/sess-' + session_id + '/' + file_loc['ephys'][0] +\
                         '/sub-' + subject + '_sess-' + session_id + '_chan-' + ch +\
                         file_loc['ephys'][1])
    elif (datatype == 'derivative'):
        chans = file_loc['chan']
        chans = [c for c in chans if c not in analysis.get_bad_channels(subject, session_id)]
        for ch in chans:
            files.append(file_loc['der_loc'] + '/sess-' + session_id + '/' + file_loc['ephys'][0] +\
                         '/' + '/'.join(params) + '/sub-' + subject + '_sess-' + session_id + '_chan-' + ch +\
                         '_' + '_'.join(params) + file_loc['ephys'][1])
    else:
        print('Wrong datatype, please input \'behavior\', \'eye\', \'raw\', or \'derivative\'')
    
    for f in files:
        if (not fs.exists(f)):
            print('File doesn\'t exist: ' + f)
    files = [f for f in files if fs.exists(f)]
    return(files)

