from lfp_tools import general
from lfp_tools import analysis
from lfp_tools import startup
import numpy as np
import pandas as pd
#import configparser
#import os.path as op

#def get_fs():
#    CP = configparser.ConfigParser()
#    CP.read_file(open(op.join(op.expanduser('~'), '.aws', 'credentials')))
#    CP.sections()
#    ak = CP.get('default', 'AWS_ACCESS_KEY_ID')
#    sk = CP.get('default', 'AWS_SECRET_ACCESS_KEY')
#    fs = s3fs.S3FileSystem(key=ak, secret=sk)
#    return(fs)

def get_brain_areas(fs, subject, exp, session):
    '''
    Gets the brain areas of all channels
    
    Parameters
    ------------------------
    fs : filesystem object
    subject : the subject
    exp : the experiment
    session : the session
    
    Returns
    -------------------
    chans : pandas dataframe that includes channels and brain areas
    '''
    with fs.open('l2l.jbferre.scratch/sub-'+subject+'_sess-'+session+'_channellocations.csv') as f:
        chans = pd.read_csv(f, names=['ch', 'area1', 'area2']).fillna('Unk')
    return(chans)

from sklearn.cluster import AgglomerativeClustering
def cluster_chans_by_coords(fs, subject, exp, session, n_clusters_t, n_clusters_a, chans_spc=None):
    '''
    Clusters the channels based on their coordinates.
    
    Parameters
    -------------------
    fs : filesystem object
    subject : the subject
    exp : the experiment
    session : the session to observe
    n_clusters_t : the number of clusters in temporal drive to separate the channels into
    n_clusters_a : the number of clusters in anterior drive to separate the channels into
    chans_spc : specific channels to include in clustering algorithm
    
    Returns
    -------------------
    label_dict : dictionary containing the channels and their labels
    '''
    with fs.open('l2l.jbferre.scratch/epos'+subject+session+'_post.csv') as f:
        coords_t = pd.read_csv(f, names=['x', 'y', 'z']).fillna(0)
    with fs.open('l2l.jbferre.scratch/epos'+subject+session+'_ant.csv') as f:
        coords_a = pd.read_csv(f, names=['x', 'y', 'z']).fillna(0)
    coords_a['chan'] = [str(c)+'a' for c in np.arange(1,len(coords_a)+1)]
    coords_t['chan'] = [str(c) for c in np.arange(1,len(coords_t)+1)]
    bad_chan = analysis.get_bad_channels(subject, exp, session)
    coords_a = coords_a[~coords_a['chan'].isin(bad_chan)]
    coords_t = coords_t[~coords_t['chan'].isin(bad_chan)]
    
    if (chans_spc != None):
        coords_a = coords_a[coords_a['chan'].isin(chans_spc)]
        coords_t = coords_a[coords_a['chan'].isin(chans_spc)]
    
    keys_a = coords_a.chan.values
    keys_t = coords_t.chan.values
    
    coords_a = coords_a[['x', 'y', 'z']].values
    coords_t = coords_t[['x', 'y', 'z']].values
    
    clustering_a = AgglomerativeClustering(n_clusters=n_clusters_a).fit(coords_a)
    clustering_t = AgglomerativeClustering(n_clusters=n_clusters_t).fit(coords_t)
    labels_a = clustering_a.labels_
    labels_t = clustering_t.labels_
    label_dict = {keys_t[i] : l for i, l in enumerate(labels_t)}
    label_dict.update({keys_a[i] : l for i, l in enumerate(labels_a)})
    return(label_dict)

def cluster_chans_by_brain_area(fs, subject, exp, session):
    '''
    Clusters channels based on known brain area.
    ---Doesn't take into account second possible brain area---
    
    Parameters
    ------------------------
    fs : filesystem object
    subject : the subject
    exp : the experiment
    session : the session
    
    Returns
    -------------------
    label_dict : dictionary containing the channels and their labels
    '''
    with fs.open('l2l.jbferre.scratch/sub-'+subject+'_sess-'+session+'_channellocations.csv') as f:
        chans = pd.read_csv(f, names=['ch', 'area1', 'area2']).fillna('Unk')
    chans_t = chans[chans['ch'].isin([str(i) for i in range(len(chans))])]
    chans_a = chans[chans['ch'].isin([str(i)+'a' for i in range(len(chans))])]
    bad_chan = analysis.get_bad_channels(subject, exp, session)
    chans_a = chans_a[~chans_a['ch'].isin(bad_chan)]
    chans_t = chans_t[~chans_t['ch'].isin(bad_chan)]
    
    keys_a = chans_a.ch.values
    keys_t = chans_t.ch.values
    
    dict_a = {c:i for i, c in enumerate(np.unique(chans_a.area1.values))}
    dict_t = {c:i for i, c in enumerate(np.unique(chans_t.area1.values))}
    labels_a = np.array([dict_a[ch] for ch in chans_a.area1.values])
    labels_t = np.array([dict_t[ch] for ch in chans_t.area1.values])
    
    label_dict = {keys_t[i] : l for i, l in enumerate(labels_t)}
    label_dict.update({keys_a[i] : l for i, l in enumerate(labels_a)})
    return(label_dict)

def get_saccades(fs, subject, exp, session, num_std=1, smooth=10, threshold_dist=1, sac_type='end'):
    def _eye_renormalization(ex, ey, cross_time, sac_time):
        x_mean = np.mean(ex[cross_time])
        y_mean = np.mean(ey[cross_time])
        ex = ex - x_mean
        ey = ey - y_mean
        x_std = np.std(ex[sac_time])
        y_std = np.std(ex[sac_time])
        ex = ex / (3*x_std)
        ey = ey / (3*y_std)
        return(ex, ey)
    
    def _distance(x,y):
        x1 = x[:-1]
        x2 = x[1:]
        y1 = y[:-1]
        y2 = y[1:]
        dist = np.sqrt(np.power(x2-x1,2) + np.power(y2-y1,2))
        return(dist)
    
    ed, ex, ey = startup.get_eye_data(fs,subject,exp,session)
    df = startup.get_behavior(fs, subject, exp, session)
    ex, ey = _eye_renormalization(ex, ey, df[(df['act']=='cross_fix')].time.values, df[(df['act']=='obj_fix')|(df['act']=='obj_fix_break')].time.values)
    ex = analysis.moving_average_dim(ex, smooth, 0)
    ey = analysis.moving_average_dim(ey, smooth, 0)
    dist = _distance(ex, ey)
    t_adjust = round(smooth/2)
    
    idx_sac = np.argwhere(dist > num_std * np.std(dist))[:,0]
    idx_sep = np.insert(np.argwhere(idx_sac[1:]-idx_sac[:-1] > 5)[:,0]+1, 0, 0)
    sac_groups = []
    for i in range(len(idx_sep)-1):
        sac_groups.append(idx_sac[idx_sep[i]:idx_sep[i+1]])
    max_sac = []
    sac_dist = []
    for s in sac_groups:
        if(sac_type=='end'):
            max_sac.append(s[-1])
        elif(sac_type=='peak'):
            temp = np.argmax(dist[s])
            max_sac.append(s[temp])
        else:
            print('sac_type bad, arguments can be \'end\', \'peak\'')
        sac_dist.append(np.sqrt(np.power(ex[s[-1]] - ex[s[0]], 2) + np.power(ey[s[-1]] - ey[s[0]], 2)))
    max_sac = np.array(max_sac) + t_adjust
    sac_dist = np.array(sac_dist)
    return(max_sac[sac_dist>threshold_dist], sac_dist[sac_dist>threshold_dist])



