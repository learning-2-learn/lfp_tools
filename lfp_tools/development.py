from lfp_tools import general
from lfp_tools import analysis
from lfp_tools import startup
import numpy as np
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


def get_saccades(fs, subject, exp, session, num_std=1, smooth=10, threshold_dist=1):
    #smooth must be small (no correction term for when max occurs)
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
    
    idx_sac = np.argwhere(dist > num_std * np.std(dist))[:,0]
    idx_sep = np.insert(np.argwhere(idx_sac[1:]-idx_sac[:-1] > 5)[:,0]+1, 0, 0)
    sac_groups = []
    for i in range(len(idx_sep)-1):
        sac_groups.append(idx_sac[idx_sep[i]:idx_sep[i+1]])
    max_sac = []
    sac_dist = []
    for s in sac_groups:
        temp = np.argmax(dist[s])
        max_sac.append(s[temp])
        sac_dist.append(np.sqrt(np.power(ex[s[-1]] - ex[s[0]], 2) + np.power(ey[s[-1]] - ey[s[0]], 2)))
    max_sac = np.array(max_sac)
    sac_dist = np.array(sac_dist)
    return(max_sac[sac_dist>threshold_dist], sac_dist[sac_dist>threshold_dist])



