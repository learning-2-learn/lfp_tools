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


def get_exploration(ar, lag=1):
    '''
    Creates simple idea of exploration vs exploitation.
    Switches between exploring vs exploiting based on lag consecutive correct/incorrect trials
    
    Parameters
    -----------------
    ar : array of correct (200) or incorrect (206) responses
    lag : amount of consecutive trials distinct before switching between explore/exploit.
        Must be greater than 0
    
    Returns
    -----------------
    exploration : array of values that are 1 (exploring) or 0 (exploiting)
    '''
    assert lag>0
    
    if np.any(ar[:lag]==206):
        exploration = list(np.ones(lag, dtype=int))
    else:
        exploration = list(np.zeros(lag, dtype=int))
    
    for i in range(lag,len(ar)):
        if exploration[-1]==1:
            if np.all(ar[i-lag:i+1]==200):
                exploration.append(0)
            else:
                exploration.append(1)
        elif exploration[-1]==0:
            if np.all(ar[i-lag:i+1]==206):
                exploration.append(1)
            else:
                exploration.append(0)
    exploration = np.array(exploration)
    return exploration

def get_categories(sb, of):
    '''
    Gets the categories of states for all blocks
    Utilizes code from Vishwa : getTrialsCategory()
    Categories are:
        0 : Perseveration
        1 : Random Search
        2 : Rule Random Exploration
        3 : Rule Favored Exploration
        4 : Rule Preferred, No Exploration
        5 : Rule Persist, No Exploration
        
    Parameters
    ----------
    sb : state dataframe for all features
    of : object features dataframe
    
    Returns
    -------
    categories : array of length (num trials), indicating the category of a given trial
    '''
    def getTrialsCategory(state, prevRule, currRule):
        '''
        Code to get category of states (for a single block)
        Code from Vishwa

        Parameters
        ----------
        state : viterbi states (12 x num trials)
        prevRule : previous rule (int)
        currRule : current rule (int)

        Returns
        -------
        trialsCategory : categories (6 x num trials)
        '''
        T = state.shape[1]
        trialsCategory = np.zeros([6,T])

        # perseveration
        # previous rule in persist state
        prevF = state[prevRule,:] == 0
        for t in range(T):
            if prevF[t] == 1:
                trialsCategory[0,t] = 1
            elif prevF[t] == 0:
                break

        # perseveration length
        numP = int(np.sum(trialsCategory[0,:]))
        # if perseveration lasts the entire block
        if numP == T:
            return trialsCategory
        # if perseveration doesn't last the entire block
        else:
            for t in range(numP,T):
                # features in persist/preferred at trial t
                SRule = state[currRule,t]
                SNonRule = state[:,t]
                SNonRule = np.delete(SNonRule,currRule,axis=0)
                # random search
                if np.sum(SNonRule <= 1) == 0 and SRule > 1:
                    trialsCategory[1,t] = 1
                # rule random exploration
                elif np.sum(SNonRule <= 1) > 0 and SRule > 1:
                    trialsCategory[2,t] = 1
                # rule favored exploration
                elif np.sum(SNonRule <= 1) > 0 and SRule <= 1:
                    trialsCategory[3,t] = 1
                # rule preferred, no exploration
                elif np.sum(SNonRule <= 1) == 0 and SRule == 1:
                    trialsCategory[4,t] = 1
                # rule persist, no exploration
                elif np.sum(SNonRule <= 1) == 0 and SRule == 0:
                    trialsCategory[5,t] = 1
                else:
                    print('trial has no category')

        # check that every trial is labeled
        if np.array_equal(np.sum(trialsCategory,axis=0),np.ones([T])) is False:
            print('trialsCategory is not correct')

        return trialsCategory
    
    of_sub = of[of['TrialNumber'].isin(sb['trialIndex'])]
    blocks = np.unique(of_sub.BlockNumber.values)

    cat_all = []
    for b in blocks:
        trials_in_block = of_sub[of_sub['BlockNumber']==b].TrialNumber.values
        sb_sub = sb[sb['trialIndex'].isin(trials_in_block)]
        states = sb_sub[['viterbi_'+str(i) for i in range(12)]].values.T

        currRule = of_sub[of_sub['BlockNumber']==b].TrialType.values[0]
        assert np.all(currRule==sb_sub.rule.values)

        if b==0:
            prevRule = -1
        else:
            prevRule = of[of['BlockNumber']==b-1].TrialType.values[0]

        cat = getTrialsCategory(states, prevRule, currRule)
        cat_all.append(cat)

    cat_all = np.hstack(cat_all)
    
    assert(np.all(np.sum(cat_all==0, axis=0)==5)), "Duplicate categories"
    categories = np.argmax(cat_all, axis=0)
    
    return(categories)

import pickle
def get_vishwa_states(fs, subject, session, of):
    '''
    Function to get the attentional states for each feature from Vishwa's/Brian's model
    States are saved in l2l.jbferre.scratch currently
    Note, the featureChoiceLikelihood dictionary may be useful, but it is NOT retrieved here
    Takes data from 'superBlocksData', there is also 'BlocksData', 
        which should be the same thing but each rule block
        
    Parameters
    ----------
    fs : file system object
    subject : subject
    session : session. Note, not all sessions are found yet
    of : object feature dataframe
    
    Returns
    -------
    sb : dataframe with attentional states for each feature
    '''
    if subject=='SA':
        subject = 'sam'
        
    file = 'l2l.jbferre.scratch/012023_Vishwa_States/aligned_01_15_23/'+subject+'_aligned.pickle'
    objects = []
    with fs.open(file, 'rb') as f:
        while True:
            try:
                objects.append(pickle.load(f))
            except EOFError:
                break

    objects = objects[0][0]
    superBlocksData = objects['superBlocksData']
    
    idx = np.array(superBlocksData['session'])==session[2:]
    if ~np.any(idx):
        print('Session Not Computed, returning...')
        return None
    
    sb = pd.DataFrame()

    temp = np.hstack([superBlocksData['trialIndex'][i] for i in range(len(idx)) if idx[i]])
    sb['trialIndex'] = temp - 1

    temp = np.hstack([superBlocksData['rule'][i] for i in range(len(idx)) if idx[i]])
    sb['rule'] = temp

    temp = np.hstack(np.array(superBlocksData['chosenObject'], dtype=object)[idx])
    sb['chosenObject'] = temp

    temp = np.hstack([superBlocksData['viterbi'][i] for i in range(len(idx)) if idx[i]])
    for i in range(12):
        sb['viterbi_'+str(i)] = np.array(temp[i], dtype=int)

    # Checks to make sure it's aligned
    ic = np.array(of[of['TrialNumber'].isin(sb['trialIndex'].values)].ItemChosen.values, dtype=int)
    rule = of[of['TrialNumber'].isin(sb['trialIndex'].values)].TrialType.values

    assert np.all(sb['chosenObject'].values==ic), 'Item chosen is not aligned'
    assert np.all(sb['rule'].values==rule), 'Rule label is not aligned'
    
    categories = get_categories(sb, of)
    sb['category'] = categories
    
    return(sb)

import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
def plot_symbols(img, loc, zoom=1):
    '''
    Places a symbol at the desired location, usually in conjecture with axvline.
    
    Parameters
    ------------------
    img : string referring to type of symbol to place
    loc : tuple with x-y position on the plot
    zoom : size of the symbol
    
    Returns
    ------------------
    ab : annotationbox object, to be added to axis in main code.
    
    Examples
    ------------------
    fig, ax = plt.subplots()

    ax.plot(np.arange(10), np.arange(10))
    ax.axvline(4, color='k', linestyle='dashed')

    ab = development.plot_symbols('eye', (4,10))
    ax.add_artist(ab)
    '''
    if (img=='eye'):
        filename = 'eyeball.png'
        zoomMult = 0.09
    elif (img=='cross'):
        filename = 'cross.png'
        zoomMult = 0.03
    elif (img=='fb_cor'):
        filename = 'fb_cor.png'
        zoomMult = 0.03
    elif (img=='fb_inc'):
        filename = 'fb_inc.png'
        zoomMult = 0.03
    elif (img=='fb'):
        filename = 'fb.png'
        zoomMult = 0.03
    elif (img=='obj'):
        filename = 'obj.png'
        zoomMult = 0.03
    arr_lena = mpimg.imread(general.get_package_data(filename))
    imagebox = OffsetImage(arr_lena, zoom=zoom*zoomMult)
    ab = AnnotationBbox(imagebox, loc, annotation_clip=False, frameon=False)
    return(ab)

def get_ruby_model(fs):
    '''
    Gets ruby's model from S3
    
    Parameters
    ------------------------
    fs : filesystem object
    
    Returns
    -------------------
    rmodel : pandas dataframe with information from russels model
    '''
    with fs.open('l2l.jbferre.scratch/WCST_model_data.csv') as f:
        rmodel = pd.read_csv(f)
    return(rmodel)

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

def get_saccades(fs, species, subject, exp, session, num_std=1, smooth=10, threshold_dist=1, sac_type='end'):
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
    
    ed, ex, ey = startup.get_eye_data(fs,species,subject,exp,session)
    df = startup.get_behavior(fs, species, subject, exp, session)
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




################################## Old code for NS-DMD (maybe will use in the future)



def bop_cluster(params):
    '''
    bop_dmd, as described in Sashidhar 2021
    ---Probably doesn't work right now, for reference only---
    
    Parameters
    -----------------
    x : data matrix with shape (num recording locations, len(t))
    t : times corresponding to each snapshot
    r : rank of fit
    p : if float, percentage of snapshots to use in each opt-dmd run
    k : number of trials to run opt-dmd
    eigs_init : Initial guess of eigenvalues
    use_shervins_code : Flag to use my copied code or Shervin's code for opt-dmd. Need to run above cell if True
    
    Returns
    -----------------
    phi_m : mean matrix of DMD modes
    phi_s : std Matrix of DMD modes
    eval_m : mean array of eigenvalues
    eval_s : std array of eigenvalues
    b_m : mean weights
    b_s : std weights
    '''
    x,t,r,p,k,eigs_init = params
    eval0 = eigs_init
    
    if(type(p)!=int):
        p = int(p*x.shape[1])
    
    phis = []
    evals = []
    bs = []
    for i in range(k):
        idx = np.random.choice(np.arange(x.shape[1]), size=p, replace=False)
        
        x_temp = x[:,idx]
        t_temp = t[idx]
        optdmd = OptDMD(x_temp, t_temp, r)
        optdmd.fit(verbose=False, eigs_guess=eval0)
        evalk = optdmd.eigs
        phik = optdmd.modes
        bk = optdmd.amplitudes
        # phik, evalk, bk = opt_dmd(x_temp, t_temp, r, eigs_init=eval0)
        if np.any(np.abs(evalk.real)/2./np.pi > 200):
            continue
        temp = np.hstack([phik.reshape(-1),evalk,bk])
        if np.any(np.isnan(temp)):
            continue
        if np.any(np.isinf(temp)):
            continue
        phis.append(phik)
        evals.append(evalk)
        bs.append(bk)
    phi_m = np.mean(phis,axis=0)
    phi_s = np.std(phis,axis=0)
    evals_r = np.exp(np.array(evals).real)
    evals_i = np.array(evals).imag / 2. / np.pi
    eval_rm = np.mean(evals_r,axis=0)
    eval_im = np.mean(evals_i,axis=0)
    eval_rs = np.std(evals_r, axis=0)
    eval_is = np.std(evals_i, axis=0)
    b_m = np.mean(bs,axis=0)
    b_s = np.std(bs,axis=0)
    return(phi_m, phi_s, eval_rm, eval_im, eval_rs, eval_is, b_m, b_s)

def flatten(t):
    return [item for sublist in t for item in sublist]


####################################################################################

# Old and not used anymore


# from scipy.io import loadmat
# def get_electrode_xyz(fs, species, subject, exp, session, chans_spc=None):
#     '''
#     Clusters the channels based on their coordinates.
#     Currenty gathers from l2l.jbferre.scratch/20211013_xyz_coords
    
#     Parameters
#     -------------------
#     fs : filesystem object
#     species : the species
#     subject : the subject
#     exp : the experiment
#     session : the session to observe
#     chans_spc : specific channels to find xyz location
#         Use 'all' to get all channels, regardless if they've been determined as 'bad'
    
#     Returns
#     -------------------
#     cl : pandas dataframe giving the coordinates of each electrode
#     '''
#     with fs.open('l2l.jbferre.scratch/20211013_xyz_coords/epos_interaural_'+subject+session[2:]+'.mat') as f:
#         f_mat = loadmat(f)
#         keys = list(f_mat.keys())
#         datakeys = [i for i in keys if '__' not in i]
#         f_data = f_mat[datakeys[0]]
#         coords = pd.DataFrame(f_data, columns=['x', 'y', 'z']).fillna(0)
#     chan = np.hstack(([str(i) for i in range(1,125)], [str(i)+'a' for i in range(1,97)]))
#     coords['ch'] = chan
    
#     if (chans_spc == 'all'):
#         coords = coords
#     elif (chans_spc != None):
#         coords = coords[coords['ch'].isin(chans_spc)]
#     else:
#         bad_chan = analysis.get_bad_channels(species, subject, exp, session)
#         coords = coords[~coords['ch'].isin(bad_chan)]
    
#     return(coords)


# def get_brain_areas(fs, speciecs, subject, exp, session):
#     '''
#     Gets the brain areas of all channels
    
#     Parameters
#     ------------------------
#     fs : filesystem object
#     species : the species
#     subject : the subject
#     exp : the experiment
#     session : the session
    
#     Returns
#     -------------------
#     chans : pandas dataframe that includes channels and brain areas
#     '''
#     # with fs.open('l2l.jbferre.scratch/sub-'+subject+'_sess-'+session+'_channellocations.csv') as f:
#     #     chans = pd.read_csv(f, names=['ch', 'area1', 'area2'])
#     #     chans[chans['ch'].isin([str(i) for i in range(200)])] = chans[chans['ch'].isin([str(i) for i in range(200)])].fillna('UnkT')
#     #     chans[chans['ch'].isin([str(i)+'a' for i in range(200)])] = chans[chans['ch'].isin([str(i)+'a' for i in range(200)])].fillna('UnkP')
#     file = 'l2l.jbferre.scratch/sub-'+subject+'_RecLocTable/sub-'+subject+'_sess-'+session+'_RecLocTable.csv'
#     if not fs.exists(file):
#         print('File does not exist')
#     else:
#         with fs.open(file) as f:
#             chans = pd.read_csv(f)
#     return(chans)