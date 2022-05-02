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


from scipy.io import loadmat
def get_electrode_xyz(fs, subject, exp, session, chans_spc=None):
    '''
    Clusters the channels based on their coordinates.
    Currenty gathers from l2l.jbferre.scratch/20211013_xyz_coords
    
    Parameters
    -------------------
    fs : filesystem object
    subject : the subject
    exp : the experiment
    session : the session to observe
    chans_spc : specific channels to find xyz location
    
    Returns
    -------------------
    cl : pandas dataframe giving the coordinates of each electrode
    '''
    with fs.open('l2l.jbferre.scratch/20211013_xyz_coords/epos_interaural_'+subject+session[2:]+'.mat') as f:
        f_mat = loadmat(f)
        keys = list(f_mat.keys())
        datakeys = [i for i in keys if '__' not in i]
        f_data = f_mat[datakeys[0]]
        coords = pd.DataFrame(f_data, columns=['x', 'y', 'z']).fillna(0)
    chan = np.hstack(([str(i) for i in range(1,125)], [str(i)+'a' for i in range(1,97)]))
    coords['ch'] = chan
    
    bad_chan = analysis.get_bad_channels(subject, exp, session)
    coords = coords[~coords['ch'].isin(bad_chan)]
    
    if (chans_spc != None):
        coords = coords[coords['ch'].isin(chans_spc)]
    
    return(coords)

from scipy.io import loadmat
def get_brian_state_model(fs, subject, exp, session):
    '''
    Gets the states from Brians state model
    NOTE: exp isn't used here, yet...
    
    Parameters
    ----------
    fs : filesystem object
    subject : the subject
    exp : the experiment
    session : the session to observe
    
    Returns
    -------------------
    sb_all : pandas dataframe giving the state as described by Brians model
    '''
    file_sess = 'l2l.jbferre.scratch/brian_state_model/sam_files_session.mat'
    file_K_3 = 'l2l.jbferre.scratch/brian_state_model/sam_most_likely_K_3.mat'
    file_K_4 = 'l2l.jbferre.scratch/brian_state_model/sam_most_likely_K_4.mat'
    
    with fs.open(file_sess) as f:
        f_sess = loadmat(f)
    with fs.open(file_K_3) as f:
        f_3 = loadmat(f)
    with fs.open(file_K_4) as f:
        f_4 = loadmat(f)
    
    fileNames = f_sess['fileNames']
    
    idx_name = 'sub-'+subject+'_sess-'+session[2:]+'_parsedbehavior.csv'
    idx = np.argwhere(fileNames==idx_name)[:,0]
    
    fileNames = fileNames[idx]
    
    sessionIndex = f_sess['sessionIndex'][0][idx]
    K3_mostLikelyBlocks = f_3['mostLikelyBlocks'][0][idx]
    K3_ruleSuper = f_3['ruleSuper'][0][idx]
    K4_mostLikelyBlocks = f_4['mostLikelyBlocks'][0][idx]
    K4_ruleSuper = f_4['ruleSuper'][0][idx]
    
    sb_all = []
    for i in range(len(idx)):
        sb = pd.DataFrame()
        sb['trialNum'] = sessionIndex[i][0] - 1
        sb['K3_ruleSuper'] = np.insert(K3_ruleSuper[i][0], 0, -1)
        for j in range(12):
            sb['K3_rule'+str(j)] = np.insert(np.array(K3_mostLikelyBlocks[i][j], dtype=int), 0, -1)
        sb['K4_ruleSuper'] = np.insert(K4_ruleSuper[i][0], 0, -1)
        for j in range(12):
            sb['K4_rule'+str(j)] = np.insert(np.array(K4_mostLikelyBlocks[i][j], dtype=int), 0, -1)
        sb_all.append(sb)
    sb_all = pd.concat(sb_all, ignore_index=True)
    
    return(sb_all)

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
    # with fs.open('l2l.jbferre.scratch/sub-'+subject+'_sess-'+session+'_channellocations.csv') as f:
    #     chans = pd.read_csv(f, names=['ch', 'area1', 'area2'])
    #     chans[chans['ch'].isin([str(i) for i in range(200)])] = chans[chans['ch'].isin([str(i) for i in range(200)])].fillna('UnkT')
    #     chans[chans['ch'].isin([str(i)+'a' for i in range(200)])] = chans[chans['ch'].isin([str(i)+'a' for i in range(200)])].fillna('UnkP')
    file = 'l2l.jbferre.scratch/sub-'+subject+'_RecLocTable/sub-'+subject+'_sess-'+session+'_RecLocTable.csv'
    if not fs.exists(file):
        print('File does not exist')
    else:
        with fs.open(file) as f:
            chans = pd.read_csv(f)
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





#This section of code is taken from https://github.com/shervinsahba/dmdz and slightly tweaked to work with my code. In the future take directly from the source instead of making copy

def varpro2_expfun(alpha, t):
    A = t[:,np.newaxis] @ alpha[:,np.newaxis].T
    return np.exp(A)


def varpro2_dexpfun(alpha, t, j):
    # computes d/d(alpha_i) where we begin indexing at 0
    if (j < 0) or (j >= len(alpha)):
        raise ValueError("varpro2_dexpfun: cannot compute %sth derivative. Index j for d/d(alpha_j) out of range."%j)
    t = t.reshape((-1, 1))
    A = scipy.sparse.lil_matrix((t.size, alpha.size), dtype=complex)
    A[:, j] = t * np.exp(alpha[j] * t)
    return scipy.sparse.csc_matrix(A)


def varpro2_opts(set_options_dict=None):
    options = {
        "lambda0": 1.0,
        "max_lambda": 52,
        "lambda_up": 2.0,
        "lambda_down": 3.0,
        "use_marquardt_scaling": True,
        "max_iterations": 30,
        "tolerance": 1.0e-1,
        "eps_stall": 1.0e-12,
        "compute_full_jacobian": True,
        "verbose": True,
        "ptf": 1
    }
    optionsmin = {
        "lambda0": 0.0,
        "max_lambda": 0,
        "lambda_up": 1.0,
        "lambda_down": 1.0,
        "use_marquardt_scaling": False,
        "max_iterations": 0,
        "tolerance": 0.0,
        "eps_stall": -np.finfo(np.float64).min,
        "compute_full_jacobian": False,
        "verbose": False,
        "ptf": 0
    }
    optionsmax = {
        "lambda0": 1.0e16,
        "max_lambda": 200,
        "lambda_up": 1.0e16,
        "lambda_down": 1.0e16,
        "use_marquardt_scaling": True,
        "max_iterations": 1.0e12,
        "tolerance": 1.0e16,
        "eps_stall": 1.0,
        "compute_full_jacobian": True,
        "verbose": True,
        "ptf": 2147483647                 # sys.maxsize() for int datatype
    }
    if not set_options_dict:
        print("Default varpro2 options used.")
    else:
        for key in set_options_dict:
            if key in options:
                if optionsmin[key] <= set_options_dict[key] <= optionsmax[key]:
                    options[key] = set_options_dict[key]
                else:
                    warnings.warn("Value %s = %s is not in valid range (%s,%s)" %
                                  (key, set_options_dict[key], optionsmin[key], optionsmax[key]), Warning)
            else:
                warnings.warn("Key %s not in options" % key, Warning)
    return options


def varpro2(y, t, phi_function, dphi_function, alpha_init,
            linear_constraint=False,
            tikhonov_regularization=0,
            prox_operator=False,
            options=None):
    """
    :param y: data matrix
    :param t: vector of sample times
    :param phi: function phi(alpha,t) that takes matrices of size (m,n)
    :param dphi: function dphi(alpha,t,i) returning the d(phi)/d(alpha)
    :param alpha_init: initial guess for vector alpha
    :param use_tikhonov_regularization: Sets L2 regularization. Zero or False will have no L2 regularization.
    Can use a scalar (gamma) or matrix: min|y - phi*b|_F^2 + |gamma alpha|_2^2
    :param prox_operator: prox operator that can be applied to vector alpha at each step
    :param options: options for varpro2
    """

    def update_alpha(alpha, rjac, rhs, djacobian_pivot, prox_operator):
        # update eigenvalues
        delta0 = scipy.linalg.lstsq(rjac, rhs)[0]
        delta0 = delta0[djacobian_pivot]
        alpha0 = alpha + delta0
        if prox_operator:
            alpha0 = prox_operator(alpha0)
            delta0 = alpha0 - alpha
        return alpha0, delta0

    def varpro2_solve(phi, y, gamma, alpha):
        # least squares solution for mode amplitudes phi @ b = y, residual, and error
        b = scipy.linalg.lstsq(phi, y)[0]
        residual = y - phi@b
        if len(alpha) == 1 or np.isscalar(alpha):
            alpha = np.ravel(alpha).item()*np.eye(*gamma.shape)
        error_last = 0.5*(np.linalg.norm(residual, 'fro')**2 + np.linalg.norm(gamma@alpha)**2)
        return b, residual, error_last

    def varpro2_svd(phi, tolrank):
        # rank truncated svd where rank is scaled by a tolerance
        U, s, Vh = np.linalg.svd(phi, full_matrices=False)
        rank = np.sum(s > tolrank*s[0])
        U = U[:, :rank]
        s = s[:rank]
        V = Vh[:rank, :].conj().T
        return U, s, V

    t = np.ravel(t)
    n_data_cols = y.shape[1]
    n_t = len(t)
    n_alpha = len(alpha_init)

    options = varpro2_opts(set_options_dict=options)
    lambda0 = options['lambda0']

    if linear_constraint:
        # TODO linear constraints functionality
        raise Exception("linear constraint functionality not yet coded!")

    if tikhonov_regularization:
        if np.isscalar(tikhonov_regularization):
            gamma = tikhonov_regularization*np.eye(n_alpha)
    else:
        gamma = np.zeros((n_alpha, n_alpha))

    if prox_operator:
        alpha_init = prox_operator(alpha_init)

    # Initialize values
    alpha = np.copy(np.asarray(alpha_init, dtype=complex))
    alphas = np.zeros((n_alpha, options['max_iterations']), dtype=complex)
    if tikhonov_regularization:
        djacobian = np.zeros((n_t*n_data_cols + n_alpha, n_alpha), dtype=complex)
        rhs_temp = np.zeros(n_t*n_data_cols + n_alpha, dtype=complex)
        raise Exception("Tikhonov part not coded")
    else:
        djacobian = np.zeros((n_t*n_data_cols, n_alpha), dtype=complex)
        rhs_temp = np.zeros(n_t*n_data_cols, dtype=complex)
    error = np.zeros(options['max_iterations'])
    # res_scale = np.linalg.norm(y, 'fro')      # TODO res_scale unused in Askham's MATLAB code. Ditch it?
    scales = np.zeros(n_alpha)
    rjac = np.zeros((2*n_alpha, n_alpha), dtype=complex)

    phi = phi_function(alpha, t)
    tolrank = n_t*np.finfo(float).eps
    U, s, V = varpro2_svd(phi, tolrank)
    b, residual, error_last = varpro2_solve(phi, y, gamma, alpha)

    for iteration in range(options['max_iterations']):
        # build jacobian matrix by looping over alpha indices
        for j in range(n_alpha):
            dphi_temp = dphi_function(alpha, t, j)  # d/(dalpha_j) of phi. sparse output.
            sp_U = scipy.sparse.csc_matrix(U)
            djacobian_a = (dphi_temp - sp_U @ (sp_U.conj().T @ dphi_temp)).todense() @ b
            if options['compute_full_jacobian']:
                djacobian_b = U@scipy.linalg.lstsq(np.diag(s), V.conj().T @ dphi_temp.conj().T.todense() @ residual)[0]
                djacobian[:n_t*n_data_cols, j] = djacobian_a.ravel(order='F') + djacobian_b.ravel(order='F')
            else:
                djacobian[:n_t*n_data_cols, j] = djacobian_a.A.ravel(order='F')  # approximate Jacobian
            if options['use_marquardt_scaling']:
                scales[j] = min(np.linalg.norm(djacobian[:n_t*n_data_cols, j]), 1.0)
                scales[j] = max(scales[j], 1e-6)
            else:
                scales[j] = 1.0

        if tikhonov_regularization:
            print("using tikhonov regularization")
            djacobian[n_t*n_data_cols + 1:, :] = gamma

        # loop to determine lambda for the levenberg part
        # precompute components that don't depend on step-size parameter lambda
        # get pivots and lapack style qr for jacobian matrix
        rhs_temp[:n_t*n_data_cols] = residual.ravel(order='F')

        if tikhonov_regularization:
            rhs_temp[n_t*n_data_cols:] = -gamma@alpha

        g = djacobian.conj().T@rhs_temp

        djacobian_Q, djacobian_R, djacobian_pivot = scipy.linalg.qr(djacobian, mode='economic',
                                                                    pivoting=True)  # TODO do i need householder reflections?
        rjac[:n_alpha, :] = np.triu(djacobian_R[:n_alpha, :])
        rhs_top = djacobian_Q.conj().T@rhs_temp
        rhs = np.concatenate((rhs_top[:n_alpha], np.zeros(n_alpha)), axis=0)

        scales_pivot = scales[djacobian_pivot]
        rjac[n_alpha:2*n_alpha, :] = lambda0*np.diag(scales_pivot)

        alpha0, delta0 = update_alpha(alpha, rjac, rhs, djacobian_pivot, prox_operator)
        phi = phi_function(alpha0, t)
        b0, residual0, error0 = varpro2_solve(phi, y, gamma, alpha0)

        # update rule
        actual_improvement = error_last - error0
        predicted_improvement = np.real(0.5*delta0.conj().T@g)
        improvement_ratio = actual_improvement/predicted_improvement

        descent = " "  # marker that indicates in output whether the algorithm needed to enter the descent loop
        if error0 < error_last:
            # rescale lambda based on actual vs pred improvement
            lambda0 = lambda0*max(1/options['lambda_down'], 1 - (2*improvement_ratio - 1)**3)
            alpha, error_last, b, residual = (alpha0, error0, b0, residual0)
        else:
            # increase lambda until something works. kinda like gradient descent
            descent = "*"
            for j in range(options['max_lambda']):
                lambda0 = lambda0*options['lambda_up']
                rjac[n_alpha:2*n_alpha, :] = lambda0*np.diag(scales_pivot)

                alpha0, delta0 = update_alpha(alpha, rjac, rhs, djacobian_pivot, prox_operator)
                phi = phi_function(alpha0, t)
                b0, residual0, error0 = varpro2_solve(phi, y, gamma, alpha0)
                if error0 < error_last:
                    alpha, error_last, b, residual = (alpha0, error0, b0, residual0)
                    break

            if error0 > error_last:
                error[iteration] = error_last
                convergence_message = "Failed to find appropriate step length at iteration %d. Residual %s. Lambda %s"%(
                iteration, error_last, lambda0)
                if options['verbose']:
                    warnings.warn(convergence_message, Warning)
                return b, alpha, alphas, error, iteration, (False, convergence_message)

        # update and status print
        alphas[:, iteration] = alpha
        error[iteration] = error_last
        if options['verbose'] and (iteration%options['ptf'] == 0):
            print("step %02d%s error %.5e lambda %.5e"%(iteration, descent, error_last, lambda0))

        if error_last < options['tolerance']:
            convergence_message = "Tolerance %s met"%options['tolerance']
            return b, alpha, alphas, error, iteration, (True, convergence_message)

        if iteration > 0:
            if error[iteration - 1] - error[iteration] < options['eps_stall']*error[iteration - 1]:
                convergence_message = "Stall detected. Residual reduced by less than %s times previous residual."%(
                options['eps_stall'])
                if options['verbose']:
                    print(convergence_message)
                return b, alpha, alphas, error, iteration, (True, convergence_message)
            pass

        phi = phi_function(alpha, t)
        U, s, V = varpro2_svd(phi, tolrank)

    convergence_message = "Failed to reach tolerance %s after maximal %d iterations. Residual %s"%(
    options['tolerance'], iteration, error_last)
    if options['verbose']:
        warnings.warn(convergence_message, Warning)
    return b, alpha, alphas, error, iteration, (False, convergence_message)


class SVD(object):

    def __init__(self, svd_rank=0):
        self.X = None
        self.U = None
        self.s = None
        self.V = None
        self.svd_rank = 0

    @staticmethod
    def cumulative_energy(s, normalize=True):
        cumulative_energy = np.cumsum(s)
        if normalize:
            cumulative_energy = cumulative_energy/s.sum()
        return cumulative_energy

    @staticmethod
    def gavish_donoho_rank(X, s, energy_threshold=0.999999):
        """
        Returns matrix rank for Gavish-Donoho singular value thresholding.
        Reference: https://arxiv.org/pdf/1305.5870.pdf
        """
        beta = X.shape[0]/X.shape[1]
        omega = 0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43
        cutoff = np.searchsorted(SVD.cumulative_energy(s), energy_threshold)
        rank = np.sum(s > omega*np.median(s[:cutoff]))
        print("Gavish-Donoho rank is {}, computed on {} of {} "
              "singular values such that cumulative energy is {}.".format(rank, cutoff, len(s), energy_threshold))
        return rank

    @staticmethod
    def svd(X, svd_rank=0, full_matrices=False, verbose=False, **kwargs):
        """
        Computes the SVD of matrix X. Defaults to economic SVD.
        :param svd_rank: 0 for Gavish-Donoho threshold, -1 for no truncation, and
            integers [1,infinty) to attempt that truncation.
        :param full_matrices: False is the economic default.
        :return: U, s, V - note that this is V not Vh!
        See documentation for numpy.linalg.svd for more information.
        """
        U, s, V = np.linalg.svd(X, full_matrices=full_matrices, **kwargs)
        V = V.conj().T

        if svd_rank == 0:
            truncation_decision = "Gavish-Donoho"
            rank = SVD.gavish_donoho_rank(X, s)
        elif svd_rank >= 1:
            truncation_decision = "manual"
            if svd_rank < U.shape[1]:
                rank = svd_rank
            else:
                rank = U.shape[1]
                warnings.warn("svd_rank {} exceeds the {} columns of U. "
                              "Using latter value instead".format(svd_rank, U.shape[1]))
        elif svd_rank == -1:
            truncation_decision="no"
            rank = X.shape[1]

        if verbose:
            print("SVD performed with {} truncation, rank {}.".format(truncation_decision, rank))

        return U[:, :rank], s[:rank], V[:, :rank]

    def fit(self, full_matrices=False, **kwargs):
        if self.X is None:
            raise ValueError('SVD instance has no data X for SVD.X')
        else:
            self.U, self.s, self.V = self.svd(self.X, svd_rank=self.svd_rank,
                                              full_matrices=full_matrices, **kwargs)
        print("Computed SVD using svd_rank={}".format(self.svd_rank))
        
        
import scipy.linalg

class OptDMD(object):

    def __init__(self, X, timesteps, rank, optimized_b=False):
        self.svd_X = SVD.svd(X, -1, verbose=False)  # TODO check
        self.X = X
        self.timesteps = timesteps
        self.rank = rank  # rank of varpro2 fit, i.e. number of exponentials

        self.optimized_b = optimized_b

        self.eigs = None        # DMD continuous-time eigenvalues
        self.modes = None       # DMD eigenvectors
        self.amplitudes = None  # DMD mode amplitude vector

    @property
    def amplitudes_mod(self):
        return np.abs(self.amplitudes)

    @property
    def omega(self):
        """
        Returns the continuous-time DMD eigenvalues.
        """
        return self.eigs

    @property
    def temporaldynamics(self):
        """
        :return: matrix that contains temporal dynamics of each mode, stored by row
        """
        return np.exp(np.outer(self.omega, self.timesteps - self.timesteps[0])) * self.amplitudes[:, None]

    @property
    def reconstruction(self):
        """
        Reconstruction of data matrix X and the mean square error
        """
        reconstruction = (self.modes @ self.temporaldynamics)
        abs_error = np.abs(self.X - reconstruction)
        print("X_dmd MSE {}".format(np.mean(abs_error**2)))
        return reconstruction, abs_error

    @staticmethod
    def compute_amplitudes(X, modes, optimized_b):
        if optimized_b:
            # Jovanovic et al. 2014, Sparsity-promoting dynamic mode decomposition,
            # https://hal-polytechnique.archives-ouvertes.fr/hal-00995141/document
            # TODO. For now, it will return the non-optimized code.
            b = scipy.linalg.lstsq(modes, X.T[0])[0]
        else:
            b = scipy.linalg.lstsq(modes, X.T[0])[0]
        return b

    @staticmethod
    def optdmd(X, t, r, projected=True, eigs_guess=None, U=None, verbose=True):
        if projected:
            if U is None:
                U, _, _ = np.linalg.svd(X, full_matrices=False)
                U = U[:, :r]
                if verbose:
                    print('data projection: U_r\'X')
            else:
                if verbose:
                    print('data projection: U_provided\'X')
            varpro_X = (U.conj().T@X).T
        else:
            if verbose:
                print('data projection: none, X')
            varpro_X = X.T

        if eigs_guess is None:
            def generate_eigs_guess(U, X, t, r):
                UtX = U.conj().T@X
                UtX1 = UtX[:, :-1]
                UtX2 = UtX[:, 1:]

                dt = np.ravel(t)[1:] - np.ravel(t)[:-1]
                dX = (UtX2 - UtX1)/dt
                Xin = (UtX2 + UtX1)/2

                U1, s1, Vh1 = np.linalg.svd(Xin, full_matrices=False)
                U1 = U1[:, :r]
                V1 = Vh1.conj().T[:, :r]
                s1 = s1[:r]
                Atilde = U1.conj().T@dX@V1/s1

                eigs_guess = np.linalg.eig(Atilde)[0]
                return eigs_guess

            eigs_guess = generate_eigs_guess(U, X, t, r)
            if verbose:
                print("eigs_guess: generated eigs seed for varpro2.")
        else:
            if verbose:
                print("eigs_guess: user provided eigs seed for varpro2.")

        if verbose:
            options = {"verbose" : True}
        else:
            options = {"verbose" : False}
        modes, eigs, eig_array, error, iteration, convergence_status = varpro2(varpro_X, t, varpro2_expfun,
                                                                               varpro2_dexpfun, eigs_guess, options=options)
        modes = modes.T

        # normalize
        b = np.sqrt(np.sum(np.abs(modes)**2, axis=0)).T
        indices_small = np.abs(b) < 10*10e-16*max(b)
        b[indices_small] = 1.0
        modes = modes/b
        modes[:, indices_small] = 0.0
        b[indices_small] = 0.0

        if projected:
            modes = U @ modes

        return eigs, modes, b

    def fit(self, projected=True, eigs_guess=None, U=None, verbose=True):
        if verbose:
            print("Computing optDMD on X, shape {} by {}.".format(*self.X.shape))
        self.eigs, self.modes, self.amplitudes = OptDMD.optdmd(self.X, self.timesteps, self.rank,
                                                               projected=projected, eigs_guess=eigs_guess, U=U, verbose=verbose)
        return self

    def sort_by(self, mode="eigs"):
        """
        Sorts DMD analysis results for eigenvalues, eigenvectors (modes, Phi), and amplitudes_mod (b)
        in order of decreasing magnitude, either by "eigs" or "b".
        """
        if mode == "mod_eigs" or mode == "eigs":
            indices = np.abs(self.eigs).argsort()[::-1]
        elif mode == "amplitudes_mod" or mode == "b" or mode == "amps":
            indices = np.abs(self.amplitudes_mod).argsort()[::-1]
        else:
            mode = "default"
            indices = np.arange(len(self.eigs))
        self.eigs = self.eigs[indices]
        self.modes = self.modes[:, indices]
        self.amplitudes = self.amplitudes[indices]
        print("Sorted DMD analysis by {}.".format(mode))




##################################This ends the block taken from https://github.com/shervinsahba/dmdz



def bop_cluster(params):
    '''
    bop_dmd, as described in Sashidhar 2021
    
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


def get_dynamics(evals, b, timesteps):
    dynamics = np.exp(np.outer(evals, timesteps - timesteps[0])) * b[:, None]
    return dynamics

def reconstruct(evals, modes, b, timesteps):
    dynamics = get_dynamics(evals, b, timesteps)
    reconstruction = (modes @ dynamics)
    return reconstruction

def mean_2_error(data, rec):
    abs_error = np.abs(data - rec)
    mean_2_error = np.mean(abs_error**2)
    return(mean_2_error)

def get_phi_init(phi, freq, time):
    '''
    phi shape: (time, trial, chan, mode)
    freq shape: (time, trial, mode)
    time shape: (time)
    '''
    phi_angle = np.angle(phi)
    phase_diff = 2*np.pi*freq * time[:,None,None]
    phi_angle_init = (phi_angle - phase_diff[:,:,None,:])%(2*np.pi)
    phi_init = np.abs(phi)*np.exp(1j*phi_angle_init)
    return(phi_init)


#############Methods for finding G/F


def cos_dist(a,b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return(cos_sim)

def get_reconstruction(x_true, soln, G):
    x_rec = np.zeros((soln.shape[1], soln.shape[2]))
    for r in range(len(soln)):
        temp = soln[r,:,:] * G[r][None,:]
        x_rec = x_rec + temp

    mse = cos_dist(x_rec.real.reshape((-1)), x_true.reshape((-1)))
    return(mse)

def grad_loss(G, x, soln, alpha_G, beta, N):
    #G is current guess, x is true data, soln is solutions
    #alpha promotes sparsity
    #beta promotes smoothness over time
    Y = np.matmul(np.transpose(soln, [2,1,0]), np.transpose(G)[:,:,None])[:,:,0] #time, chan
    Y2 = Y - x.T
    l2_term_G = np.matmul(np.transpose(soln, [2,0,1]), Y2[:,:,None])[:,:,0].T
    
    alpha_term_G = np.ones((G.shape)) * alpha_G
    alpha_term_G[G<0] = -alpha_term_G[G<0]
    
    beta_term = np.zeros((G.shape))
    for i in range(1,N+1):
        beta_term[:,:-i] = beta_term[:,:-i] - beta*(G[:,i:]-G[:,:-i])
        beta_term[:,i:] =  beta_term[:,i:]  + beta*(G[:,i:]-G[:,:-i])
    
    dloss_G = l2_term_G + alpha_term_G + beta_term
    
    return(dloss_G)
        

def get_G_init(soln, x, beta, N):
    G_init = np.empty((soln.shape[0], soln.shape[-1]))
    for r in range(soln.shape[0]):
        for t in range(soln.shape[2]):
            top = 2 * (soln[r,:,t] @ x[:,t])
            bot = (2 * (soln[r,:,t] @ soln[r,:,t])) + 2*beta*N
            G_init[r,t] = top / bot
    return(G_init)

def get_G(params):
    x_true, freqs, phi, t_len_all, sr, offsets, alpha, beta, N, lr_G, maxiter, idx = params
    
    t = np.arange(t_len_all) / sr

    soln = []
    for r in range(0,freqs.shape[-1]):
        for i in range(len(freqs)):
            temp = np.exp(2*np.pi*1j*((t+offsets[i]) * freqs[i,r]))
            temp2 = phi[i,:,r][:,None]*temp
            soln.append(temp2)
    soln = np.array(soln).real
    soln = soln[idx]

    stopiter = np.mean(soln**2)**0.5
    G = get_G_init(soln, x_true, beta, N)
    G[G<0]=0
    
    for i in range(maxiter):
        dLdG = grad_loss(G, x_true, soln, alpha, beta, N)
        G = G - lr_G*dLdG
        G[G<0]=0
            
        if np.mean(np.abs(dLdG)) < stopiter:
            break
    return(soln, G)

def flatten(t):
    return [item for sublist in t for item in sublist]

def find_BG_exact(x, soln):
    top = np.sum(soln*x[None,:,:], axis=1)
    bot = np.sum(soln**2, axis=1)
    G = top / bot
    B = np.sum(soln[:,None] * soln[None,:], axis=2) / bot[:,None]
    return(B, G)

def get_F_from_BG(B, G, variance_thresh=0.01):
    F = np.empty(G.shape)
    for t in range(G.shape[1]):
        G_sub = G[:,t]
        B_sub = B[:,:,t].T
        u,s,vh = np.linalg.svd(B_sub)
        idx = s**2 / (s@s) > variance_thresh
        B_inv = vh.T[:,idx] @ np.diag(1./s[idx]) @ u.T[idx]
        F[:,t] = B_inv @ G_sub
    return(F)