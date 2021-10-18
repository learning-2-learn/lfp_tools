from lfp_tools import general
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
from matplotlib.widgets import Slider

def calculates_kmeans_error_curve(points, kmax, num_per_k=1):
    '''
    Finds the error curve for using kmeans
    From https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
    Modified to take average of a few realizations
    
    Parameters
    -----------------
    points : the points to calculate kmeans on
    kmax : the maximum value of k to choose for kmeans
    num_per_k : number of realizations to try for each value of k
    
    Returns
    -----------------
    sse : mean L2 error for each k
    '''
    sse = []
    for k in range(1, kmax+1):
        curr_sse_all = []
        for n in range(num_per_k):
            kmeans = KMeans(n_clusters = k).fit(points)
            centroids = kmeans.cluster_centers_
            pred_clusters = kmeans.predict(points)
            curr_sse = 0

            # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
            for i in range(len(points)):
                curr_center = centroids[pred_clusters[i]]
                curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
            
            curr_sse_all.append(curr_sse)

        sse.append(np.mean(curr_sse_all))
    return np.array(sse)

def hist_laxis(data, n_bins, range_limits, normalized=False):
    '''
    Calculates histograms over the last axis only.
    Code obtained through https://stackoverflow.com/questions/44152436/calculate-histograms-along-axis 
    
    Parameters
    -------------------
    data : n dimensional data, histogram is taken over last axis
    n_bins : number of bins to use in histogram
    range_limits : list or array of size 2. First element is lower range limit, second element is higher range limit
    normalized : truth value telling whether to normalize histogram or not. Defaults to False
    
    Returns
    -------------------
    bins : edge limits of the bins
    counts : histogram counts of data
    '''
    R = range_limits
    N = data.shape[-1]
    bins = np.linspace(R[0],R[1],n_bins+1)
    data2D = data.reshape(-1,N)
    idx = np.searchsorted(bins, data2D,'right')-1

    bad_mask = (idx==-1) | (idx==n_bins)
    scaled_idx = n_bins*np.arange(data2D.shape[0])[:,None] + idx

    limit = n_bins*data2D.shape[0]
    scaled_idx[bad_mask] = limit

    counts = np.bincount(scaled_idx.ravel(),minlength=limit+1)[:-1]
    counts.shape = data.shape[:-1] + (n_bins,)
    
    #My addition:
    if normalized:
        counts = counts / np.sum(counts, axis=-1)[:,None]
        
    return bins, counts

def coherence(sig1, sig2, sr=1000, nperseg=None, axis=-1):
    '''
    Calculates the coherence based on welches method.
    Uses scipy.signal.welch and scipy.signal.csd
    
    Parameters
    -----------------
    sig1 : first signal for coherence calculations
    sig2 : second signal for coherence calculations
    sr : sampling rate. Defaults to 1000
    nperseg : length of each windowed segment (see scipy.signal.welch). Defaults to None
    axis : axis of which to calculate the coherence over. Defaults to -1
    
    Returns
    -----------------
    f1 : frequencies where the coherence is calculated
    coh : the normalized coherence value
    phase : the phase values calculated for the coh.
            positive values mean that sig1 leads sig2
    '''
    f1, psd1 = ss.welch(sig1, fs=sr, nperseg=nperseg, axis=axis)
    f2, psd2 = ss.welch(sig2, fs=sr, nperseg=nperseg, axis=axis)
    fc, csd  = ss.csd(sig1, sig2, fs=sr, nperseg=nperseg, axis=axis)
    assert np.any(f1==f2) and np.any(f1==fc), 'Frequencies not the same for PSDS and CSD'
    
    phase = np.angle(csd)
    coh = np.abs(csd)**2 / (psd1 * psd2)
    
    return(f1, coh, phase)

def find_peaks(data, sr_new, avg_freq, sr_orig=1000, height=0.7, prominence=0.2):
    '''
    Finds the peaks of LFP data
    Assumes that the data is hilbert transformed, bandpassed data
    Assumes that for a filter on data with freq = avg(27, 37), the distance between peaks is minimum 80ms
    Estimates the filter size based on w1f1 = w2f2, 
        where w is the width (std) of the filter and f is the frequency
    It also includes a downsampling correction term (1/ds)
    
    Paramters
    -----------------
    data : the data to find the peaks for. The last dimension is the time dimension
    sr_new : sampling rate of input data
    avg_freq : the average frequency of the band of choice
    sr_orig : sampling rate of the original data
    
    Returns
    -----------------
    peak_times : array object matrix where the last dimension gives the peak times
    peak_amps : array object matrix where the last dimension gives the peak amplitudes
    '''
    def find_peaks_(ar, height, prominence, distance):
        temp = ss.find_peaks(ar, height=height, prominence=prominence, distance = distance)
        peak_loc = temp[0]
        peak_amp = temp[1]['peak_heights']
        return(peak_loc, peak_amp)
    
    d0 = 80
    dist = d0 * np.mean([27,37]) / avg_freq
    ds = sr_orig / sr_new
    dist_ds = dist / ds
    
    peaks = np.empty_like(data[...,0], dtype=object)
    general.func_over_last_dim(data, peaks, 1, find_peaks_, height=height, prominence=prominence, distance=dist_ds)
    
    return(peaks)

from scipy.ndimage import gaussian_filter1d
def filt_data_gauss(data, sr_new, avg_freq, sr_orig=1000.):
    '''
    Filters data with a Gaussian filter.
    Assumes that the data is hilbert transformed, bandpassed data
    Assumes that for a filter on data with freq = avg(27, 37), the width (std) = 25 ms
    Estimates the filter size based on w1f1 = w2f2, 
        where w is the width (std) of the filter and f is the frequency
    It also includes a downsampling correction term (1/ds)
    
    Parameters
    -----------------
    data : the data to filter
    sr_new : sampling rate of data (in ms)
    avg_freq : the average freq of the data
    sr_orig : the original sampling frequency of the data
    
    Returns
    -----------------
    data_filt : the filtered data
    '''
    f0 = 25
    std = f0 * np.mean([27,37]) / avg_freq
    ds = sr_orig / sr_new
    std_ds = std / ds
    
    data_filt = gaussian_filter1d(data, std_ds, axis=-1, mode='nearest')
    return(data_filt)

from mpl_toolkits.mplot3d import Axes3D
def plot_brain_coords(x,y,z,plot_type,plot_colors):
    '''
    Plots the xyz coordinates of all of the electrodes with a given color scheme
    
    Parameters
    --------------------
    x : array of x coordinates
    y : array of y coordinates
    z : array of z coordinates
    plot_type : type of brain plot to do. 'single' refers to a grayscale. 'multi' refers to multiple colors
    plot_colors : if plot_type=='single', this is the amplitude of the grayscale. if plot_type=='multi', this is a list of
        colors referring to the color of each electrode.
        
    Returns
    --------------------
    fig, ax : the figure and axis of the plotted object
    '''
    fig = plt.figure()
    ax = Axes3D(fig)
    
    if (plot_type=='single'):
        ax.scatter(x, y, z, c=plot_colors, cmap='gray_r', alpha=1)
    elif (plot_type=='multi'):
        for c in np.unique(plot_colors):
            idx = plot_colors==c
            ax.scatter(x[idx], y[idx], z[idx], color=c, alpha=1)
    return(fig, ax)

def get_sac_strategy_idx(sac_seq):
    '''
    Finds the trial indicies for specific strategy
    ---Continually updating strategies included---
    
    Strategies included:
        Ring Around the Rosie (R1 - R5 : number of saccades in ring)
        Boomerang (B3, B5 : ABA and ABCBA specifically)
        ABCA, etc (specific combinations)
        O5 : other unincluded combinations to get 5
        6 : any strategy that takes 6 or more saccades
        
    Parameters
    -------------------------
    saq_seq : list of arrays of saccade sequences
    
    Returns
    -------------------------
    strat_dict : dictionary of number of saccades for specific strategies listed above
    '''
    strat_dict = {'R'+str(i) : [] for i in range(1,6)}
    strat_dict.update({'B'+str(i) : [] for i in [3,5]})
    strat_dict['6'] = []
    strat_dict['ABCA'] = []
    strat_dict['BACA'] = []
    strat_dict['ABAC'] = []
    strat_dict['ABCAC'] = []
    strat_dict['O5'] = []
    
    for i, a in enumerate(sac_seq):
        if(len(a)==1):
            strat_dict['R1'].append(i)
        elif(len(a)==2 and len(np.unique(a))==len(a)):
            strat_dict['R2'].append(i)
        elif(len(a)==3 and len(np.unique(a))==len(a)):
            strat_dict['R3'].append(i)
        elif(len(a)==3 and len(np.unique(a))!=len(a)):
            strat_dict['B3'].append(i)
        elif(len(a)==4 and len(np.unique(a))==len(a)):
            strat_dict['R4'].append(i)
        elif(len(a)==4 and a[0]==a[3]):
            strat_dict['ABCA'].append(i)
        elif(len(a)==4 and a[1]==a[3]):
            strat_dict['BACA'].append(i)
        elif(len(a)==4 and a[0]==a[2]):
            strat_dict['ABAC'].append(i)
        elif(len(a)==5 and len(np.unique(a[:4]))==4):
            strat_dict['R5'].append(i)
        elif(len(a)==5 and a[0]==a[4] and a[1]==a[3]):
            strat_dict['B5'].append(i)
        elif(len(a)==5 and a[2]==a[4]):
            strat_dict['ABCAC'].append(i)
        elif(len(a)==5):
            strat_dict['O5'].append(i)
        elif(len(a)>=6):
            strat_dict['6'].append(i)
    return(strat_dict)

def get_saccade_seq(df, dtype='all', length=0):
    '''
    Finds the sequence of saccades for all trials
    
    Parameters
    ---------------------
    df : behavior dataframe
    dtype : all, 'last', 'first', 'middle'
    length : if dtype is anything but all, gives the number of trials to include
        last : last 'length'
        first : first 'length' starting at the second trial
        middle : all trials between 2+'length' and -'length'
    
    Returns
    ---------------------
    allSeq : list (len of trials) of arrays (len of number of saccades)
    '''
    allSeq = []
    
    if (dtype=='all'):
        trials = np.unique(df.trial.values)
    elif (dtype=='last'):
        trials = np.unique(df[df['trialRelB']>=-length].trial.values)
    elif (dtype=='first'):
        trials = np.unique(df[(df['trialRel']>=2) & (df['trialRel']<2+length)].trial.values)
    elif (dtype=='middle'):
        trials = np.unique(df[(df['trialRel']>=2+length) & (df['trialRelB']<-length)].trial.values)
    for t in trials:
        temp = df[(df['trial']==t) & (df['act'].isin(['obj_fix_break', 'obj_fix'])) & (df['ignore']==0) & (df['badGroup']==0)].encode.values
        allSeq.append(temp)
    return(allSeq)

def reorder_chans(chans):
    '''
    Takes a list of channels and reorders them by number and drive.
    Specifically, it returns the idx to reorder channels
    
    Parameters
    ----------
    chans : list or array of channel names. Names must be strings of integers or integers with 'a' on the end
    
    Returns
    -------
    idx : array of indicies that sort the channels
    '''
    chans = np.array(chans)
    chans_temp = np.sort([int(c) for c in chans if 'a' not in c])
    chans_post = np.sort([int(c.split('a')[0]) for c in chans if 'a' in c])
    chans_temp = [str(c) for c in chans_temp]
    chans_post = [str(c) + 'a' for c in chans_post]
    chans_sorted = chans_temp + chans_post
    idx = np.array([list(chans).index(c) for c in chans_sorted])
    return (idx)

def get_reordered_idx(df, i_type, params=[]):
    """
    Finds idx orderings and trial changes for ordering trials in unique way.
    \'rule\' organizes by rule, in order of appearance
    \'rule_basic\' just gives trial changes for rules (doesn't reorganize)
    \'feedback\' organizes by whether the subject was correct or not, in order of appearance
    \'feedback_rule' organizes by both rule and feedback, in order
    \'time_learned' organizes by what relative trial it is
    
    Parameters
    --------------
    
    df : dataframe describing behavior
    i_type : the control, specified above
    
    Returns
    --------------
    idx : indicies for reorganizing trials
    hlines : the trial number for segregating reorganization. E.g. separating each rule
    
    """
    idx = []
    hlines = []
    if (i_type == 'feedback'):
        for i in [200,206]:
            idx.append(np.argwhere(df[df['act']=='fb'].response.values==i)[:,0])
            hlines.append(len(np.hstack(idx)) - 0.5)
        idx = np.hstack(idx)
    elif (i_type == 'feedback_previous'):
        for i in [200,206]:
            idx.append((np.argwhere(df[df['act']=='fb'].response.values==i)[:,0]+1)[:-1])
            hlines.append(len(np.hstack(idx)) - 0.5)
        idx = np.hstack(idx)
    elif (i_type == 'rule'):
        for i in range(12):
            idx.append(np.argwhere(df[df['act']=='fb'].rule.values==i)[:,0])
            hlines.append(len(np.hstack(idx)) - 0.5)
        idx = np.hstack(idx)
    elif (i_type == 'rule_basic'):
        idx = np.arange(len(df[df['act']=='fb']))
        hlines = np.argwhere(df[(df['act']=='fb')].trialRel.values==0)[1:,0] - 0.5
    elif (i_type == 'feedback_rule'):
        for j in [200,206]:
            for i in range(12):
                idx.append(np.argwhere((df[df['act']=='fb'].rule.values==i) & (df[df['act']=='fb'].response.values==j))[:,0])
                hlines.append(len(np.hstack(idx)) - 0.5)
            hlines.append(hlines[-1])
        idx = np.hstack(idx)
    elif (i_type == 'time_learned'):
        if (params!=[]):
            numTrials = params[0]
        else:
            numTrials = np.max(df[df['act']=='fb'].trialRel.values)
        for i in range(numTrials):
            idx.append(np.argwhere(df[df['act']=='fb'].trialRel.values==i)[:,0])
            hlines.append(len(np.hstack(idx)) - 0.5)
        idx = np.hstack(idx)
    else:
        print('Type not found, please use one of the following:\n \
        \'rule\', \'rule_basic\', \'feedback\', \'feedback_previous\', \'feedback_rule\', \'time_learned\'')
        return([],[])
    return(idx, np.array(hlines)[:-1])

        
def plot_grid(function, plots, grid=(3,4), figsize=(14,7), sharex=True, sharey=True, titles=[], vlines=[], vline_colors=[], hlines=[], saveFig=None, **plt_kwargs):
    """
    Possibility of adding more plotting functions...
    
    Plots a grid of individual plots with specified function
    
    Parameters
    ----------
    function : can either be a string that specifies the function or a function that produces the desired plot
    plots : list of plots, must be of length greater than gridsize
    grid : (num_vertical, num_horizontal), number of plots in grid in each direction
    figsize : (size in x direction, size in y direction)
    sharex : boolean to tell whether to share x axis
    sharey : boolean to tell whether to share y axis
    titles : list of titles for plots in same order as plots
    vlines : list of vertical lines to include
    hlines : list of horizontal lines to include
    saveFig : string of filename to save figure as. If None, does not save
    **plt_kwargs : arguments to be passed into the plotting function
    
    Functions
    ---------
    'imshow' : plots with matplotlib.pyplot.imshow
        plots elements should be in the form of an input to matplotlib.pyplot.imshow
    '_mean_and_std' : plots mean and standard deviation with matplotlib.pyplot.plot
        plots elements should have a shape of (x, ((list of arrays), (list of arrays), ...up to four times))
    '_mean_and_sample' : plots mean and spaghetti plot with matplotlib.pyplot.plot
        plots elements should have a shape of (x, ((list of arrays), (list of arrays), ...up to four times))
    """
    
    def _imshow(pl, ax=None, **plt_kwargs):
        if ax is None:
            ax = plt.gca()
        ax.imshow(pl, **plt_kwargs)
        return(ax)
    
    def _mean_and_std(pl, ax=None, **plt_kwargs):
        if ax is None:
            ax = plt.gca()
        color_mean = ['blue', 'red', 'green', 'yellow']
        color_std = ['cornflowerblue', 'lightcoral', 'lawngreen', 'lightyellow']
        x = pl[0]
        y = pl[1]
        for k in range(len(y)):
            p_mean = np.mean(y[k], axis=0)
            p_std = np.std(y[k], axis=0)
            ax.plot(x, p_mean, color=color_mean[k])
            ax.fill_between(x, p_mean+p_std, p_mean-p_std, color=color_std[k], alpha=0.5, **plt_kwargs)
        return(ax)
    
    def _mean_and_sample(pl, ax=None, **plt_kwargs):
        if ax is None:
            ax = plt.gca()
        color_mean = ['blue', 'red', 'green', 'yellow']
        color_all = ['cornflowerblue', 'lightcoral', 'lawngreen', 'lightyellow']
        num_spag = 30
        x = pl[0]
        y = pl[1]
        for k in range(len(y)):
            for spag in y[k][np.random.choice(np.array(len(y[k])), num_spag, replace=False)]:
                ax.plot(x, spag, color=color_all[k], linewidth=2, alpha=0.4)
        for k in range(len(y)):
            p_mean = np.mean(y[k], axis=0)
            ax.plot(x, p_mean, color=color_mean[k])
        return(ax)
    
    maxi=grid[0]
    maxj=grid[1]
    
    fig, ax = plt.subplots(maxi,maxj,figsize=figsize,sharex=sharex, sharey=sharey)
    fig.tight_layout(pad=1)
    
    if (vline_colors==[]):
        for i in range(len(vlines)):
            vline_colors.append('k')
    
    for i in range(maxi):
        for j in range(maxj):
            if (function=='imshow'):
                ax[i,j] = _imshow(plots[j + i*maxj], ax[i,j], **plt_kwargs)
            elif (function=='mean_and_std'):
                ax[i,j] = _mean_and_std(plots[j + i*maxj], ax[i,j], **plt_kwargs)
            elif (function=='mean_and_sample'):
                ax[i,j] = _mean_and_sample(plots[j + i*maxj], ax[i,j], **plt_kwargs)
            elif (callable(function)):
                ax[i,j] = function(plots[j + i*maxj], ax[i,j], **plt_kwargs)
            else:
                print('Bad function variable')
                return
            
            if (titles!=[]):
                ax[i,j].set_title(titles[j + i*maxj])
            for line in hlines:
                ax[i,j].axhline(line, color='black', ls='-', lw=0.7)
            for line in range(len(vlines)):
                ax[i,j].axvline(vlines[line], color=vline_colors[line], ls='-', lw=0.7)
    
    if(saveFig!=None):
        plt.savefig(saveFig, bbox_inches = 'tight')
        

def plot_slider(sigs, vlines=[], markers=[], num_sigs=10, offset=['auto', 0], xrange=['auto', 0], colors=None):
    """
    Plots multiple signals with sliding window.
    Don't forget to use the command "%matplotlib notebook" first!
    Doesn't work in python lab. Switch to tree and try again
    
    Parameters
    ----------
    sigs : array (or list) of arrays to plot
    vlines : list of black, vertical lines to plot
    markers : list of specific vertical lines to plot per signal
        Should be of shape (len(sigs), any number, 4), where the last argument gives idx, color, marker, marker_size
    num_sigs : number of arrays to show in the screen at any one time
    offset : list of two objects. First (string) describes how to correct offset by second (number) object:
        'auto' automatically finds offset
        'rel' allows user to adjust automatic offset by multiplicative factor (second arg)
        'abs' allows user to set absolute offset (second arg)
    xrange : list of two objects. First (string) describes how to correct xrange by second (number) object:
        'auto' automatically finds xrange
        'rel' allows user to adjust automatic xrange by multiplicative factor (second arg)
        'abs' allows user to set absolute xrange (second arg)
    colors : list of colors of each plot. Should be longer than number of plots
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    minRange = 0
    if (xrange[0]=='auto'):
        maxRange = int(len(sigs[0]) / 100)
    elif (xrange[0]=='abs'):
        maxRange = xrange[1]
    elif (xrange[0]=='rel'):
        maxRange = int(len(sigs[0]) / 100) * xrange[1]
    else:
        print('Incorrect xrange args')
        maxRange = int(len(sigs[0]) / 100)
        
    if (offset[0]=='auto'):
        offset = 2 * np.std(sigs)
    elif (offset[0]=='rel'):
        offset = 2 * np.std(sigs) * offset[1]
    elif (offset[0]=='abs'):
        offset = offset[1]
    else:
        print('Incorrect offset args')
        offset = 2 * np.std(sigs)
        
    numPlots = len(sigs)

    plt.yticks([], [])

    for i in range(numPlots):
        if (colors==None):
            plt.plot(np.arange(len(sigs[i])), sigs[i] + offset*i)
        else:
            plt.plot(np.arange(len(sigs[i])), sigs[i] + offset*i, color=colors[i])
        if (markers != []):
            for m in markers[i]:
                plt.plot(m[0], offset*i, color=m[1], marker=m[2], markersize = m[3])
        
    for v in vlines:
        plt.axvline(v, color='k')

    plt.axis([minRange, maxRange, -offset, (num_sigs-1)*offset])

    aypos = plt.axes([0.01, 0.25, 0.03, 0.60])
    axpos = plt.axes([0.22, 0.1, 0.6, 0.04])

    yspos = Slider(aypos, 'Y', -offset, (numPlots-1)*offset, orientation='vertical')
    xspos = Slider(axpos, 'X', 0, len(sigs[0]) - maxRange, orientation='horizontal')

    def update(val):
        ypos = yspos.val
        xpos = xspos.val
        ax.axis([xpos,xpos + maxRange - minRange,ypos,ypos+num_sigs*offset])
        fig.canvas.draw_idle()

    yspos.on_changed(update)
    xspos.on_changed(update)

    plt.show()
    return(xspos, yspos)

def moving_average_dim(ar, size, dim):
    """
    Calculates the moving average along dimension
    
    Parameters
    --------------
    ar: array to be averaged
    size: size of window
    dim: dimension to calculate over
    
    Returns
    --------------
    Moving average along dim
    """
    br = np.apply_along_axis(_moving_average, dim, ar, size)
    return(br)

def moving_average_points(ar, size, points, ds):
    '''
    Finds where certain points end up after taking a moving average and downsampling
    Takes same parameters as analysis.moving_average_dim()
    points must be contained in ar
    
    Parameters
    ---------------
    ar : list or array containing points to find end location
    size : size of moving average window
    points : list or array of points to find location after averaging and downsampling
    ds : how much downsampling is done on ar
    
    Returns
    ---------------
    p_final : array of points and where they end up after doing averaging and downsampling
    '''
    ar = np.array(ar)
    ar_ds = moving_average_dim(ar, size, 0)[::ds]
    p_final = np.empty((len(points)))
    
    for i, p in enumerate(points):
        if (not np.any(p==ar)):
            print('One or more points in points not in ar')
            return([])
        p_idx = np.argwhere(p==ar)[:,0]
        points_idx = np.zeros((len(ar)), dtype=int)
        points_idx[p] = 1
        points_avg = moving_average_dim(points_idx, size, 0)
        points_ds = points_avg[::ds]
        p_final[i] = np.mean(ar_ds[np.argwhere(points_ds != 0)[:,0]])
    
    return(p_final)


def butter_pass_filter(data, cutoff, fs, btype, order=5):
    """ 
    Butter pass filters a signal with a butter filter
    
    Parameters
    ----------
    data: the signal to filter
    cutoff: the cutoff frequency
    fs: sampling rate
    btype: either \'high\' or \'low\', determines low pass or high pass filter
    order: the order of the filter
        
    Returns
    -------
    Either high or low pass filtered data
    """
    b, a = _butter_pass(cutoff, fs, btype, order=order)
    y = ss.filtfilt(b, a, data)
    return y

def get_psd(lfp, sr, params=[4096], method='welch'):
    """
    Finds the psd for the lfp data.
    Double check for correctness.
    
    Parameters
    ---------------
    lfp: signal to fft
    sr: sampling rate
    method: 'welch', 'fft', how the psd is calculated
    
    Returns
    ---------------
    freq: the frequencies associated with the power
    power: the power at each frequency
    """
    if(method=='welch'):
        freq, power = ss.welch(lfp, fs = sr, nperseg = params[0])
    elif(method=='fft'):
        power = np.abs(np.fft.fft(lfp))
        power = power[:int(len(power)/2)]
        freq = np.arange(len(power)) * sr / (2 * len(power))
    else:
        print('Wrong method, defaulting to \'welch\'')
        freq, power = get_psd(lfp, sr, params=[4096], method='welch')
    return(freq, power)

def time_Slicer(ar, timePoints, timeLength):
    """ 
    Selects desired epochs with desired lengths from general array
    
    Parameters
    ----------
    ar: array to be sliced
    timePoints: list of beginnings of epochs
    timeLength: length of epochs
        
    Returns
    -------
    ar[idx]: 2 dimensional array of shape (len(timPoints), timeLength).
        Includes time array for each epoch
    """
    idx = timePoints[:,np.newaxis] + np.arange(0,timeLength)
    return(ar[idx])

def beh_get_breaks(df, num_std=5):
    """
    Finds the trials where a break occurs.
    
    Parameters
    ---------------
    df: behavioral dataframe
    num_std: number of standard deviations above which is considered a break
    
    Returns
    ---------------
    c: an array of what trial a break occurs
    """
    a = df[df['encode']==150].time.values
    a2 = df[df['encode']==151].time.values
    b = a[1:] - a2[:-1]
    c = np.argwhere(np.mean(b) + num_std * np.std(b) < b)[:,0]
    if (c.size > 0):
        return (c + 0.5)
    else:
        return (c)
    
def get_chan_neighbors(subject, exp, chan):
    """
    Finds the nearest neighbor channels to a given channel.
    
    Parameters
    --------------
    subject: subject selected
    exp: experiment selected
    chan: input channel to find neighbors
    
    Returns
    --------------
    list of locations of nearest neighbors
    """
    channels = general.load_json_file('sub-'+subject+'_exp-'+exp+'_channels.json')
    if ('a' in chan):
        drive_chans = np.array(channels['drive_pfc'])
    else:
        drive_chans = np.array(channels['drive_temp'])
    if (chan not in np.hstack(drive_chans)):
        print('Channel name not in drive, check name.')
        return []
    elif (chan == '0'):
        print('Channel name needs to be positive')
        return []
    loc = np.argwhere(drive_chans == chan)
    ch_nn = [
        drive_chans[loc[0,0]+1, loc[0,1]],
        drive_chans[loc[0,0]-1, loc[0,1]],
        drive_chans[loc[0,0], loc[0,1]+1],
        drive_chans[loc[0,0], loc[0,1]-1]]
    return [ch for ch in ch_nn if ch != '0' and ch != '0a']

def get_bad_channels(subject, exp, session):
    """
    Finds and returns the bad channels of a given subject and session.
    
    Parameters
    ---------------
    subject: the subject's name
    exp: the experiment selected
    session: the session id
    
    Returns
    ---------------
    List of bad channels
    """
    bad_channels = general.load_json_file('sub-'+subject+'_exp-'+exp+'_bad_channels.json')
    all_sessions = list(bad_channels.keys())
    if (subject + session in all_sessions):
        return (bad_channels[subject + session])
    else:
        print('Either bad channels haven\'t been identified or incorrect subject/session')
        return ([])
    
    
    #Helper functions
    
def _butter_pass(cutoff, fs, btype, order=5):
    """ 
    Builds a butter pass filter
    
    Parameters
    ----------
    cutoff: the cutoff frequency
    fs: sampling rate
    btype: either \'high\' or \'low\', determines low pass or high pass filter
    order: the order of the filter
        
    Returns
    -------
    Either high or low pass filtered
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = ss.butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a

def _moving_average(a, n):
    """
    Calculates the moving average of an array.
    Function taken from Jaime here:
    https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
    
    Parameters
    --------------
    a: array to be averaged
    n: size of window
    
    Returns
    --------------
    Moving average
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n