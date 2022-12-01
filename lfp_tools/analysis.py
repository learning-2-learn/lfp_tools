from lfp_tools import general
from lfp_tools import startup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
from matplotlib.widgets import Slider
from sklearn.linear_model import LinearRegression

def explore_vs_exploit(ar, criterion=2, cor=200, inc=206):
    """
    Function for evaluating if an agent is in an explore or exploit state
    With this model, the agent changes between states upon criterion number of correct or incorrect
    E.g. the array 200, 200, 200, 206, 206, 200 (criterion=2) will be evaluated as
    exploit, exploit, exploit, exploit, explore, explore
    
    Parameters
    ----------
    ar : array with int elements equal to one of 2 values (corresponding to cor or inc parameters)
        array of responses to compute exploring vs exploiting
    criterion : int
        threshold for when to change state
    cor : int
        value of being correct (associated with exploiting)
    inc : int
        value of being incorrect (associated with exploration)
        
    Returns
    -------
    explore : array of ints (1 or 0)
        If 1, corresponds to being in an exploration state
        If 0, corresponds to being in an exploitation state
    """
    explore = np.empty(len(ar), dtype=int)
    if np.all(ar[:criterion]==cor):
        explore[:criterion] = 0
    else:
        explore[:criterion] = 1
    
    for i in range(criterion,len(ar)):
        if explore[i-1]==1:
            if np.all(ar[i-criterion+1:i+1]==200):
                explore[i] = 0
            else:
                explore[i] = 1
        elif explore[i-1]==0:
            if np.all(ar[i-criterion+1:i+1]==206):
                explore[i] = 1
            else:
                explore[i] = 0
    
    return(explore)



#############
#Decoders

from sklearn.svm import LinearSVC
class linear_decoder:
    """
    Class for decoding y based on x with sklearn.svm.LinearSVC
    
    Paramaters
    ----------
    num_samples : (int)
        number of times to rerun find_acc for statistics
    train_size : (float), between 0 and 1
        percentage of train/test split
    C : (float)
        regularization parameter in LinearSVC. Low values correspond to high regularization
    dual : (bool)
        Flag indicating solving the dual problem
        If the number of samples is greater than the number of parameters priortize dual=False
    fit_intercept : (bool)
        Flag indicating whether or not to fit the intercept with LinearSVC
    
    Methods
    -------
    find_C(x, y, min_C=-14, max_C=0):
        Method for finding the best regularization. 
        Tries every regularization from 10^min_C to 10^max_C in for every power of 10
        
        Parameters
        ----------
        x : float array of size (num data points, dimension of each data point)
            data matrix
        y : int array of size (num data points)
            corresponding element answer
        min_C : (int) less than max_C
            minimum regularization to try
        max_C : (int)
            maximum regularization to try
            
        Returns
        -------
        c_hyper : array of floats
            all values of regularization tried
        train_scores : array of floats with size (c_hyper)
            train accuracy for each value of C
        test_scores : array of floats with size (c_hyper)
            test accuracy for each value of C
    
    find_acc(x, y):
        Method for finding the accuracy of fitting y to x with a linear SVC
        Tries fitting self.num_samples number of times
        
        Parameters
        ----------
        x : float array of size (num data points, dimension of each data point)
            data matrix
        y : int array of size (num data points)
            corresponding element answer
            
        Returns
        -------
        train_scores : array of floats with size (self.num_samples)
            train accuracy for each run
        test_scores : array of floats with size (self.num_samples)
            test accuracy for each run
        chance_scores : array of floats with size (self.num_samples)
            chance accuracy for each run
        coefs : array of floats with size (self.num_samples, dimension of each data point)
            coefficients for each run
    """
    def __init__(self, num_samples=50, train_size=0.8, C=1*10**-7, dual=False, fit_intercept=False):
        """
        Initializes linear decoder class
        
        Paramaters
        ----------
        num_samples : (int)
            number of times to rerun find_acc for statistics
        train_size : (float), between 0 and 1
            percentage of train/test split
        C : (float)
            regularization parameter in LinearSVC. Low values correspond to high regularization
        dual : (bool)
            Flag indicating solving the dual problem
            If the number of samples is greater than the number of parameters priortize dual=False
        fit_intercept : (bool)
            Flag indicating whether or not to fit the intercept with LinearSVC
        """
        self.train_size = train_size
        self.C = C
        self.dual = dual
        self.fit_intercept = fit_intercept
        self.num_samples = num_samples
        
    def find_C(self, x, y, min_C=-14, max_C=0):
        """
        Method for finding the best regularization. 
        Tries every regularization from 10^min_C to 10^max_C in for every power of 10
        
        Parameters
        ----------
        x : float array of size (num data points, dimension of each data point)
            data matrix
        y : int array of size (num data points)
            corresponding element answer
        min_C : (int) less than max_C
            minimum regularization to try
        max_C : (int)
            maximum regularization to try
            
        Returns
        -------
        c_hyper : array of floats
            all values of regularization tried
        train_scores : array of floats with size (c_hyper)
            train accuracy for each value of C
        test_scores : array of floats with size (c_hyper)
            test accuracy for each value of C
        """
        c_hyper = np.array([1*10**i for i in range(min_C, max_C)])
        
        train_scores = np.empty((len(c_hyper)))
        test_scores = np.empty((len(c_hyper)))
        chance_scores = np.empty((len(c_hyper)))
        coefs = np.empty((len(c_hyper), x.shape[1]))
        
        for i,c in enumerate(c_hyper):
            self.C = c
            x_sub, y_sub = self._balance_data(x, y)
            x_train, x_test, y_train, y_test = self._train_test_split(x_sub, y_sub)
            x_train, x_test = self._norm_data(x_train, x_test)
            
            train_scores[i], test_scores[i], coefs[i] = self._fit_data(x_train, y_train, x_test, y_test)
            
        return(c_hyper, train_scores, test_scores)
    
    def find_acc(self, x, y):
        """
        Method for finding the accuracy of fitting y to x with a linear SVC
        Tries fitting self.num_samples number of times
        
        Parameters
        ----------
        x : float array of size (num data points, dimension of each data point)
            data matrix
        y : int array of size (num data points)
            corresponding element answer
            
        Returns
        -------
        train_scores : array of floats with size (self.num_samples)
            train accuracy for each run
        test_scores : array of floats with size (self.num_samples)
            test accuracy for each run
        chance_scores : array of floats with size (self.num_samples)
            chance accuracy for each run
        coefs : array of floats with size (self.num_samples, dimension of each data point)
            coefficients for each run
        """
        train_scores = np.empty((self.num_samples))
        test_scores = np.empty((self.num_samples))
        chance_scores = np.empty((self.num_samples))
        coefs = np.empty((self.num_samples, x.shape[1]))

        for i in range(self.num_samples):
            x_sub, y_sub = self._balance_data(x, y)
            x_train, x_test, y_train, y_test = self._train_test_split(x_sub, y_sub)
            x_train, x_test = self._norm_data(x_train, x_test)
            
            train_scores[i], test_scores[i], coefs[i] = self._fit_data(x_train, y_train, x_test, y_test)
            chance_scores[i] = self._fit_chance(x_train, y_train, x_test, y_test)

        return(train_scores, test_scores, chance_scores, coefs)
        
    def _balance_data(self, x, y):
        """
        Helper function for balancing the data
        Balances data by taking a random subselection of points such that
            there are equal numbers of points with unique values of y
        
        Parameters
        ----------
        x : float array of size (num data points, dimension of each data point)
            data matrix
        y : int array of size (num data points)
            corresponding element answer
        
        Returns
        -------
        x_sub : float array of size (balanced number of data points, dimension of each data point)
            data matrix
        y_sub : int array of size (balanced number of data points)
            corresponding element answer
        """
        y_vals = np.unique(y)
        max_num = np.min([np.sum(y==r) for r in y_vals])
        
        x_sub = np.empty((len(y_vals), max_num, x.shape[1]))
        y_sub = np.empty((len(y_vals), max_num), dtype=int)
        for i,r in enumerate(y_vals):
            idx = np.random.choice(np.arange(np.sum(y==r)), max_num, replace=False)
            x_sub[i] = x[y==r][idx]
            y_sub[i] = y[y==r][idx]
            
        x_sub = np.vstack(x_sub)
        y_sub = np.hstack(y_sub)
        return(x_sub, y_sub)
        
    def _train_test_split(self, x, y):
        """
        Helper function for getting the train test split
        
        Parameters
        ----------
        x : float array of size (num data points, dimension of each data point)
            data matrix
        y : int array of size (num data points)
            corresponding element answer
        
        Returns
        -------
        x_train : float array of size (num training data points, dimension of each data point)
            data matrix
        y_train : int array of size (num training data points)
            corresponding element answer
        x_test : float array of size (num testing data points, dimension of each data point)
            data matrix
        y_test : int array of size (num testing data points)
            corresponding element answer
        """
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        for val in np.unique(y):
            idx_train = np.random.choice(
                np.arange(np.sum(y==val)), 
                int(np.sum(y==val)*self.train_size),
                replace=False
            )
            idx_test = np.array([i for i in np.arange(np.sum(y==val)) if i not in idx_train])
            x_train.append(x[y==val][idx_train])
            x_test.append(x[y==val][idx_test])
            y_train.append(y[y==val][idx_train])
            y_test.append(y[y==val][idx_test])
        return(
            np.vstack(x_train),
            np.vstack(x_test),
            np.hstack(y_train),
            np.hstack(y_test)
        )
    
    def _norm_data(self, x_train, x_test):
        """
        Helper function for normalizing the data
        Normalizes by finding the mean and standard deviation of the training set
        Affects both the train and test set equally
        
        Parameters
        ----------
        x_train : float array of size (num data points, dimension of each data point)
            data matrix
        x_test : float array of size (num data points, dimension of each data point)
            data_matrix
        
        Returns
        -------
        x_train : float array of size (num data points, dimension of each data point)
            data matrix normalized
        x_test : float array of size (num data points, dimension of each data point)
            data_matrix normalized
        """
        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        x_train = (x_train - mean[None,:]) / std[None,:]
        x_test = (x_test - mean[None,:]) / std[None,:]
        return(x_train, x_test)
    
    def _fit_data(self, x_train, y_train, x_test, y_test):
        """
        Helper function for fitting the data
        
        Parameters
        ----------
        x_train : float array of size (num training data points, dimension of each data point)
            data matrix
        y_train : int array of size (num training data points)
            corresponding element answer
        x_test : float array of size (num testing data points, dimension of each data point)
            data matrix
        y_test : int array of size (num testing data points)
            corresponding element answer
            
        Returns
        -------
        train_scores : (float)
            fit score on training data
        test_scores : (float)
            fit score on testing data
        coefs : array of floats
            coefficients for LinearSVC
        """
        svc = LinearSVC(C=self.C, dual=self.dual, fit_intercept=self.fit_intercept)
        svc.fit(x_train, y_train)
        train_scores = svc.score(x_train, y_train)
        test_scores = svc.score(x_test, y_test)
        coefs = svc.coef_[0]
        return(train_scores, test_scores, coefs)
    
    def _fit_chance(self, x_train, y_train, x_test, y_test):
        """
        Helper function for finding the chance accuracies by scrambling to true answers
        
        Parameters
        ----------
        x_train : float array of size (num training data points, dimension of each data point)
            data matrix
        y_train : int array of size (num training data points)
            corresponding element answer
        x_test : float array of size (num testing data points, dimension of each data point)
            data matrix
        y_test : int array of size (num testing data points)
            corresponding element answer
            
        Returns
        -------
        chance_scores : (float)
            chance score on training data
        """
        svc = LinearSVC(C=self.C, dual=self.dual, fit_intercept=self.fit_intercept)
        svc.fit(x_train, np.random.choice(y_train, len(y_train), replace=False))
        chance_scores = svc.score(x_test, np.random.choice(y_test, len(y_test), replace=False))
        return(chance_scores)



#############
#Between ###, functions for finding probabilities of choosing features

def _get_trial_type(of):
    '''
    Gets the trial type for every trial in of. Trial type can be:
        start : starting trials in a rule block (before or including 2 incorrect trials)
        search : search period
        end : last 8/8 or 16/20 trials
        none : first block, last block, late, or no fixation trials
        
    Parameters
    ----------
    of : object feature dataframe
    
    Returns
    -------
    ts : array of strings indicating trial type
    '''
    def _get_bad_trials(of):
        idx = (of.Response.values=='Late') | \
              (of.Response.values=='NoFixation') | \
              (of.BlockNumber.values==0) | \
              (of.BlockNumber.values==np.max(of.BlockNumber.values))
        return(idx)
        
    def _get_start_trials(of, ts):
        res = of.Response.values
        tarc = of.TrialAfterRuleChange.values

        block = 'start'
        num_in_block = 0
        for i in range(len(res)):
            if ts[i]=='':
                if block=='search':
                    if tarc[i]==0:
                        block = 'start'
                        num_in_block = 0

                if block=='start':
                    if num_in_block<1:
                        ts[i] = block
                    else:
                        num_in_block = 0
                        ts[i] = block
                        block = 'search'

                    if res[i]=='Incorrect':
                        num_in_block += 1
        return(ts)
    
    def _get_end_trials(of, ts):
        res = of.Response.values
        block = 'end'
        num_in_block = 0
        thresh = 8 #last 8/8 or 16/20

        for i in range(len(res)-1,0-1,-1):
            if ts[i]=='start':
                block = 'end'
                num_in_block = 0
                thresh = 8

            if ts[i]=='':
                if block=='end':
                    if num_in_block<thresh-1:
                        ts[i] = block
                    else:
                        ts[i] = block
                        num_in_block = 0
                        block='search'

                    if res[i]=='Correct':
                        num_in_block += 1
                    elif res[i]=='Incorrect':
                        thresh = 16
        return(ts)
    
    ts = np.empty(len(of), dtype='<U20')

    idx = _get_bad_trials(of)
    ts[idx] = 'none'

    ts = _get_start_trials(of, ts)
    ts = _get_end_trials(of, ts)

    ts[ts==''] = 'search'

    return(ts)

def _get_seq_trials(of, seq, post_seq=1):
    '''
    Finds all of the trials that obey the given sequence

    Parameters
    ----------
    of : object feature dataframe
    seq : sequence of correct or incorrect responses
    post_seq : number of trials after sequence to include 
        (must be either Correct or Incorrect)

    Returns
    -------
    trials_sub : array of array of trial sequences that obey seq
    '''
    res = of.Response.values
    trials = of.TrialNumber.values
    assert len(res)==len(trials)

    res_temp = np.array([res[i:i+len(seq)+post_seq] for i in range(len(res)-len(seq)-post_seq)])
    trials_temp = np.array([trials[i:i+len(seq)+post_seq] for i in range(len(trials)-len(seq)-post_seq)])
    idx = np.all(res_temp[:,:len(seq)] == np.array(seq), axis=1) & \
          np.all((res_temp[:,len(seq):]=='Correct') | (res_temp[:,len(seq):]=='Incorrect'), axis=1)
    trials_sub = trials_temp[idx]

    return(trials_sub)

def _get_card_feat(of_sub, dtype='single'):
    '''
    Gets the card features of the choice at specific trials

    Parameters
    ----------
    of_sub : object feature dataframe preselected for desired trials
    dtype : either 'single' (gets individual features of chosen card)
        or 'all' (gets features of all cards)

    Returns
    -------
    c / colors : array of colors
    s / shapes : array of shapes
    p / patterns : array of patterns
    '''
    assert dtype=='single' or dtype=='all', 'wrong \'dtype\', either \'single\' or \'all\''

    patterns = of_sub[['Item'+str(i)+'Pattern' for i in range(4)]].values
    colors = of_sub[['Item'+str(i)+'Color' for i in range(4)]].values
    shapes = of_sub[['Item'+str(i)+'Shape' for i in range(4)]].values

    if dtype=='all':
        return(colors, shapes, patterns)
    else:
        choice = np.array(of_sub.ItemChosen.values, dtype=int)
        idx = (np.arange(len(choice)), choice)

        p = patterns[idx]
        c = colors[idx]
        s = shapes[idx]
        return(c, s, p)

def subselect_elements(a, b):
    '''
    Subselects elements from one array that exist in second array
    
    Parameters
    ----------
    a : numbers in first array
    b : numbers in second array
    
    Returns
    -------
    idx : array of bools saying if elements in array a are in array b
    '''
    idx = np.empty(len(a), dtype=bool)
    for i in range(len(a)):
        if np.any(a[i]==b):
            idx[i] = True
        else:
            idx[i] = False
    return(idx)

def prob_a_on_b(of, seq, a, b, trials_b=None):
    '''
    Calculates the probabilities that a feature on card a was chosen on card b
    
    Parameters
    ----------
    of : object feature dataframe
    seq : sequence of correct or incorrect responses
    a : first card idx
    b : second card idx
    trials_b : specific trials to include (aligns with when card b is chosen)
        
    Returns
    -------
    prob : probability of choosing a feature on card a on card b
    chance : chance of choosing a feature on card a on card b
    '''
    trials = _get_seq_trials(of, seq, post_seq=1+b-len(seq))
    if trials_b is not None:
        idx = subselect_elements(trials[:,b], trials_b)
        trials = trials[idx]
    
    a_c, a_s, a_p = _get_card_feat(of[of['TrialNumber'].isin(trials[:,a])])
    b_c, b_s, b_p = _get_card_feat(of[of['TrialNumber'].isin(trials[:,b])])
    b_c_all, b_s_all, b_p_all = _get_card_feat(of[of['TrialNumber'].isin(trials[:,b])], 'all')
    
    prob = (a_c==b_c) | (a_s==b_s) | (a_p==b_p)
    prob = np.sum(prob) / len(prob)
    
    chance = (a_c[:,None]==b_c_all) | (a_s[:,None]==b_s_all) | (a_p[:,None]==b_p_all)
    chance = np.mean(np.sum(chance, axis=1))/4
    
    return(prob, chance)


def prob_a_on_b_exclude_c(of, seq, a, b, c, trials_b=None):
    '''
    Calculates the probabilities that a feature on card a was chosen on card b, excluding card c
    Also, we enforce that at least one of the features on card a was chosen on card c
    
    Parameters
    ----------
    of : object feature dataframe
    seq : sequence of correct or incorrect responses
    a : first card idx
    b : second card idx
    c : excluded card idx (must be between a and b)
    trials_b : specific trials to include (aligns with when card b is chosen)
        
    Returns
    -------
    prob : probability of choosing a feature on card a on card b
    chance : chance of choosing a feature on card a on card b
    '''
    assert c!=a
    assert c!=b
    
    trials = _get_seq_trials(of, seq, post_seq=1+b-len(seq))
    if trials_b is not None:
        idx = subselect_elements(trials[:,b], trials_b)
        trials = trials[idx]
    
    a_c, a_s, a_p = _get_card_feat(of[of['TrialNumber'].isin(trials[:,a])])
    b_c, b_s, b_p = _get_card_feat(of[of['TrialNumber'].isin(trials[:,b])])
    c_c, c_s, c_p = _get_card_feat(of[of['TrialNumber'].isin(trials[:,c])])
    b_c_all, b_s_all, b_p_all = _get_card_feat(of[of['TrialNumber'].isin(trials[:,b])], 'all')
    
    idx = (a_c==c_c) | (a_s==c_s) | (a_p==c_p)
    
    prob = ((a_c==b_c) & (a_c!=c_c)) | \
           ((a_s==b_s) & (a_s!=c_s)) | \
           ((a_p==b_p) & (a_p!=c_p))
    prob = np.sum(prob[idx]) / len(prob[idx])
    
    chance = ((a_c[:,None]==b_c_all) & (a_c!=c_c)[:,None]) | \
             ((a_s[:,None]==b_s_all) & (a_s!=c_s)[:,None]) | \
             ((a_p[:,None]==b_p_all) & (a_p!=c_p)[:,None])
    chance = np.mean(np.sum(chance[idx], axis=1))/4
    
    return(prob, chance)


def prob_a_on_b_include_c(of, seq, a, b, c, trials_b=None):
    '''
    Calculates the probability that a feature on card a was chosen on card b, 
        if that feature was the only one chosen (from a) on card c
    Also, we enforce that at least one of the features on card a was chosen on card b
    
    Parameters
    ----------
    of : object feature dataframe
    seq : sequence of correct or incorrect responses
    a : first card idx
    b : second card idx
    c : included card idx (must be between a and b)
    trials_b : specific trials to include (aligns with when card b is chosen)
        
    Returns
    -------
    prob : probability of choosing a feature on card a on card b
    chance : chance of choosing a feature on card a on card b
    '''
    assert c!=a
    assert c!=b
    
    trials = _get_seq_trials(of, seq, post_seq=1+b-len(seq))
    if trials_b is not None:
        idx = subselect_elements(trials[:,b], trials_b)
        trials = trials[idx]
    
    a_c, a_s, a_p = _get_card_feat(of[of['TrialNumber'].isin(trials[:,a])])
    b_c, b_s, b_p = _get_card_feat(of[of['TrialNumber'].isin(trials[:,b])])
    c_c, c_s, c_p = _get_card_feat(of[of['TrialNumber'].isin(trials[:,c])])
    b_c_all, b_s_all, b_p_all = _get_card_feat(of[of['TrialNumber'].isin(trials[:,b])], 'all')
    
    idx1 = (a_c==b_c) | (a_s==b_s) | (a_p==b_p)
    idx2 = np.sum([a_c==c_c, a_s==c_s, a_p==c_p], axis=0)==1
    idx = idx1 & idx2
    
    prob = ((a_c==b_c) & (a_c==c_c)) | \
           ((a_s==b_s) & (a_s==c_s)) | \
           ((a_p==b_p) & (a_p==c_p))
    prob = np.sum(prob[idx]) / len(prob[idx])
    
    
    cond1 = np.array([(a_c[:,None]==b_c_all), (a_s[:,None]==b_s_all), (a_p[:,None]==b_p_all)])[:,idx]
    cond2 = np.array([(a_c==c_c), (a_s==c_s), (a_p==c_p)])[:,idx]
    chance = np.sum(
        np.sum(
            np.sum(
                cond2[:,:,None]*cond1, 
                axis=0
            )[None,:,:] * cond1, 
            axis=0
        ), 
        axis=1
    )
    chance = np.mean(chance) / 3
    
    return(prob, chance)

#############

def get_monkey_choices(of, min_block=2, add_prev_trial=False):
    '''
    Helper function to find what the monkey chose and what cards were available in the WCST dataset
    Removes trials where the monkey was NOT correct or incorrect
    
    Parameters
    ------------------
    of : object feature dataframe
    min_block : minimum block number to consider. Set to 0 to start at beginning
    add_prev_trial : flag indicating whether or not to include a trial from the previous block to help with testing
    
    Returns
    ------------------
    item_chosen : index of card chosen
    res : response (correct or incorrect) of trial
    cards : cards for each trial, where integers are encoded to features.
    '''
    idx = of[
        (of['BlockNumber']>=min_block) & 
        (of['BlockNumber']<np.max(of.BlockNumber.values)) & 
        (of['Response'].isin(['Correct', 'Incorrect']))
    ].index.values
    if add_prev_trial:
        idx = np.insert(idx, 0, idx[0]-1)
    of_sub = of[of.index.isin(idx)]
    
    FEATURE_NAMES = np.array([
        'CIRCLE', 'SQUARE', 'STAR', 'TRIANGLE', 
        'CYAN', 'GREEN', 'MAGENTA', 'YELLOW', 
        'ESCHER', 'POLKADOT', 'RIPPLE', 'SWIRL'
    ])

    feature_dict = {f:i for i,f in enumerate(FEATURE_NAMES)}
    
    item_chosen = np.array(of_sub.ItemChosen.values, dtype=int)
    res = of_sub.Response.values
    rules = np.array([feature_dict[r] for r in of_sub.CurrentRule.values])
    cards = np.empty((len(of_sub), 4, 3), int)
    for i in range(4):
        for j, dim in enumerate(['Shape', 'Color', 'Pattern']):
            temp = of_sub['Item'+str(i)+dim].values
            cards[:,i,j] = np.array([feature_dict[f] for f in temp])
            
    return(item_chosen, res, cards, rules)

#below, all for saccade related things
def sac_get_stereotypical_response(df, sac, dir_l=-110, dir_h=-70, sac_delay_t=50, obj_delay_t=50):
    '''
    Finds the stereotypical responses based on the desired parameters
    
    Parameters
    ----------------
    df : behavioral dataframe
    sac : saccade dataframe
    dir_l : lower bound on direction
    dir_h : upper bound on direction
    sac_delay_t : time delay between end of one saccade and start of next saccade
    obj_delay_t : time delay between start of all saccades after objects turn on
    
    Returns
    ----------------
    sac : saccade dataframe with extra column of stereotypical responses
    '''
    trials = np.unique(df[df['response'].isin([200,206])]['trial'].values)
    sac = sac.copy()
    sac['stereotypical'] = 0
    idx_val = []
    for i,t in enumerate(trials):
        zone = sac[sac['trial']==t].interval.values
        direction = sac[sac['trial']==t].direction.values
        sac_start_t = sac[sac['trial']==t].time_start.values
        sac_end_t = sac[sac['trial']==t].time_end.values
        obj_start_t = df[(df['trial']==t) & (df['act']=='obj_on')].time.values[0] + 63
        sac_idx = sac[sac['trial']==t].index.values

        if np.any('prep'==zone):
            direction = direction[~(zone=='prep')]
            sac_start_t = sac_start_t[~(zone=='prep')]
            sac_end_t = sac_end_t[~(zone=='prep')]
            sac_idx = sac_idx[~(zone=='prep')]
            zone = zone[~(zone=='prep')]

        temp = (direction>dir_l) & \
               (direction<dir_h) & \
               np.insert((sac_start_t[1:] - sac_end_t[:-1] < sac_delay_t), 0, True) & \
               (sac_start_t - obj_start_t < obj_delay_t)
        # print(t, len(sac_idx), np.sum(temp==True), np.argwhere(temp==False))
        # if len(sac_idx)>0:
        if np.sum(temp==False)>0:
            idx_val.append(sac_idx[:np.argwhere(temp==False)[0,0]])
    sac.loc[np.hstack(idx_val), 'stereotypical'] = 1
    return(sac)

def create_saccade_dataframe(fs, species, subject, exp, session, num_std=0.2, dir_l=-110, dir_h=-70, sac_delay_t=50, obj_delay_t=50, troubleshoot=False):
    '''
    Creates saccade dataframe from scratch.
    This may take a few moments, mainly to load in data
    
    Parameters
    ------------------
    fs : s3 file system object
    species : species
    subject : the subject
    exp : the experiment
    session : the session identifier
    num_std : for saccade detection, includes saccades above this threshold
    dir_l : for stereotypical saccade detection, lower bound on direction
    dir_h : for stereotypical saccade detection, upper bound on direction
    sac_delay_t : for stereotypical saccade detection, time delay between end of one saccade and start of next saccade
    obj_delay_t : for stereotypical saccade detection, time delay between start of all saccades after objects turn on
    troubleshoot : flag for returning extra variables of use
    
    Returns
    ------------------
    sac : saccade dataframe
    
    if troubleshoot
    Returns
    ------------------
    sac : saccade dataframe
    ex : x eye position
    ex : y eye position
    '''
    def _sac_get_trials(df, times):
        trial_starts = df[df['act']=='cross_on'].time.values
        last_response = df[df['encode'].isin([200,202,204,206])].encode.values[-1]
        if last_response==206:
            last_time = df[df['encode'].isin([200,202,204,206])].time.values[-1]+5500
        else:
            last_time = df[df['encode'].isin([200,202,204,206])].time.values[-1]+1900
        trial_ends = np.insert(
            trial_starts[1:], 
            len(trial_starts)-1, 
            last_time
        )

        trials = df[df['act']=='cross_on'].trial.values
        res = df[df['act']=='cross_on'].response.values
        break_trials = beh_get_breaks(df)
        break_trials = np.array(
            np.insert(
                break_trials,
                len(break_trials), 
                df.trial.values[-1]+0.5
            ) - 0.5,
            dtype=int
        )

        idx = np.array([np.argwhere(b==trials)[:,0] for b in break_trials])[:,0]
        trial_ends[idx[res[idx]==200]] = df[(df['trial'].isin(idx[res[idx]==200])) & \
                                            (df['encode']==200)].time.values+1900
        for i in [202,204,206]:
            trial_ends[idx[res[idx]==i]] = df[(df['trial'].isin(idx[res[idx]==i])) & \
                                                (df['encode']==i)].time.values+5500

        sac_trials = np.zeros(len(times),dtype=int)-1
        for i,t in enumerate(trials):
            sac_trials[(times>=trial_starts[i]) & (times<trial_ends[i])] = t

        return(sac_trials)
    
    def _sac_zone(df, sac):
        # part where no task is being performed
        zone = np.array(['-' for i in range(len(sac))], dtype='<U19')
        zone[sac['trial'].values==-1] = 'no_task'

        # between cross turning on and through cross fixation
        t_starts = df[(df['act']=='cross_on') & (df['response'].isin([200,202,206]))].time.values
        t_ends = df[(df['act']=='cross_fix') & (df['response'].isin([200,202,206]))].time.values+350

        region = np.empty((len(t_starts), len(sac)), dtype=bool)
        for i in range(len(t_starts)):
            region[i] = (sac['time_start'].values > t_starts[i]) & \
                        (sac['time_start'].values <= t_ends[i]) & \
                        (zone=='-')
        idx = np.sum(region, axis=0)==1
        zone[idx] = 'prep'

        # end of required cross fixation and objects turning on
        t_starts = df[(df['act']=='cross_fix') & (df['response'].isin([200,202,206]))].time.values+350
        t_ends = df[(df['encode']==29) & (df['response'].isin([200,202,206]))].time.values+63

        region = np.empty((len(t_starts), len(sac)), dtype=bool)
        for i in range(len(t_starts)):
            region[i] = (sac['time_start'].values > t_starts[i]) & \
                        (sac['time_start'].values <= t_ends[i]) & \
                        (zone=='-')
        idx = np.sum(region, axis=0)==1
        zone[idx] = 'pre_obj'

        # objects turning on to object fixation
        t_starts = df[(df['encode']==29) & (df['response'].isin([200,206]))].time.values+63
        t_ends = df[(df['act']=='fb') & (df['response'].isin([200,206]))].time.values - 800

        region = np.empty((len(t_starts), len(sac)), dtype=bool)
        for i in range(len(t_starts)):
            region[i] = (sac['time_start'].values > t_starts[i]) & \
                        (sac['time_start'].values <= t_ends[i]) & \
                        (zone=='-')
        idx = np.sum(region, axis=0)==1
        zone[idx] = 'obj'

        # object fixation to fb
        t_starts = df[(df['act']=='fb') & (df['response'].isin([200,206]))].time.values - 800
        t_ends = df[(df['act']=='fb') & (df['response'].isin([200,206]))].time.values

        region = np.empty((len(t_starts), len(sac)), dtype=bool)
        for i in range(len(t_starts)):
            region[i] = (sac['time_start'].values > t_starts[i]) & \
                        (sac['time_start'].values <= t_ends[i]) & \
                        (zone=='-')
        idx = np.sum(region, axis=0)==1
        zone[idx] = 'obj_fix'

        # feedback to next cross on
        t_starts = df[(df['act']=='fb') & (df['response'].isin([200]))].time.values
        t_ends = t_starts + 2100

        region = np.empty((len(t_starts), len(sac)), dtype=bool)
        for i in range(len(t_starts)):
            region[i] = (sac['time_start'].values > t_starts[i]) & \
                        (sac['time_start'].values <= t_ends[i]) & \
                        (zone=='-')
        idx = np.sum(region, axis=0)==1
        zone[idx] = 'fb'

        t_starts = df[(df['act']=='fb') & (df['response'].isin([206]))].time.values
        t_ends = t_starts + 5700

        region = np.empty((len(t_starts), len(sac)), dtype=bool)
        for i in range(len(t_starts)):
            region[i] = (sac['time_start'].values > t_starts[i]) & \
                        (sac['time_start'].values <= t_ends[i]) & \
                        (zone=='-')
        idx = np.sum(region, axis=0)==1
        zone[idx] = 'fb'

        # no cross fixation
        t_starts = df[(df['act']=='cross_on') & (df['response'].isin([204]))].time.values
        t_ends = df[(df['encode']==204) & (df['response'].isin([204]))].time.values + 6000

        region = np.empty((len(t_starts), len(sac)), dtype=bool)
        for i in range(len(t_starts)):
            region[i] = (sac['time_start'].values > t_starts[i]) & \
                        (sac['time_start'].values <= t_ends[i]) & \
                        (zone=='-')
        idx = np.sum(region, axis=0)==1
        zone[idx] = 'no_cross_fix'

        # late response
        t_starts = df[(df['act']=='cross_on') & (df['response'].isin([202]))].time.values
        t_ends = df[(df['encode']==202) & (df['response'].isin([202]))].time.values + 6000

        region = np.empty((len(t_starts), len(sac)), dtype=bool)
        for i in range(len(t_starts)):
            region[i] = (sac['time_start'].values > t_starts[i]) & \
                        (sac['time_start'].values <= t_ends[i]) & \
                        (zone=='-')
        idx = np.sum(region, axis=0)==1
        zone[idx] = 'late_response'

        zone[zone=='-'] = 'unsure'

        return(zone)
    
    def _is_left(x1, x2, y1, y2, px, py):
        '''
        left is defined as left of line when going from x1,y1 to x2,y2
        '''
        val = ((x2 - x1)*(py - y1) - (y2 - y1)*(px - x1)) > 0
        return(val)

    def _check_what_trial(times, t_starts, t_ends):
        idx = np.zeros(len(times), dtype=int) - 1
        for i in range(len(t_starts)):
            temp = np.argwhere((times>=t_starts[i]) & (times<t_ends[i]))[:,0]
            idx[temp] = i
        return idx

    def _combine_objs(ar):
        '''
        Combines array elements so that every kind of saccade is represented

        Parameters
        ----------------
        ar : 2d array that's 4 (4 objects) by the number of saccades

        Returns
        ----------------
        loc : 1d array of length number of saccades
        '''
        loc = np.array([{'-':'n','s':'s0','h':'h0'}[t] for t in ar[0]], dtype='<U19')

        for j in range(ar.shape[1]):
            for i in range(1,4):
                # if string doesn't have any objects yet
                if loc[j]=='n':
                    if ar[i,j]=='h' or ar[i,j]=='s':
                        loc[j] = ar[i,j]+str(i)
                # if string does have objects
                elif ar[i,j]=='h' or ar[i,j]=='s':
                    loc[j] = loc[j] + '_' + ar[i,j] + str(i)
        return(loc)

    def _check_obj_loc(ex, ey, obj_x, obj_y, obj_s):
        scaling = 200/24 #scaling of image compared to bounding boxes
        loc = np.array([['-','-','-','-'] for i in range(len(ex))], dtype='<U19').T

        soft = (ex>obj_x-4.5) & \
               (ex<obj_x+4.5) & \
               (ey>obj_y-4.5) & \
               (ey<obj_y+4.5)

        for i in range(4):
            loc[i,soft[i]]='s'
    
        for i in range(4):
            idx = np.argwhere(obj_s[i]=='CIRCLE')[:,0]
            r = ((ex[idx] - obj_x[i,idx])**2 + (ey[idx] - obj_y[i,idx])**2)**0.5
            loc[i,idx[r<=scaling/2]] = 'h'

            idx = np.argwhere(obj_s[i]=='SQUARE')[:,0]
            r = 0.76 * scaling/2 #0.76 is size of box in image
            left = ex[idx] - obj_x[i,idx] > -r
            right = ex[idx] - obj_x[i,idx] < r
            down = ey[idx] - obj_y[i,idx] > -r
            up = ey[idx] - obj_y[i,idx] < r
            loc[i,idx[(left) & (right) & (down) & (up)]] = 'h'

            idx = np.argwhere(obj_s[i]=='TRIANGLE')[:,0]
            r = scaling/2
            down = ey[idx] - obj_y[i,idx] > -0.72*r
            left = ~_is_left(obj_x[i,idx]-r,obj_x[i,idx],\
                            obj_y[i,idx]-0.72*r, obj_y[i,idx]+r,\
                            ex[idx], ey[idx])
            right = _is_left(obj_x[i,idx]+r,obj_x[i,idx],\
                            obj_y[i,idx]-0.72*r, obj_y[i,idx]+r,\
                            ex[idx], ey[idx])
            loc[i,idx[(down) & (left) & (right)]] = 'h'

            idx = np.argwhere(obj_s[i]=='STAR')[:,0]
            r = scaling/2
            #lines are numbered by 1 to 5, where they are 
            #down left, up right, left, down right, up left
            l1 = _is_left(obj_x[i,idx],obj_x[i,idx]-0.6*r,\
                            obj_y[i,idx]+r, obj_y[i,idx]-r,\
                            ex[idx], ey[idx])
            l2 = _is_left(obj_x[i,idx]-0.6*r,obj_x[i,idx]+r,\
                            obj_y[i,idx]-r, obj_y[i,idx]+0.28*r,\
                            ex[idx], ey[idx])
            l3 = _is_left(obj_x[i,idx]+r,obj_x[i,idx]-r,\
                            obj_y[i,idx]+0.28*r, obj_y[i,idx]+0.28*r,\
                            ex[idx], ey[idx])
            l4 = _is_left(obj_x[i,idx]-r,obj_x[i,idx]+0.6*r,\
                            obj_y[i,idx]+0.28*r, obj_y[i,idx]-r,\
                            ex[idx], ey[idx])
            l5 = _is_left(obj_x[i,idx]+0.6*r,obj_x[i,idx],\
                            obj_y[i,idx]-r, obj_y[i,idx]+r,\
                            ex[idx], ey[idx])
            center = (l1) & (l2) & (l3) & (l4) & (l5)
            a1 = (~l1) & (l3) & (l4)
            a2 = (~l2) & (l4) & (l5)
            a3 = (~l3) & (l5) & (l1)
            a4 = (~l4) & (l1) & (l2)
            a5 = (~l5) & (l2) & (l3)
            loc[i,idx[(center) | (a1) | (a2) | (a3) | (a4) | (a5)]] = 'h'

        loc_small = _combine_objs(loc)

        return(loc_small)

    def _sac_obj_location(times, ex, ey, df):
        '''
        Finds what object the eye position is currently in

        Parameters
        ----------------
        times : time of eye position
        ex : x eye position at each time
        ey : y eye position at each time
        df : behavioral dataframe with object features
        '''
        t_starts = df[(df['encode']==29) & (df['response'].isin([200,206]))].time.values+63
        t_ends = df[(df['act']=='fb') & (df['response'].isin([200,206]))].time.values
        df_temp = df[(df['encode']==29) & (df['response'].isin([200,206]))]
        obj_x_loc = []
        obj_y_loc = []
        obj_shape = []
        for i in range(4):
            obj_x_loc.append(df_temp['Item'+str(i)+'_xPos'].values)
            obj_y_loc.append(df_temp['Item'+str(i)+'_yPos'].values)
            obj_shape.append(df_temp['Item'+str(i)+'Shape'].values)
        obj_x_loc = np.array(obj_x_loc)
        obj_y_loc = np.array(obj_y_loc)
        obj_shape = np.array(obj_shape, dtype=object)

        idx = _check_what_trial(times, t_starts, t_ends)
        objs = _check_obj_loc(ex, ey, obj_x_loc[:,idx], obj_y_loc[:,idx], obj_shape[:,idx])
        return objs
    
    print('Loading dataframe and eye data...')
    print('Be patient, this is the slowest part...')
    df = startup.get_behavior(fs, species, subject, exp, session, import_obj_features=True)
    ep, ex, ey = startup.get_eye_data(fs, species, subject, exp, session)
    print('Finished loading data')
    
    print('Renormalizing eye data...')
    ex, ey = eye_calibration(ex, ey, df, trouble_shoot_plot=troubleshoot)
    
    print('Detecting Saccades...')
    sac_dist, sac_dir, sac_start, sac_end, sac_peak = eye_saccades(ex, ey, num_std=num_std)
    
    print('Creating Dataframe...')
    sac = pd.DataFrame()
    sac['distance'] = sac_dist
    sac['direction'] = sac_dir
    sac['time_start'] = sac_start
    sac['time_peak'] = sac_peak
    sac['time_end'] = sac_end

    sac['x_start'] = ex[sac.time_start.values]
    sac['y_start'] = ey[sac.time_start.values]
    sac['x_end'] = ex[sac.time_end.values]
    sac['y_end'] = ey[sac.time_end.values]
    sac['pupil_start'] = ep[sac.time_start.values]
    sac['pupil_end'] = ep[sac.time_end.values]
    
    a = sac['time_start'].values[1:]
    b = sac['time_end'].values[:-1]
    fix_len = a - b
    fix_len = np.insert(fix_len, len(fix_len), -1)
    sac['fix_len'] = fix_len

    fix_std_x = np.array([np.std(ex[np.arange(b[i],a[i])]) for i in range(len(a))])
    sac['fix_std_x'] = np.insert(fix_std_x, len(fix_std_x), -1)
    fix_std_y = np.array([np.std(ey[np.arange(b[i],a[i])]) for i in range(len(a))])
    sac['fix_std_y'] = np.insert(fix_std_x, len(fix_std_x), -1)
    
    sac['trial'] = _sac_get_trials(df, sac['time_start'].values)
    sac['interval'] = _sac_zone(df, sac)
    
    #object starting locations
    sac['obj_start'] = '-'
    idx = sac[sac['interval'].isin(['obj','obj_fix'])].index.values

    sac.loc[idx, 'obj_start'] = _sac_obj_location(
        sac[sac['interval'].isin(['obj','obj_fix'])].time_start.values,
        ex[sac[sac['interval'].isin(['obj','obj_fix'])].time_start.values],
        ey[sac[sac['interval'].isin(['obj','obj_fix'])].time_start.values],
        df
    )

    #Object ending locations
    sac['obj_end'] = '-'
    idx = sac[sac['interval'].isin(['obj','obj_fix'])].index.values

    sac.loc[idx, 'obj_end'] = _sac_obj_location(
        sac[sac['interval'].isin(['obj','obj_fix'])].time_end.values,
        ex[sac[sac['interval'].isin(['obj','obj_fix'])].time_end.values],
        ey[sac[sac['interval'].isin(['obj','obj_fix'])].time_end.values],
        df
    )
    
    #Stereotypical response
    sac = sac_get_stereotypical_response(df, sac, dir_l=dir_l, dir_h=dir_h, sac_delay_t=sac_delay_t, obj_delay_t=obj_delay_t)
    
    cols = ['trial', 'interval', 'obj_start', 'obj_end', 'distance', 'direction', 'stereotypical', \
            'time_start', 'time_end', 'time_peak', \
            'fix_len', 'fix_std_x', 'fix_std_y', \
            'x_start', 'y_start', 'x_end', 'y_end', 'pupil_start', 'pupil_end']
    sac = sac[cols]
    
    if troubleshoot:
        return sac, ex, ey
    else:
        return sac

def eye_saccades(ex, ey, num_std=1, smooth=10):
    '''
    Detects saccades
    
    Parameters
    -----------------
    ex : normalized eye (x) data
    ey : normalized eye (y) data
    num_std : number of standard deviations of distance traveled to quantify saccade
    smooth : amount of smoothing to incur before finding saccades
    
    Returns
    -----------------
    sac_dist : distance saccade travelled in total
    sac_dir : direction of each saccade in degrees
    sac_start : time start of each saccade
    sac_end : time end of each saccade
    sac_peak : peak time of each saccade
    '''
    def _distance(x,y):
        x1 = x[:-1]
        x2 = x[1:]
        y1 = y[:-1]
        y2 = y[1:]
        dist = np.sqrt(np.power(x2-x1,2) + np.power(y2-y1,2))
        return(dist)
    
    ex = moving_average_dim(ex, smooth, 0)
    ey = moving_average_dim(ey, smooth, 0)
    dist = _distance(ex, ey)
    t_adjust = int(np.round(smooth/2))
    
    idx_sac = np.argwhere(dist > num_std * np.std(dist))[:,0]
    idx_sep = np.insert(np.argwhere(idx_sac[1:]-idx_sac[:-1] > 5)[:,0]+1, 0, 0)
    sac_groups = []
    for i in range(len(idx_sep)-1):
        sac_groups.append(idx_sac[idx_sep[i]:idx_sep[i+1]])
    max_sac = []
    sac_dist = []
    sac_start = []
    sac_end = []
    sac_peak = []
    sac_dir = []
    for s in sac_groups:
        sac_start.append(s[0])
        sac_end.append(s[-1])
        
        temp = np.argmax(dist[s])
        sac_peak.append(s[temp])
        
        sac_dist.append(np.sqrt(np.power(ex[s[-1]] - ex[s[0]], 2) + np.power(ey[s[-1]] - ey[s[0]], 2)))
        sac_dir.append(np.angle((ex[s[-1]] - ex[s[0]]) + 1j*(ey[s[-1]] - ey[s[0]]), deg=True))
        
    sac_start = np.array(sac_start, dtype=int) + t_adjust
    sac_end = np.array(sac_end, dtype=int) + t_adjust
    sac_peak = np.array(sac_peak, dtype=int) + t_adjust
    sac_dist = np.array(sac_dist)
    sac_dir = np.array(sac_dir)
    return(sac_dist, sac_dir, sac_start, sac_end, sac_peak)

def eye_calibration(ex, ey, df, trouble_shoot_plot=False):
    '''
    This function renormalizes the eye data based on the fixation cross and object locations
    NOTE : Make sure df has object features
    
    Parameters
    ------------------
    ex : x position of eye data
    ey : y position of eye data
    df : behavioral dataframe with object features included
    trouble_shoot_plot : bool flag that specifies informative plot about scaling variable
    
    Returns
    ------------------
    ex3 : x position of renormalized data
    ey3 : y position of renormalized data
    '''
    #cross fixation times
    cr_fix = df[df['act']=='cross_off'].time.values-500
    
    #object locations and fixation times (only last one with full 800 ms of fixation)
    x = []
    y = []
    t = []
    for i,val in enumerate([2300,2500,2700,2900]):
        df_temp = df[
            (df['act']=='obj_fix') & \
            (df['encode']==val) & \
            (df['response'].isin([200,206]))
        ]
        x.append(df_temp['Item'+str(i)+'_xPos'].values)
        y.append(df_temp['Item'+str(i)+'_yPos'].values)
        t.append(df_temp.time.values)
    x = np.hstack(x)
    y = np.hstack(y)
    t = np.hstack(t)
    
    #expands to 800 ms
    tx_temp = np.hstack([np.arange(s,s+800) for s in t])
    x_temp = np.hstack([np.ones(800)*s for s in x])
    ty_temp = np.hstack([np.arange(s,s+800) for s in t])
    y_temp = np.hstack([np.ones(800)*s for s in y])
    
    #Re-means based on cross fixation time
    ex_m = np.mean(ex[np.hstack([np.arange(c,c+350) for c in cr_fix])])
    ey_m = np.mean(ey[np.hstack([np.arange(c,c+350) for c in cr_fix])])

    ex2 = ex - ex_m
    ey2 = ey - ey_m
    
    #Re-scales based on object final fixation times
    model = LinearRegression(fit_intercept=False)
    model.fit(np.array([x_temp]).T, ex2[tx_temp])
    x_reg = model.coef_[0]
    model.fit(np.array([y_temp]).T, ey2[ty_temp])
    y_reg = model.coef_[0]
    ex3 = ex2 / x_reg
    ey3 = ey2 / y_reg
    
    #troubleshooting. Probably can remove this, but may be helpful in evalutating code
    
    if trouble_shoot_plot:
        fig, ax = plt.subplots(1,2,figsize=(6,3),sharex=False,sharey=True)
        fig.tight_layout(pad=0)
        ax[0].scatter(x_temp, ex2[tx_temp])
        ax[0].plot(
            [-8, 8], 
            [-8 * x_reg,8 * x_reg],
            color='red'
        )
        ax[1].scatter(y_temp, ey2[ty_temp])
        ax[1].plot(
            [-8, 8], 
            [-8 * y_reg, 8 * y_reg],
            color='red'
        )
        for i in range(-8,9):
            if np.any(x_temp==i):
                ax[0].scatter(i, np.mean(ex2[tx_temp][x_temp==i]), color='red')
            if np.any(y_temp==i):
                ax[1].scatter(i, np.mean(ey2[ty_temp][y_temp==i]), color='red')
        ax[0].set_xlabel('x vis degrees')
        ax[1].set_xlabel('y vis degrees')
        ax[0].set_ylabel('x/y eye units')
        fig.subplots_adjust(bottom=0.2)
        fig.subplots_adjust(left=0.12)
    
    return(ex3, ey3)

#Above, all for saccade related things


def dataframe_insert_row(row_number, df, row_value):
    '''
    Inserts row into pandas dataframe. Code taken from
    https://www.geeksforgeeks.org/insert-row-at-given-position-in-pandas-dataframe/
    
    Parameters
    --------------------
    row_number : row number to insert into
    df : dataframe
    row_value : value of row to insert
    
    Returns
    --------------------
    df : dataframe with inserted row
    '''
    start_upper = 0
    end_upper = row_number
    start_lower = row_number
    end_lower = df.shape[0]
    upper_half = [*range(start_upper, end_upper, 1)]
    lower_half = [*range(start_lower, end_lower, 1)]
    lower_half = [x.__add__(1) for x in lower_half]
    index_ = upper_half + lower_half
    df.index = index_
    df.loc[row_number] = row_value
    df = df.sort_index()
    return df

from sklearn.cluster import KMeans
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

def get_saccade_seq(df, dtype='all', length=0, ignore_first=True):
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
    ignore_first : flag to ignore first, stereotypical saccade that occurs before ~200ms
    
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
        if not ignore_first:
            allSeq.append(temp)
        else:
            a2 = df[(df['trial']==t) & (df['act'].isin(['obj_fix_break', 'obj_fix'])) & (df['ignore']==0) & (df['badGroup']==0)].time.values
            a1 = df[(df['trial']==t) & (df['act'].isin(['cross_off'])) & (df['ignore']==0) & (df['badGroup']==0)].time.values
            if len(a2)>0 and len(a1)>0 and a2[0]-a1[0]<200:
                allSeq.append(temp[1:])
            else:
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

def get_bad_channels(species, subject, exp, session):
    """
    Finds and returns the bad channels of a given subject and session.
    
    Parameters
    ---------------
    species : the species
    subject: the subject's name
    exp: the experiment selected
    session: the session id
    
    Returns
    ---------------
    List of bad channels
    """
    bad_channels = general.load_json_file('sp-'+species+'_sub-'+subject+'_exp-'+exp+'_bad_channels.json')
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