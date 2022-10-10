import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import zscore
import time
import csv
import os
import nibabel
from sklearn.metrics.pairwise import euclidean_distances
from scipy.ndimage.filters import gaussian_filter
import scipy.io as sio

from ridge_tools import cross_val_ridge, corr
import time as tm

subject_runs = dict(F = [4,5,6,7],
        G = [3,4,5,6],
        H = [3,4,9,10],
        I = [7,8,9,10],
        J = [7,8,9,10],
        K = [7,8,9,10],
        L = [7,8,9,10],
        M = [7,8,9,10],
        N= [7,8,9,10])

surfaces = dict( F = 'fMRI_story_F',
        G = 'fMRI_story_G',
        H = 'fMRI_story_H',
        I = 'fMRI_story_I',
        J = 'fMRI_story_J',
        K = 'fMRI_story_K',
        L = 'fMRI_story_L',
        M = 'fMRI_story_M',
        N = 'fMRI_story_N')

transforms = dict( F = 'F_ars_auto2',
        G = 'G_ars_auto2',
        H = 'H_ars_auto2',
        I = 'I_ars_auto2',
        J = 'J_ars_auto2',
        K = 'K_ars_auto2',
        L = 'L_ars_auto2',
        M = 'M_ars_auto2',
        N = 'N_ars_auto2')
    
def load_transpose_zscore(file): 
    dat = nibabel.load(file).get_data()
    dat = dat.T
    return zscore(dat,axis = 0)

def smooth_run_not_masked(data,smooth_factor):
    smoothed_data = np.zeros_like(data)
    for i,d in enumerate(data):
        smoothed_data[i] = gaussian_filter(data[i], sigma=smooth_factor, order=0, output=None,
                 mode='reflect', cval=0.0, truncate=4.0)
    return smoothed_data
    
def zscore_word_data(data):
    # zscores time over each time window, and returns a 2D data structure.
    # to zscore over all time windows, and not by time window, use function above
    n_words = data.shape[0]
    data = np.reshape(data,[n_words,-1])
    data = np.nan_to_num(zscore(data))
    return data

def delay_one(mat, d):
        # delays a matrix by a delay d. Positive d ==> row t has row t-d
    new_mat = np.zeros_like(mat)
    if d>0:
        new_mat[d:] = mat[:-d]
    elif d<0:
        new_mat[:d] = mat[-d:]
    else:
        new_mat = mat
    return new_mat

def delay_mat(mat, delays):
        # delays a matrix by a set of delays d.
        # a row t in the returned matrix has the concatenated:
        # row(t-delays[0],t-delays[1]...t-delays[last] )
    new_mat = np.concatenate([delay_one(mat, d) for d in delays],axis = -1)
    return new_mat

# train/test is the full NLP feature
# train/test_pca is the NLP feature reduced to 10 dimensions via PCA that has been fit on the training data
# feat_dir is the directory where the NLP features are stored
# train_indicator is an array of 0s and 1s indicating whether the word at this index is in the training set
def get_nlp_features_fixed_length(layer, seq_len, feat_type, feat_dir, train_indicator, SKIP_WORDS=20, END_WORDS=5176):
  
    if layer == -1 and feat_type == 'bert':
        all_layers_train = []
        all_layers_test = []
        for layer2 in range(13):
            loaded = np.load(feat_dir + feat_type + '_length_'+str(seq_len)+ '_layer_' + str(layer2) + '.npy')
            train = loaded[SKIP_WORDS:END_WORDS,:][train_indicator]
            test = loaded[SKIP_WORDS:END_WORDS,:][~train_indicator]
        
            pca = PCA(n_components=10, svd_solver='full')
            pca.fit(train)
            train_pca = pca.transform(train)
            test_pca = pca.transform(test)
    
            all_layers_train.append(train_pca)
            all_layers_test.append(test_pca)

        return train_pca,test_pca, np.hstack(all_layers_train), np.hstack(all_layers_test)

  
    loaded = np.load(feat_dir + feat_type + '_length_'+str(seq_len)+ '_layer_' + str(layer) + '.npy')
    if feat_type == 'elmo':
        train = loaded[SKIP_WORDS:END_WORDS,:][:,:512][train_indicator]   # only forward LSTM
        test = loaded[SKIP_WORDS:END_WORDS,:][:,:512][~train_indicator]   # only forward LSTM
    elif feat_type == 'gpt2':
        train = loaded[SKIP_WORDS:END_WORDS,:][train_indicator]
        test = loaded[SKIP_WORDS:END_WORDS,:][~train_indicator]
    else:
        print('Unrecognized NLP feature type {}. Available options elmo, gpt2'.format(feat_type))
    
    pca = PCA(n_components=10, svd_solver='full')
    pca.fit(train)
    train_pca = pca.transform(train)
    test_pca = pca.transform(test)

    return train, test, train_pca, test_pca 

def load_features(feat_name_split, delay, train_indicator, feat_type='', feat_dir=''):
        SKIP_WORDS = 20
        END_WORDS = 5176
        if 'emb' in feat_name_split:
            train, test,_,_ = get_nlp_features_fixed_length(0, 1, feat_type, feat_dir, train_indicator)
        elif 'context' in feat_name_split and feat_type == 'elmo':
            train, test,_,_ = get_nlp_features_fixed_length(1, 25, feat_type, feat_dir, train_indicator)
        elif 'context' in feat_name_split and feat_type == 'gpt2':
            train, test,_,_ = get_nlp_features_fixed_length(11, 25, feat_type, feat_dir, train_indicator)
        else:    
            print('Unrecognized feat type {}'.format(feat_name_split))
            return None

        feature_train = np.roll(train,delay,axis=0)
        feature_test = np.roll(test,delay,axis=0)
        
        return feature_train, feature_test


# train_indicator is an array of 0s and 1s indicating whether the word at this index is in the training set
def load_features_to_regress_out(feat_name_list, train_indicator, feat_type='', feat_dir='',SKIP_WORDS=20, END_WORDS=5176):
        
    if len(feat_name_list) == 0:
        return [],[]
        
    regress_features_train = []
    regress_features_test = []
    
    print(feat_name_list)
    for feat_name in feat_name_list:
        
        feat_name_split = feat_name.split('-')
        
        if 'prev' in feat_name_split or 'back' in feat_name_split:
            delay = int(feat_name_split[1])
        elif 'next' in feat_name_split or 'fwd' in feat_name_split:
            delay = -int(feat_name_split[1])
        else:
            delay = 0
        
        if delay == 0:
            print('using delay of {}'.format(delay))
            feature_train, feature_test = load_features(feat_name_split, delay, train_indicator, feat_type, feat_dir)

            regress_features_train.append(feature_train)
            regress_features_test.append(feature_test)
        else:
            while abs(delay) > 0:
                print('using delay of {}'.format(delay))
        
                feature_train, feature_test = load_features(feat_name_split, delay, train_indicator, feat_type, feat_dir)

                regress_features_train.append(feature_train)
                regress_features_test.append(feature_test)  
                
                delay = delay - np.sign(delay)
                
                if 'back' in feat_name_split or 'fwd' in feat_name_split:   # only want the features at a particular delayed position, not all feats up to that point
                        break
                                        
    return np.hstack(regress_features_train), np.hstack(regress_features_test)


def CV_ind(n, n_folds):
    ind = np.zeros((n))
    n_items = int(np.floor(n/n_folds))
    for i in range(0,n_folds -1):
        ind[i*n_items:(i+1)*n_items] = i
    ind[(n_folds-1)*n_items:] = (n_folds-1)
    return ind

def TR_to_word_CV_ind(TR_train_indicator,SKIP_WORDS=20,END_WORDS=5176):
    time = np.load('./data/time_fmri.npy')
    runs = np.load('./data/runs_fmri.npy') 
    time_words = np.load('./data/time_words_fmri.npy')
    time_words = time_words[SKIP_WORDS:END_WORDS]
        
    word_train_indicator = np.zeros([len(time_words)], dtype=bool)    
    words_id = np.zeros([len(time_words)],dtype=int)
    # w=find what TR each word belongs to
    for i in range(len(time_words)):                
        words_id[i] = np.where(time_words[i]> time)[0][-1]
        
        if words_id[i] <= len(runs) - 15:
            offset = runs[int(words_id[i])]*20 + (runs[int(words_id[i])]-1)*15
            if TR_train_indicator[int(words_id[i])-offset-1] == 1:
                word_train_indicator[i] = True
    return word_train_indicator        

def run_class_time_CV_crossval_ridge(data, predict_feat_dict, 
                   regress_feat_names_list=[], SKIP_WORDS = 5, END_WORDS = 5176,
                   delays = [0], encoding=1, detrend = True, do_correct = [], n_folds = 4, splay = [],
                   do_acc = True, frequency= 0, downsampled=1, seed=0):

    # name = subject name
    # features = NLP features for all words
    # SKIP_WORDS = how many words to skip from the beginning in case the features are not good there
    # END_WORDS = how many words to skip from the end in case the features are not good there
    # method = ridge method: plain, svd, kernel_ridge, kernel_ridge_svd, ridge_sk
    # lambdas = lambdas to try
    # delays = look at current word + which other words? 0 = current, -1 previous, +1 next.
    #          most common is [-2,-1,0,1,2]
    # detrend = remove mean of last 5 words? MAYBE THIS SHOULD BE FALSE WHEN LOOKING AT CONTEXT
    # do_correct = not used now
    # n_folds = number of CV folds
    # splay = only do the analysis on the words in this array
    # do_acc = run single subject classification

    # detrend
    predict_feat_type = predict_feat_dict['feat_type']
    nlp_feat_type = predict_feat_dict['nlp_feat_type']
    feat_dir = predict_feat_dict['nlp_feat_dir']
    layer = predict_feat_dict['layer']
    seq_len = predict_feat_dict['seq_len']


    n_words = data.shape[0]
    if detrend:
        running_mean = np.vstack([np.mean(np.mean(data[i-5:i,:,:],2),0) for i in range(5,n_words)])
        data[5:] = np.stack([(data[5:,:,i]-running_mean).T for i in range(data.shape[2])]).T
      
    n_words = data.shape[0]
    n_time = data.shape[2]
    n_sensor = data.shape[1]

    ind = CV_ind(n_words, n_folds=n_folds)

    corrs = np.zeros((n_folds, n_time))
    acc = np.zeros((n_folds, n_time))

    preds_d = [] 
    all_preds = []

    all_test_data = []

    for ind_num in range(n_folds):
        start_time = time.time()

        train_ind = ind!=ind_num
        test_ind = ind==ind_num
        
        if predict_feat_type == 'elmo' or predict_feat_type == 'bert':
                train_features,test_features,_,_ = get_nlp_features_fixed_length(layer, seq_len, nlp_feat_type, feat_dir, train_ind)
        else:
                train_features,test_features = load_features_to_regress_out([predict_feat_type], train_ind, nlp_feat_type, feat_dir)

        # split data
        train_data = data[train_ind]
        test_data = data[test_ind]

        # normalize data
        train_data = zscore_word_data(train_data)
        test_data = zscore_word_data(test_data)

        all_test_data.append(test_data)

        train_features = np.nan_to_num(zscore(train_features)) 
        test_features = np.nan_to_num(zscore(test_features)) 
        
        # if regressing out features, do it now
        if len(regress_feat_names_list) > 0:

            print('working on regressing out {}'.format(regress_feat_names_list))

            regress_train_features, regress_test_features = load_features_to_regress_out(regress_feat_names_list, train_ind, nlp_feat_type, feat_dir)
            
            preds_test,preds_train,_,_ = cross_val_ridge(regress_train_features,train_features,regress_test_features, n_splits = 10, lambdas = np.array([10**i for i in range(-6,10)]), method = 'kernel_ridge',do_plot = False)  
            
            train_features = np.reshape(train_features-preds_train, train_features.shape)
            test_features = np.reshape(test_features-preds_test, test_features.shape)
            print('done regressing out')

        if encoding == 0: # decoding experiment
            preds,_,_,_ = cross_val_ridge(train_data,train_features,test_data,n_splits = 10,lambdas = np.array([10**i for i in range(-6,10)]), method = 'kernel_ridge',do_plot = False)
        else:
            preds,_,_,_ = cross_val_ridge(train_features,train_data,test_features,n_splits = 10,lambdas = np.array([10**i for i in range(-6,10)]), method = 'kernel_ridge',do_plot = False)
        
        if encoding == 0:
            corrs[ind_num,:] = corr(preds,test_features).mean(0)
        if encoding == 1:
            corrs[ind_num,:] = corr(preds,test_data).reshape(n_sensor,n_time).mean(0)
        
        all_preds.append(preds)
        n_pred = preds.shape[0]
        #del  weights

        print('CV fold ' + str(ind_num) + ' ' + str(time.time()-start_time))


    return corrs, np.vstack(all_preds), np.vstack(all_test_data)


def prepare_fmri_features(train_features, test_features, word_train_indicator, TR_train_indicator, SKIP_WORDS=20, END_WORDS=5176):
        
    time = np.load('./data/time_fmri.npy')
    runs = np.load('./data/runs_fmri.npy') 
    time_words = np.load('./data/time_words_fmri.npy')
    time_words = time_words[SKIP_WORDS:END_WORDS]
        
    words_id = np.zeros([len(time_words)])
    # w=find what TR each word belongs to
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words[i]> time)[0][-1]
        
    all_features = np.zeros([time_words.shape[0], train_features.shape[1]])
    all_features[word_train_indicator] = train_features
    all_features[~word_train_indicator] = test_features
        
    p = all_features.shape[1]
    tmp = np.zeros([time.shape[0], p])
    for i in range(time.shape[0]):
        tmp[i] = np.mean(all_features[(words_id<=i)*(words_id>i-1)],0)
    tmp = delay_mat(tmp, np.arange(1,5))

    # remove the edges of each run
    tmp = np.vstack([zscore(tmp[runs==i][20:-15]) for i in range(1,5)])
    tmp = np.nan_to_num(tmp)
        
    return tmp[TR_train_indicator], tmp[~TR_train_indicator]

  

def run_class_time_CV_fmri_crossval_ridge(data, predict_feat_dict,
                                          regress_feat_names_list = [],method = 'kernel_ridge', 
                                          lambdas = np.array([0.1,1,10,100,1000]),
                                          detrend = False, n_folds = 4, skip=5):
    
    predict_feat_type = predict_feat_dict['feat_type']
    nlp_feat_type = predict_feat_dict['nlp_feat_type']
    feat_dir = predict_feat_dict['nlp_feat_dir']
    layer = predict_feat_dict['layer']
    seq_len = predict_feat_dict['seq_len']
        
        
    n_words = data.shape[0]
    n_voxels = data.shape[1]
        
    print(n_words)

    ind = CV_ind(n_words, n_folds=n_folds)

    corrs = np.zeros((n_folds, n_voxels))
    acc = np.zeros((n_folds, n_voxels))
    acc_std = np.zeros((n_folds, n_voxels))
    preds_d = np.zeros((data.shape[0], data.shape[1]))

    all_test_data = []
    all_preds = []
    
    for ind_num in range(n_folds):
        train_ind = ind!=ind_num
        test_ind = ind==ind_num
        
        word_CV_ind = TR_to_word_CV_ind(train_ind)
        if nlp_feat_type == 'brain': 
                word_CV_ind = train_ind
        
        if predict_feat_type == 'elmo' or predict_feat_type == 'bert':
                tmp_train_features,tmp_test_features,_,_ = get_nlp_features_fixed_length(layer, seq_len, nlp_feat_type, feat_dir, word_CV_ind)
        else:
                tmp_train_features,tmp_test_features = load_features_to_regress_out(predict_feat_type.split('+'), word_CV_ind, nlp_feat_type, feat_dir)
        
        if nlp_feat_type != 'brain': 
            train_features,test_features = prepare_fmri_features(tmp_train_features, tmp_test_features, word_CV_ind, train_ind)
        else: # no need to concatenate multiple TRs for brain to brain predictions
            train_features = tmp_train_features
            test_features = tmp_test_features
        
        if len(regress_feat_names_list) > 0:
            tmp_regress_train_features, tmp_regress_test_features = load_features_to_regress_out(regress_feat_names_list, word_CV_ind, nlp_feat_type, feat_dir) 
            
            if nlp_feat_type != 'brain':     
                regress_train_features,regress_test_features = prepare_fmri_features(tmp_regress_train_features, tmp_regress_test_features, word_CV_ind, train_ind)
            else:
                regress_train_features = tmp_regress_train_features
                regress_test_features = tmp_regress_test_features
                
        # split data
        train_data = data[train_ind]
        test_data = data[test_ind]

        # skip TRs between train and test data
        if ind_num == 0: # just remove from front end
            train_data = train_data[skip:,:]
            train_features = train_features[skip:,:]
        elif ind_num == n_folds-1: # just remove from back end
            train_data = train_data[:-skip,:]
            train_features = train_features[:-skip,:]
        else:
            test_data = test_data[skip:-skip,:]
            test_features = test_features[skip:-skip,:]

        # normalize data
        train_data = np.nan_to_num(zscore(np.nan_to_num(train_data)))
        test_data = np.nan_to_num(zscore(np.nan_to_num(test_data)))
        all_test_data.append(test_data)
        
        train_features = np.nan_to_num(zscore(train_features))
        test_features = np.nan_to_num(zscore(test_features))

        print('features size: {}, data size: {}'.format(train_features.shape, train_data.shape))
        
        # if regressing out features, do it now
        if len(regress_feat_names_list) > 0:
            
            # skip TRs between train and test data
            if ind_num == 0: # just remove from front end
                regress_train_features = regress_train_features[skip:,:]
            elif ind_num == n_folds-1: # just remove from back end
                regress_train_features = regress_train_features[:-skip,:]
            else:
                regress_test_features = regress_test_features[skip:-skip,:]

            print('after regressing, features size: {}, data size: {}'.format(regress_train_features.shape, train_features.shape))
            
            preds_test,preds_train,_,_ = cross_val_ridge(regress_train_features,train_features,regress_test_features, n_splits = 10, lambdas = np.array([10**i for i in range(-6,10)]), method = 'kernel_ridge',do_plot = False)
                        
            train_features = np.reshape(train_features-preds_train, train_features.shape)
            test_features = np.reshape(test_features-preds_test, test_features.shape)
            print('done regressing out')
        
        start_time = tm.time()

        preds,_,_,chosen_lambdas = cross_val_ridge(train_features,train_data,test_features, n_splits = 10, lambdas = np.array([10**i for i in range(-6,10)]), method = 'kernel_ridge',do_plot = False)
        corrs[ind_num,:] = corr(preds,test_data)
        all_preds.append(preds)     

        print('fold {} completed, took {} seconds'.format(ind_num, tm.time()-start_time))


    return corrs, acc, acc_std, np.vstack(all_preds), np.vstack(all_test_data)
