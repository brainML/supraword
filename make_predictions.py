import argparse
import numpy as np
import warnings

from utils import delay_mat, delay_one, zscore_word_data, run_class_time_CV_crossval_ridge, run_class_time_CV_fmri_crossval_ridge
from utils import subject_runs, surfaces, transforms, load_transpose_zscore, smooth_run_not_masked, CV_ind


from sklearn.decomposition import PCA
from scipy.stats import zscore
import scipy.io as sio

import time as tm
from scipy import signal
import pickle as pk

SKIP_WORDS = 20
END_WORDS = 5176
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--predict_feat_type", required=True)
    parser.add_argument("--nlp_feat_type", required=True)
    parser.add_argument("--nlp_feat_dir", required=True)
    parser.add_argument("--layer", type=int, required=False)
    parser.add_argument("--sequence_length", type=int, required=False)
    parser.add_argument("--output_fname_prefix", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--recording_type", required=True)
    parser.add_argument("--regress_feat_types", default=0)
    parser.add_argument("--encoding", type=int, default=1)
    
    args = parser.parse_args()
    print(args)
    
    if args.regress_feat_types != '0':
        regress_feat_names_list = np.sort(args.regress_feat_types.split('+'))
    else:
        regress_feat_names_list = []
        
    predict_feat_dict = {'feat_type':args.predict_feat_type, 
                         'nlp_feat_type':args.nlp_feat_type,
                         'nlp_feat_dir':args.nlp_feat_dir,
                         'layer':args.layer,
                         'seq_len':args.sequence_length}
    warnings.filterwarnings("ignore")

    if args.recording_type == 'meg':
    
        # predict the previous n words, current word, and the next n words
        delays =  list(np.arange(args.delay,-args.delay-1,-1)) 
        #print(delays)

        mat = sio.loadmat('./data/meg/{}_HP_notDetrended_25ms.mat'.format(args.subject))
        data = mat['data']

        n_words = data.shape[0]
        n_time = data.shape[2]
        n_sensor = data.shape[1]

        data = data[SKIP_WORDS:END_WORDS]
        data = delay_mat(data,delays)
        print(data.shape)


        corrs_t, preds_t, test_t = run_class_time_CV_crossval_ridge(data, 
                                                                    predict_feat_dict,
                                                                    regress_feat_names_list = regress_feat_names_list,
                                                                    delays = delays,
                                                                    encoding=args.encoding,
                                                                    detrend = False,
                                                                    do_correct = [],
                                                                    splay = [],
                                                                    do_acc = False,
                                                                    SKIP_WORDS = SKIP_WORDS,
                                                                    END_WORDS = END_WORDS)
    elif args.recording_type == 'fmri':
        
        # load fMRI data
        data = np.load('./data/fmri/data_subject_{}.npy'.format(args.subject))
        corrs_t, _, _, preds_t, test_t = run_class_time_CV_fmri_crossval_ridge(data,
                                                                    predict_feat_dict,
                                                                    regress_feat_names_list=regress_feat_names_list)
        
    else:
        print('Unrecognized recording_type {}. Options: meg, fmri'.format(args.recording_type))
        
   
    if args.predict_feat_type == 'elmo' or args.predict_feat_type == 'bert':
        if args.regress_feat_types != '0':
            fname = 'predict_{}_with_{}_layer_{}_len_{}_regress_out_{}'.format(args.output_fname_prefix, args.predict_feat_type, args.layer, args.sequence_length, '+'.join(regress_feat_names_list))
        else:
            fname = 'predict_{}_with_{}_layer_{}_len_{}'.format(args.output_fname_prefix, args.predict_feat_type, args.layer, args.sequence_length)
    else:
        if args.regress_feat_types != '0':
            fname = 'predict_{}_with_{}_regress_out_{}'.format(args.output_fname_prefix, args.predict_feat_type, '+'.join(regress_feat_names_list))
        else:
            fname = 'predict_{}_with_{}'.format(args.output_fname_prefix, args.predict_feat_type)

    print('saving: {}'.format(args.output_dir + fname))
    np.save(args.output_dir + fname + '.npy', {'corrs_t':corrs_t,'preds_t':preds_t,'test_t':test_t}) 
