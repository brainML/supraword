# supraword
code for the paper "Combining computational controls with natural text reveals new aspects of meaning composition"

The requirements for running the code are:
- python>=3.7
- numpy
- scipy
- sklearn
- pickle
- nibabel


The fMRI data can be found at:

http://www.cs.cmu.edu/~fmri/plosone/

The MEG data can be found at:

https://kilthub.cmu.edu/articles/dataset/RSVP_reading_of_book_chapter_in_MEG/20465898


Demo & Running on our data:

For results with fMRI data & GPT-2 model:

1. Full-context results command:

`python make_predictions.py --subject I --recording_type fmri --predict_feat_type prev-1-context --nlp_feat_type gpt2 --nlp_feat_dir ./data/gpt2/ --layer 11 --sequence_length 25 --output_fname_prefix fmri_I_gpt2 --output_dir ./results/ --regress_feat_types 0`

Expected runtime: 11min per participant

Expected output: 

```python
import numpy as np
loaded = np.load('./results/predict_fmri_I_gpt2_with_prev-1-context.npy',allow_pickle=True)
np.mean(loaded.item()['corrs_t'],0)[:25]
array([ 0.02902328,  0.01342933,  0.02277539,  0.01873558,  0.01578286,
        0.03741205,  0.0140556 , -0.02027801, -0.05186078, -0.01530757,
        0.05624073,  0.00524233, -0.00483687, -0.04228178, -0.0344222 ,
        0.09171423,  0.00107362, -0.02035646, -0.04976865,  0.1292592 ,
        0.0736794 ,  0.01735717, -0.01939861, -0.05969443, -0.07343419])
 ```
        
These are the encoding model performances (Pearson correlation) when predicting each voxel using the full context embedding from GPT2. This participant has 25263 voxels so we display the encoding model performance for the first 25 here.


2. Supra-word meaning:

`
python make_predictions.py --subject I --recording_type fmri --predict_feat_type prev-1-context --nlp_feat_type gpt2 --nlp_feat_dir ./data/gpt2/ --layer 11 --sequence_length 25 --output_fname_prefix fmri_I_gpt2 --output_dir ./results/ --regress_feat_types emb+prev-24-emb`

Expected runtime: 11min per participant

Expected output: 

``` python
import numpy as np
loaded = np.load('./results/predict_fmri_I_gpt2_with_prev-1-context_regress_out_emb+prev-24-emb.npy',allow_pickle=True)
np.mean(loaded.item()['corrs_t'],0)[:25]
array([-3.00731641e-02, -2.02737538e-02, -1.95567325e-02, -1.23296141e-02,
       -2.04915875e-02, -7.92662865e-03, -1.67378930e-02, -7.40570716e-03,
       -5.61102306e-02, -2.61383422e-02,  6.33055815e-02,  3.17776063e-03,
       -5.31062895e-06, -5.74376843e-02, -2.45183904e-02,  1.21760972e-01,
        3.15045688e-02, -4.91309244e-03, -4.75399857e-02,  1.52140082e-01,
        1.08481354e-01,  3.36226835e-02, -9.64076746e-03, -3.87531218e-02,
       -3.62025592e-02])
```

These are the encoding model performances (Pearson correlation) when predicting each voxel using the supra-word meaning (i.e. residual context embedding) from the penultimate layer of GPT-2. This participant has 25263 voxels so we display the encoding model performance for the first 25 here.


For results with MEG data & GPT-2 model:

1. Full-context results command:

`python make_predictions.py --subject D --recording_type meg --predict_feat_type prev-1-context --nlp_feat_type gpt2 --nlp_feat_dir ./data/gpt2/ --layer 11 --sequence_length 25 --output_fname_prefix meg_D_gpt2 --output_dir ./results/ --regress_feat_types 0`

Expected runtime: 261min per participant

Expected output: 

``` python
import numpy as np
loaded = np.load('./results/predict_meg_D_gpt2_with_prev-1-context.npy',allow_pickle=True)
np.mean(loaded.item()['corrs_t'],0)[:25]
array([0.01374132, 0.01776978, 0.01301394, 0.02475237, 0.02904284,
       0.02338376, 0.03465129, 0.03131738, 0.02112555, 0.02579657,
       0.02320374, 0.02312739, 0.01368657, 0.01851471, 0.02092678,
       0.01719037, 0.01477792, 0.01266554, 0.01921032, 0.0055954 ])
```

2. Supra-word meaning: 

`python make_predictions.py --subject D --recording_type meg --predict_feat_type prev-1-context --nlp_feat_type gpt2 --nlp_feat_dir ./data/gpt2/ --layer 11 --sequence_length 25 --output_fname_prefix meg_D_gpt2 --output_dir ./results/ --regress_feat_types emb+prev-24-emb`

Expected runtime: 261min per participant

```python
loaded = np.load('./results/predict_meg_D_gpt2_with_prev-1-context_regress_out_emb+prev-24-emb.npy', allow_pickle=True)
array([ 0.00762598, -0.0009894 ,  0.00106046,  0.00494969,  0.00305991,
        0.00255424,  0.00797679,  0.00432561,  0.00046927,  0.01576027,
        0.00598737,  0.00678923,  0.00185299,  0.00193773,  0.01042306,
        0.00111698,  0.00804389, -0.00332516,  0.01382172,  0.00334234])
```



For results with fMRI data & ELMo model:

1. Full-context results command:

`python make_predictions.py --subject I --recording_type fmri --predict_feat_type prev-1-context --nlp_feat_type elmo --nlp_feat_dir ./data/elmo/ --layer 1 --sequence_length 25 --output_fname_prefix fmri_I --output_dir ./results/ --regress_feat_types 0`

Expected runtime: 11min per participant

Expected output: 

```python
import numpy as np
loaded = np.load('./results/predict_fmri_I_with_prev-1-context.npy',allow_pickle=True)
np.mean(loaded.item()['corrs_t'],0)[:25]
array([ 0.04949864,  0.01109111,  0.05698816,  0.0180301 ,  0.01225051,
        0.02465458,  0.01527751, -0.00331915, -0.00756025,  0.01988832,
        0.07582428,  0.03013169, -0.01074882, -0.02515144,  0.01221097,
        0.14168146,  0.06732073, -0.00217565, -0.03507615,  0.13591334,
        0.10483074,  0.05328697, -0.00260458, -0.04421964, -0.05260348])
```        

These are the encoding model performances (Pearson correlation) when predicting each voxel using the full context embedding. This participant has 25263 voxels so we display the encoding model performance for the first 25 here.

2. Supra-word meaning:

`python make_predictions.py --subject I --recording_type fmri --predict_feat_type prev-1-context --nlp_feat_type elmo --nlp_feat_dir ./data/elmo/ --layer 1 --sequence_length 25 --output_fname_prefix fmri_I --output_dir ./results/ --regress_feat_types emb+prev-24-emb`

Expected runtime: 11min per participant

Expected output: 

```python
import numpy as np
loaded = np.load('./results/predict_fmri_I_with_prev-1-context_regress_out_emb+prev-24-emb.npy',allow_pickle=True)
np.mean(loaded.item()['corrs_t'],0)[:25]
array([ 0.01295854, -0.02818651,  0.00588465, -0.03213746, -0.07086528,
       -0.0276626 , -0.04230826, -0.05919864, -0.06583684, -0.00955516,
        0.04938263,  0.00235666, -0.04789345, -0.07087525, -0.01519331,
        0.06970931,  0.01230002, -0.04463024, -0.0689993 ,  0.06543832,
        0.03395289, -0.0118709 , -0.05123952, -0.06949959, -0.03490737])
```

These are the encoding model performances (Pearson correlation) when predicting each voxel using the supra-word meaning (i.e. residual context embedding). This participant has 25263 voxels so we display the encoding model performance for the first 25 here.

For results with MEG data & ELMo model: 

1. Full-context: 

`python make_predictions.py --subject D --recording_type meg --predict_feat_type prev-1-context --nlp_feat_type elmo --nlp_feat_dir ./data/elmo/ --layer 1 --sequence_length 25 --output_fname_prefix meg_D --output_dir ./results/ --regress_feat_types 0`

Expected runtime: 261min per participant

Expected output: 

```python
import numpy as np
loaded = np.load('./results/predict_meg_D_with_prev-1-context.npy',allow_pickle=True)
print(np.mean(loaded.item()['corrs_t'],0))
array([0.01614299, 0.01912072, 0.01405379, 0.021767  , 0.02621933,
       0.02671252, 0.03433858, 0.02835071, 0.02823389, 0.02346108,
       0.01917116, 0.02398674, 0.01968492, 0.02208394, 0.0172712 ,
       0.01664276, 0.01779479, 0.01714125, 0.01138398, 0.00638619])
```

These are the encoding model performances (Pearson correlation) when predicting each 25ms window in the 500ms word presentation using the full context embedding.

2. Supra-word meaning: 

`python make_predictions.py --subject D --recording_type meg --predict_feat_type prev-1-context --nlp_feat_type elmo --nlp_feat_dir ./data/elmo/ --layer 1 --sequence_length 25 --output_fname_prefix meg_D --output_dir ./results/ --regress_feat_types emb+prev-24-emb`

Expected runtime: 90min per participant

Expected output: 

```python
import numpy as np
loaded = np.load('./results/predict_meg_D_with_prev-1-context_regress_out_emb+prev-24-emb.npy',allow_pickle=True)
print(np.mean(loaded.item()['corrs_t'],0))
array([ 0.00567645,  0.0025197 , -0.00394649,  0.00748673,  0.0067108 ,
        0.00760484, -0.00291998,  0.01019912,  0.008096  ,  0.00986465,
       -0.00028717,  0.00382227,  0.00421218,  0.00365157,  0.00820432,
       -0.00013302,  0.00513469,  0.00072389,  0.01121881, -0.00176342])
```

These are the encoding model performances (Pearson correlation) when predicting each 25ms window in the 500ms word presentation using the supra-word meaning (i.e. residual context embedding).
