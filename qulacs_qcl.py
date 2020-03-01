import numpy as np
import matplotlib.pyplot as plt
import time

## Iris dataset
import pandas as pd
from sklearn import datasets
from sklearn.metrics import roc_curve, auc, roc_auc_score
#from scipy.interpolate import spline


# class label, lepton 1 pT, lepton 1 eta, lepton 1 phi, lepton 2 pT, lepton 2 eta, lepton 2 phi, 
# missing energy magnitude, missing energy phi, MET_rel, axial MET, M_R, M_TR_2, R, MT2, S_R, 
# M_Delta_R, dPhi_r_b, cos(theta_r1)
df = pd.read_csv("Files/SUSY_1M.csv",names=('isSignal','lep1_pt','lep1_eta','lep1_phi','lep2_pt','lep2_eta','lep2_phi','miss_ene','miss_phi','MET_rel','axial_MET','M_R','M_TR_2','R','MT2','S_R','M_Delta_R','dPhi_r_b','cos_theta_r1'))

nevt_offset=500000
uin=UIN
jobn=JOBN
nevt = NEVT
niter = NITER
nvar = NVAR
if nvar == 3:
    SelectedFeatures = ['lep1_pt','lep2_pt','miss_ene']
elif nvar == 5:
    SelectedFeatures = ['lep1_pt','lep2_pt','miss_ene','M_TR_2','M_Delta_R']
elif nvar == 7:
    SelectedFeatures = ['lep1_pt','lep1_eta','lep2_pt','lep2_eta','miss_ene','M_TR_2','M_Delta_R']
opt='OPTION'


y_train_label = df['isSignal'].values[nevt*(jobn-1):nevt*jobn]
y_test_label = df['isSignal'].values[nevt_offset+nevt*(jobn-1):nevt_offset+nevt*jobn]
x_train = df[SelectedFeatures].values[nevt*(jobn-1):nevt*jobn] 
x_test = df[SelectedFeatures].values[nevt_offset+nevt*(jobn-1):nevt_offset+nevt*jobn]
y_train = np.eye(2)[pd.Series(y_train_label,dtype='int32')] # one-hot representation - shape:(150, 2)
y_test = np.eye(2)[pd.Series(y_test_label,dtype='int32')] # one-hot representation - shape:(150, 2)
label_names = ['background','signal']


from qcl_classification import QclClassification

# Random number
random_seed = RNUM+jobn
np.random.seed(random_seed)

#b Circuit paramter
nqubit = len(SelectedFeatures)
c_depth = NDEPTH
num_class = 2

# Instance of QclClassification
qcl = QclClassification(nqubit, c_depth, num_class)

# Training with BFGS
res, theta_init, theta_opt = qcl.fit(x_train, y_train, uin_type=uin, maxiter=niter)


# Testing
qcl.set_input_state(x_test, uin)
Zprob = qcl.pred(theta_opt) # Update model parameter theta    
Zpred = np.argmax(Zprob, axis=1)

# Check prediction for traincing sample (overfitting?)
qcl.set_input_state(x_train, uin)
Zprob_train = qcl.pred(theta_opt) # Update model parameter theta    
Zpred_train = np.argmax(Zprob_train, axis=1)


# Signal and Background efficiencies
if len(y_test_label) > 0 and len(y_test_label) == len(Zpred):
    n_sig = np.sum(y_test_label==1)
    n_bg = np.sum(y_test_label==0)
    n_sig_match = np.sum(y_test_label+Zpred==2)
    n_bg_match = np.sum(y_test_label+Zpred==0)
    print 'Sig eff=',1.*n_sig_match/n_sig,' BG eff=',1.*(n_bg-n_bg_match)/n_bg


prob_test_signal = Zprob[:,1]
fpr, tpr, thresholds = roc_curve(y_test_label, prob_test_signal, drop_intermediate=False)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Testing (area = %0.3f)' % roc_auc)
plt.plot([0, 0], [1, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig('ROC_SUSY_uin'+str(uin)+'_'+str(nqubit)+'d_'+str(nevt)+'evt_iter'+str(niter)+'_depth'+str(c_depth)+'_'+opt+'_run'+str(jobn)+'.pdf')

prob_train_signal = Zprob_train[:,1]
fpr_tr, tpr_tr, thresholds_tr = roc_curve(y_train_label, prob_train_signal, drop_intermediate=False)
roc_auc_tr = auc(fpr_tr, tpr_tr)
plt.plot(fpr_tr, tpr_tr, color='darkblue', lw=2, label='Training (area = %0.3f)' % roc_auc_tr)
plt.legend(loc='lower right')
plt.savefig('ROC2_SUSY_uin'+str(uin)+'_'+str(nqubit)+'d_'+str(nevt)+'evt_iter'+str(niter)+'_depth'+str(c_depth)+'_'+opt+'_run'+str(jobn)+'.pdf')

print ''
print 'AUC(training) =',roc_auc_tr
print 'AUC(testing) =',roc_auc
