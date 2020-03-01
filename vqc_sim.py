from qiskit import Aer
from qiskit.aqua import run_algorithm, QuantumInstance
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import SPSA, COBYLA
from qiskit.aqua.components.feature_maps import SecondOrderExpansion, FirstOrderExpansion
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.aqua.input import ClassificationInput

sfrom qiskit.tools.visualization import plot_histogram

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

import logging
from qiskit.aqua import set_qiskit_aqua_logging
set_qiskit_aqua_logging(logging.DEBUG)  # choose INFO, DEBUG to see the log

# class label, lepton 1 pT, lepton 1 eta, lepton 1 phi, lepton 2 pT, lepton 2 eta, lepton 2 phi, missing energy magnitude, missing energy phi, MET_rel, axial MET, M_R, M_TR_2, R, MT2, S_R, M_Delta_R, dPhi_r_b, cos(theta_r1)
df = pd.read_csv("Files/SUSY_10K.csv",names=('isSignal','lep1_pt','lep1_eta','lep1_phi','lep2_pt','lep2_eta','lep2_phi','miss_ene','miss_phi','MET_rel','axial_MET','M_R','M_TR_2','R','MT2','S_R','M_Delta_R','dPhi_r_b','cos_theta_r1'))


feature_dim = NQUBIT   # dimension of each data point
if feature_dim == 3:
    SelectedFeatures = ['lep1_pt', 'lep2_pt', 'miss_ene']
elif feature_dim == 5:
    SelectedFeatures = ['lep1_pt','lep2_pt','miss_ene','M_TR_2','M_Delta_R']
elif feature_dim == 7:
    SelectedFeatures = ['lep1_pt','lep1_eta','lep2_pt','lep2_eta','miss_ene','M_TR_2','M_Delta_R']

jobn = JOBN
training_size = NEVT
testing_size = NEVT
shots = 1024
uin_depth = NDEPTH_UIN
uvar_depth = NDEPTH_UVAR
niter = NITER
backend_name = 'BACKENDNAME'
option = 'OPTION'
random_seed = 10598+1010*uin_depth+101*uvar_depth+jobn

df_sig = df.loc[df.isSignal==1, SelectedFeatures]
df_bkg = df.loc[df.isSignal==0, SelectedFeatures]

df_sig_training = df_sig.values[:training_size]
df_bkg_training = df_bkg.values[:training_size]
df_sig_test = df_sig.values[training_size:training_size+testing_size]
df_bkg_test = df_bkg.values[training_size:training_size+testing_size]
training_input = {'1':df_sig_training, '0':df_bkg_training}
test_input = {'1':df_sig_test, '0':df_bkg_test}

datapoints, class_to_label = split_dataset_to_data_and_labels(test_input)


from qiskit import IBMQ
IBMQ.load_account()
print("Available backends:",IBMQ.providers())
provider0 = IBMQ.get_provider(project='icepp')
print("Backends for project 'icepp':",provider0.backends())

backend = provider0.get_backend(backend_name)
properties = backend.properties()
coupling_map = backend.configuration().coupling_map
print("coupling_map =",coupling_map)


simulator = Aer.get_backend('qasm_simulator')

optimizer = COBYLA(maxiter=niter, disp=True)
feature_map = FEATMAPExpansion(feature_dimension=feature_dim, depth=uin_depth)
var_form = RYRZ(num_qubits=feature_dim, depth=uvar_depth)
vqc = VQC(optimizer, feature_map, var_form, training_input, test_input)

quantum_instance = QuantumInstance(simulator, shots=shots, seed_simulator=random_seed, seed_transpiler=random_seed, skip_qobj_validation=True)
result = vqc.run(quantum_instance)

counts = vqc.get_optimal_vector()
print("Counts (w/o noise) =",counts)
plot = plot_histogram(counts, title='Bit counts (w/o noise)')
plot.savefig('BitCounts_sim_'+backend_name+'_'+str(feature_dim)+'d_'+str(training_size*2)+'evt_iter'+str(niter)+'_FMAPuin-depth'+str(uin_depth)+'_uvar-depth'+str(uvar_depth)+'_'+option+'.pdf')
plot.clf()


predicted_probs, predicted_labels = vqc.predict(datapoints[0])
predicted_classes = map_label_to_class_name(predicted_labels, vqc.label_to_class)

n_sig = np.sum(datapoints[1]==1)
n_bg = np.sum(datapoints[1]==0)
n_sig_match = np.sum(datapoints[1]+predicted_labels==2)
n_bg_match = np.sum(datapoints[1]+predicted_labels==0)

print(" --- Testing success ratio: ",result['testing_accuracy'],"(w/o noise)")
print(" ---   Signal eff =",n_sig_match/n_sig, ", Background eff =",(n_bg-n_bg_match)/n_bg, " (w/o noise)")


from sklearn.metrics import roc_curve, auc, roc_auc_score
prob_test_signal = predicted_probs[:,1]
fpr, tpr, thresholds = roc_curve(datapoints[1], prob_test_signal, drop_intermediate=False)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Testing w/o noise (area = %0.3f)' % roc_auc)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

datapoints_tr, class_to_label_tr = split_dataset_to_data_and_labels(training_input)
predicted_probs_tr, predicted_labels_tr = vqc.predict(datapoints_tr[0])
prob_train_signal = predicted_probs_tr[:,1]
fpr_tr, tpr_tr, thresholds_tr = roc_curve(datapoints_tr[1], prob_train_signal, drop_intermediate=False)
roc_auc_tr = auc(fpr_tr, tpr_tr)
plt.plot(fpr_tr, tpr_tr, color='darkblue', lw=2, label='Training w/o noise (area = %0.3f)' % roc_auc_tr)
plt.legend(loc="lower right")
plt.savefig('ROC2_sim_'+backend_name+'_'+str(feature_dim)+'d_'+str(training_size*2)+'evt_iter'+str(niter)+'_FMAPuin-depth'+str(uin_depth)+'_uvar-depth'+str(uvar_depth)+'_'+option+'.pdf')
plt.clf()
print(f'AUC(training) w/o noise = {roc_auc_tr:.3f}')
print(f'AUC(testing) w/o noise = {roc_auc:.3f}')
