# QML_HEP

The python codes and relevant files used in [arXiv:2002.09935](https://arxiv.org/abs/2002.09935) are provided here.

## QCL algorithm with Qulacs simulator:
The [Qulacs](https://github.com/qulacs/qulacs) simulator has to be first installed. The original implementation of the QCL algorithm is taken from [Quantum Native Dojo](https://github.com/qulacs/quantum-native-dojo) (in Japanese) and then modified for this study.

- run_qulacs_loop.sh : shell script to run Qulacs python jobs
- qulacs_qcl.py : main python script
- qcl_classification.py : QCL implementation
- qcl_utils.py : utilities used in QCL

The qcl_classification.py and qcl_utils.py in Quantum Native Dojo are licensed under the BSD 3-Clause "New" or "Revised" License. 

## VQC algorithm with IBM Qiskit:
The IBM [Qiskit](https://github.com/Qiskit/qiskit) framework has to be first installed. The VQC codes used for IBM Q quantum machine and QASM simulator are provided separately.

- run_vqc_qc_loop.sh : shell script to run VQC on IBM Q machine
- vqc_qc.py : main python script to run on IBM Q machine
- run_vqc_sim_loop.sh : shell script to run VQC on QASM simulator
- vqc_sim.py : main python script to run on QASM simulator
