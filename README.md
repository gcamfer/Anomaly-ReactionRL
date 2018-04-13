# AnomalyDetectionRL

Using Reinforcement Learning in order to detect anomalies.

The dataset used is NSL-KDD with data of multiple anomalies

Using deep Q-Learning with keras to generate the network

AD.py presents the simple anomaly detection

multiAD.py presents the multiple anomaly detection selecting one of all available atacks in the dataset

tipeAD.py selects only the attack type between normal, DoS, Probe, R2L, U2R.

adversarialAD.py pretends to use the beneficts of adversarial to send most optimal attacks and train a better defender
