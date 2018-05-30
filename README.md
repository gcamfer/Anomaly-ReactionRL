# AnomalyDetectionRL


## Overview
Using Reinforcement Learning in order to detect anomalies and maybe a future response
The dataset used is NSL-KDD with data of multiple anomalies

Using deep Q-Learning with keras/tensorflow to generate the network
## Simple anomaly detection
- Detects normal or anomaly
- Train set in: [AD.py](https://github.com/gcamfer/Anomaly-ReactionRL/blob/master/AD.py)
- Test set in: [test.py](https://github.com/gcamfer/Anomaly-ReactionRL/blob/master/test.py)

## Multiple anomaly detection (39+1 labels)
- Detects each attack in the dataset
- Train set in: [multiAD.py](https://github.com/gcamfer/Anomaly-ReactionRL/blob/master/multiAD.py)
- Test set in: [multi_test.py](https://github.com/gcamfer/Anomaly-ReactionRL/blob/master/multi_test.py)

## Type anomaly detection (4+1 labels)
- Detects only the attack type between normal, DoS, Probe, R2L, U2R
- Train set in: [typeAD.py](https://github.com/gcamfer/Anomaly-ReactionRL/blob/master/typeAD.py)
- Test set in: [type_test.py](https://github.com/gcamfer/Anomaly-ReactionRL/blob/master/type_test.py)

- Train Dueling DDQN (tensorflow) in [typeAD_tf.py](https://github.com/gcamfer/Anomaly-ReactionRL/blob/master/typeAD_tf.py)

## Adversarial/Multi Agent RL
- Try to improve the inequality of attacks to produce better training
- Train set in: [adversarialAD.py](https://github.com/gcamfer/Anomaly-ReactionRL/blob/master/adversarialAD.py)
- Test set in: [adversarial_test.py](https://github.com/gcamfer/Anomaly-ReactionRL/blob/master/adversarial_test.py)
