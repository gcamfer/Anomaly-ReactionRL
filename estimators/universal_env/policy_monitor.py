import sys
import os

import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pandas as pd



import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import  confusion_matrix

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
  sys.path.append(import_path)


from estimators import PolicyEstimator
from worker import make_copy_params_op

from network_classification import NetworkClassificationEnv


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




class PolicyMonitor(object):
    """
    Helps evaluating a policy by running an episode in an environment,
    saving a video, and plotting summaries to Tensorboard.

    Args:
        env: environment to run in
        policy_net: A policy estimator
        summary_writer: a tf.train.SummaryWriter used to write Tensorboard summaries
    """
    def __init__(self, env, policy_net, summary_writer, saver=None):


        self.env = env
        self.global_policy_net = policy_net
        self.summary_writer = summary_writer
        self.saver = saver
        self.counter = 0
    
        self.checkpoint_path = os.path.abspath(os.path.join(summary_writer.get_logdir(), "../checkpoints/model"))
    
    
        # Local policy net
        with tf.variable_scope("policy_eval"):
          self.policy_net = PolicyEstimator(policy_net.num_outputs,policy_net.observation_space)
    
        # Op to copy params from global policy/value net parameters
        self.copy_params_op = make_copy_params_op(
          tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
          tf.contrib.slim.get_variables(scope="policy_eval", collection=tf.GraphKeys.TRAINABLE_VARIABLES))

    def _policy_net_predict(self, state, sess):
        feed_dict = { self.policy_net.states: state }
        preds = sess.run(self.policy_net.predictions, feed_dict)
        return preds["probs"]
    

    def eval_once(self, sess):
        with sess.as_default(), sess.graph.as_default():
            # Copy params to local model
            global_step, _ = sess.run([tf.train.get_global_step(), self.copy_params_op])


        # Run an episode
        done = False
        state = self.env.reset()
        total_reward = 0.0
        episode_length = 0
        while not done:
            action_probs = self._policy_net_predict(state, sess)
            action = np.random.choice(np.arange(len(action_probs[0])), p=action_probs[0])
            next_state, reward, done = self.env.step(action)
            # # # # # #        
            total_reward += reward
            episode_length += 1
            state = next_state
            
            
         # Run the test evaluation
        total_reward = 0
    
        estimated_labels = np.zeros(len(self.env.attack_types),dtype=int)
        estimated_correct_labels = np.zeros(len(self.env.attack_types),dtype=int)
        true_labels = np.zeros(len(self.env.attack_types),dtype=int)
        
        
        formated_test_path = "../../datasets/formated/formated_test_ADFA.data"
        
        ########################################################################
        # Aguments needed by enviroment:
        
        # Map from attack to type
        attack_map = {'Normal': 'Normal',
                      'Generic': 'Generic',
                      'Exploits': 'Exploits',
                      'Fuzzers':'Fuzzers',
                      'DoS':'DoS',
                      'Reconnaissance':'Reconnaissance',
                      'Analysis':'Analysis',
                      'Backdoor':'Backdoor',
                      'Shellcode':'Shellcode',
                      'Worms':'Worms'
                    }
        

        
        ##########################################################################
        
        env = NetworkClassificationEnv('test',
            attack_map,
            formated_test_path = formated_test_path
            )
        
        states , labels = env.get_full()
        
        #speed up
#        states,labels = states[1:int(len(labels)/10)],labels[1:int(len(labels)/10)]
        
        
        action_probs = self._policy_net_predict(states, sess)


        all_actions=np.array([])
        # TODO: fix performance in this loop
        for indx in range(len(action_probs)):
            all_actions = np.append(all_actions,np.random.choice(np.arange(len(action_probs[indx])), p=action_probs[indx]))
            print("\rEpoch {}/{}".format(indx,len(action_probs)), end="")
 
        labs,estimated_v = np.unique(all_actions,return_counts=True)
        estimated_labels[labs.astype(np.int32)] = estimated_v
        total_reward = np.sum(all_actions==labels)
        equal = np.where(all_actions==labels)
        labs, correct = np.unique(labels[equal],return_counts=True)
        estimated_correct_labels[labs.astype(np.int32)] = correct
       
        
        labs,true_lab_v = np.unique(labels,return_counts=True)
        true_labels[labs.astype(np.int32)] = true_lab_v
        
        
        Accuracy = estimated_correct_labels / true_labels
        Mismatch = estimated_labels - true_labels
    
        print('\r\nTotal reward: {} | Number of samples: {} | Accuracy = {:.2f}%'.format(total_reward,
              len(states),float(100*total_reward/len(states))))
        outputs_df = pd.DataFrame(index = self.env.attack_types,columns = ["Estimated","Correct","Total","Acuracy"])
        for indx,att in enumerate(self.env.attack_types):
           outputs_df.iloc[indx].Estimated = estimated_labels[indx]
           outputs_df.iloc[indx].Correct = estimated_correct_labels[indx]
           outputs_df.iloc[indx].Total = true_labels[indx]
           outputs_df.iloc[indx].Acuracy = Accuracy[indx]*100
           outputs_df.iloc[indx].Mismatch = abs(Mismatch[indx])
    

    
        print(outputs_df)
    
        fig, ax = plt.subplots()
        width = 0.35
        pos = np.arange(len(true_labels))
        plt.bar(pos, estimated_correct_labels,width,color='g')
        plt.bar(pos+width,
                 (np.abs(estimated_correct_labels-true_labels)),width,
                 color='r')
        plt.bar(pos+width,np.abs(estimated_labels-estimated_correct_labels),width,
                 bottom=(np.abs(estimated_correct_labels-true_labels)),
                 color='b')
    
        
        ax.set_xticks(pos+width/2)
        ax.set_xticklabels(self.env.attack_types,rotation='vertical')
        #ax.set_yscale('log')
    
        #ax.set_ylim([0, 100])
        ax.set_title('Test set scores: Acc = {:.2f}'.format(100*total_reward/len(states)))
        plt.legend(('Correct estimated','False negative','False positive'))
        plt.tight_layout()
        #plt.show()
        
        RESULTS_DIR = "results/"
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        
        plt.savefig('results/A3C_test_type_{}.svg'.format(self.counter), format='svg', dpi=1000)
        self.counter += 1
        
        
        
        #%% Agregated precision

        aggregated_data_test = labels
        
        print('Performance measures on Test data')
        print('Accuracy =  {:.2f}'.format(accuracy_score( aggregated_data_test,all_actions)))
        print('F1 =  {:.2f}'.format(f1_score(aggregated_data_test,all_actions, average='weighted')))
        print('Precision_score =  {:.2f}'.format(precision_score(aggregated_data_test,all_actions, average='weighted')))
        print('recall_score =  {:.2f}'.format(recall_score(aggregated_data_test,all_actions, average='weighted')))
        
        cnf_matrix = confusion_matrix(aggregated_data_test,all_actions)
        np.set_printoptions(precision=2)
        plt.figure()
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=self.env.attack_types, normalize=True,
                              title='Normalized confusion matrix')
        plt.savefig('results/confusion_matrix_A3C_{}.svg'.format(self.counter), format='svg', dpi=1000)
        
        
        
        
        
        
        
        # Add summaries
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=total_reward, tag="eval/total_reward")
        episode_summary.value.add(simple_value=episode_length, tag="eval/episode_length")
        episode_summary.value.add(simple_value=float(100*total_reward/len(states)), tag="eval/test_accuracy")
        self.summary_writer.add_summary(episode_summary, global_step)
        self.summary_writer.flush()

        if self.saver is not None:
            self.saver.save(sess, self.checkpoint_path)
            
    

        tf.logging.info("Eval results at step {}: total_reward {}, Accuracy {:.2f},episode_length {}".format(global_step,
                        total_reward, float(100*total_reward/len(states)) ,episode_length))
    
        return total_reward, episode_length

    def continuous_eval(self, eval_every, sess, coord):
        """
        Continuously evaluates the policy every [eval_every] seconds.
        """
        try:
            while not coord.should_stop():
                self.eval_once(sess)
                # Sleep until next evaluation cycle
                time.sleep(eval_every)

        except tf.errors.CancelledError:
            return
