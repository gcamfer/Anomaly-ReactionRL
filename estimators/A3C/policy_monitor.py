import sys
import os

import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score


from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
  sys.path.append(import_path)


from estimators import PolicyEstimator
from worker import make_copy_params_op

from my_enviroment import my_env



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
            action_probs = self._policy_net_predict([state], sess)
            action = np.random.choice(np.arange(len(action_probs[0])), p=action_probs[0])
            next_state, reward, done = self.env.step(action)
            # # # # # #        
            total_reward += reward
            episode_length += 1
            state = next_state
            
            
            
        
        # test
        formated_test_path = "../../datasets/formated/formated_test_type.data"
        
        #TEST
        env = my_env('test',formated_test_path = formated_test_path) 
        total_reward = 0    
    
        true_labels = np.zeros(len(env.attack_types),dtype=int)
        estimated_labels = np.zeros(len(env.attack_types),dtype=int)
        estimated_correct_labels = np.zeros(len(env.attack_types),dtype=int)
        
        
        states , labels = env.get_full()
        
        true_labels = np.sum(labels,0)
        
            
        action_probs = self._policy_net_predict(states, sess)
        
        estimated_actions = np.zeros([len(labels),len(env.attack_types)])
        estimated_actions = np.zeros([len(labels),len(env.attack_types)])
        for indx in range(len(action_probs)):
            action = np.random.choice(np.arange(len(action_probs[indx])), p=action_probs[indx])
            estimated_labels[action] +=1
            estimated_actions[indx,action] = 1
            if action == np.argmax(labels.iloc[indx].values):
                total_reward += 1
                estimated_correct_labels[action] += 1
        
        normal_f1_score = f1_score(labels['normal'],estimated_actions[:,0])
        dos_f1_score = f1_score(labels['DoS'],estimated_actions[:,1])
        probe_f1_score = f1_score(labels['Probe'],estimated_actions[:,2])
        r2l_f1_score = f1_score(labels['R2L'],estimated_actions[:,3])
        u2r_f1_score = f1_score(labels['U2R'],estimated_actions[:,4])
            
        Accuracy = [normal_f1_score,dos_f1_score,probe_f1_score,r2l_f1_score,u2r_f1_score]
        
        Mismatch = abs(estimated_correct_labels - true_labels)+abs(estimated_labels-estimated_correct_labels)
    
        print('\r\nTotal reward: {} | Number of samples: {} | Accuracy = {}%'.format(total_reward,
              len(states),float(100*total_reward/len(states))))
        outputs_df = pd.DataFrame(index = env.attack_types,columns = ["Estimated","Correct","Total","F1_Score"])
        for indx,att in enumerate(env.attack_types):
           outputs_df.iloc[indx].Estimated = estimated_labels[indx]
           outputs_df.iloc[indx].Correct = estimated_correct_labels[indx]
           outputs_df.iloc[indx].Total = true_labels[indx]
           outputs_df.iloc[indx].F1_Score = Accuracy[indx]*100
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
        ax.set_xticklabels(env.attack_types,rotation='vertical')
        #ax.set_yscale('log')
    
        #ax.set_ylim([0, 100])
        ax.set_title('Test set scores: Acc = {:.2f}'.format(100*total_reward/len(states)))
        plt.legend(('Correct estimated','False negative','False positive'))
        plt.tight_layout()
        #plt.show()
        
        RESULTS_DIR = "results/"
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        
        plt.savefig('results/A3C_test_type_{}.eps'.format(self.counter), format='eps', dpi=1000)
        self.counter += 1
        
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
