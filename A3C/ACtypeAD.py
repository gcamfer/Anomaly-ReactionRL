'''
Type anomaly detection file
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
import scipy.signal


import sys
import os
from time import time
from time import sleep

import tensorflow as tf
import tensorflow.contrib.slim as slim

import threading
import multiprocessing

from data_preprocessing import data_cls
from my_enviroment import env

from estimators import ValueEstimator, PolicyEstimator
from policy_monitor import PolicyMonitor
from worker import Worker


tf.flags.DEFINE_string("model_dir", "/tmp/a3c", "Directory to write Tensorboard summaries and videos to.")
tf.flags.DEFINE_integer("t_max", 5, "Number of steps before performing an update")
tf.flags.DEFINE_integer("max_global_steps", None, "Stop training after this many steps in the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer("eval_every", 300, "Evaluate the policy every N seconds")
tf.flags.DEFINE_boolean("reset", False, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_integer("parallelism", None, "Number of threads to run. If not set we run [num_cpu_cores] threads.")

FLAGS = tf.flags.FLAGS

#def make_env(wrap=True):
#  env = gym.envs.make(FLAGS.env)
#  # remove the timelimitwrapper
#  env = env.env
#  if wrap:
#    env = atari_helpers.AtariEnvWrapper(env)
#  return env



if __name__ == "__main__":
  
    
    sess = tf.Session()
    
    
    kdd_10_path = '../../datasets/kddcup.data_10_percent_corrected'
    kdd_path = '../../datasets/kddcup.data'

    # Valid actions = '0' supose no attack, '1' supose attack
    epsilon = 1  # exploration

    minibatch_size = 2

    #3max_memory = 100
    decay_rate = 0.99
    gamma = 0.001
    
    
    hidden_size = 100
    hidden_layers = 3
    

    # Initialization of the enviroment
    env = RLenv(kdd_path,'train',join_path='../../datasets/corrected')
    
    iterations_episode = 100
    num_episodes = int(env.data_shape[0]/(iterations_episode)/10)
	
    valid_actions = list(range(len(env.attack_types))) # only detect type of attack
    num_actions = len(valid_actions)
    
    # Initialization of the Agent
    obs_size = env.data_shape[1]-len(env.attack_types)
    
    agent = actor_critic(valid_actions,obs_size,sess,
                          epoch_length = iterations_episode,
                          epsilon = epsilon,
                          decay_rate = decay_rate,
                          gamma = gamma,
                          hidden_size=hidden_size,
                          hidden_layers=hidden_layers,
                          minibatch_size=minibatch_size,
                          mem_size = 1000)    
    
    
    # Statistics
    reward_chain = []
    actor_loss_chain = []
    critic_loss_chain=[]

    
    # Main loop
    for epoch in range(num_episodes):
        start_time = time()
        t_critic_loss = 0.
        t_actor_loss = 0.
        total_reward_by_episode = 0
        # Reset enviromet, actualize the data batch
        states = env.reset()
        
        done = False
       

        # Iteration in one episode
        for i_iteration in range(iterations_episode):
            

            # Get actions for actual states following the policy
            actions_prob = agent.actor_model.predict(states)[0]
            actions = np.random.choice(np.arange(len(actions_prob)), p=actions_prob)
            #Enviroment actuation for this actions
            next_states, reward, done = env.act(actions)
            # If the epoch*batch_size*iterations_episode is largest than the df
            if next_states.shape[0] != 1:
                break # finished df
            
            agent.learn(states,actions,next_states,reward,done)
            
            # Train network, update loss after at least minibatch_learns
            if epoch*iterations_episode + i_iteration >= minibatch_size:
                critic_loss = agent.update()
                t_critic_loss += critic_loss
                #t_actor_loss += actor_loss
            update_end_time = time()

            # Update the state
            states = next_states
            
            
            # Update statistics
            total_reward_by_episode += reward
        
        if next_states.shape[0] != 1:
                break # finished df
        # Update user view
        reward_chain.append(total_reward_by_episode)    
       # actor_loss_chain.append(t_actor_loss)
        critic_loss_chain.append(t_critic_loss)
        
        end_time = time()
        print("\r|Epoch {:03d}/{:03d} | Actor Loss {:4.4f} |Critic Loss {:4.4f} |" 
                "Tot reward in ep {:03d}| time: {:2.2f}|"
                .format(epoch, num_episodes 
                ,t_actor_loss,t_critic_loss, total_reward_by_episode,(end_time-start_time)))
        print("\r|Estimated: {}|Labels: {}".format(env.estimated_labels,env.true_labels))
        
    # Save trained model weights and architecture, used in test
    agent.model_network.model.save_weights("models/type_model.h5", overwrite=True)
    with open("models/type_model.json", "w") as outfile:
        json.dump(agent.model_network.model.to_json(), outfile)
        
    # Save test dataset deleting the data used to train
    print("Shape: ",env.data_shape)
    print("Used: ",num_episodes*iterations_episode)
    #env.save_test()
    
#    # Plot training results
#    plt.figure(1)
#    plt.subplot(211)
#    plt.plot(np.arange(len(reward_chain)),reward_chain)
#    plt.title('Total reward by episode')
#    plt.xlabel('n Episode')
#    plt.ylabel('Total reward')
#    
#    plt.subplot(212)
#    plt.plot(np.arange(len(loss_chain)),loss_chain)
#    plt.title('Loss by episode')
#    plt.xlabel('n Episode')
#    plt.ylabel('loss')
#    plt.tight_layout()
#    #plt.show()
#    plt.savefig('results/train_type.eps', format='eps', dpi=1000)




