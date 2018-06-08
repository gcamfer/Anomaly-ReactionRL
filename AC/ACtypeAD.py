'''
Type anomaly detection file
'''

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections


import itertools

from my_enviroment import my_env

from estimators import ValueEstimator, PolicyEstimator




# Initialization of the enviroment
def make_env():
    kdd_train = '../../datasets/NSL/KDDTrain+.txt'
    kdd_test = '../../datasets/NSL/KDDTest+.txt'
    
    formated_train_path = "../../datasets/formated/formated_train_type.data"
    formated_test_path = "../../datasets/formated/formated_test_type.data"
    batch_size = 1
    iterations_episode = 100
    
    env = my_env('train',train_path=kdd_train,test_path=kdd_test,
                formated_train_path = formated_train_path,
                formated_test_path = formated_test_path,
                batch_size=batch_size,
                iterations_episode=iterations_episode)
    return env

env = make_env()

VALID_ACTIONS = list(range(env.action_space))

num_episodes = 50

gamma = 0.99

batch_size = 1

    
policy_estimator = PolicyEstimator(num_outputs=len(VALID_ACTIONS),
                             observation_space=env.observation_space,reuse=True)
value_estimator = ValueEstimator(observation_space=env.observation_space,
                           reuse=True)
        
    # Global step iterator
global_counter = itertools.count()


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    
#    # Load a previous checkpoint if it exists
#    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
#    if latest_checkpoint:
#        print("Loading model checkpoint: {}".format(latest_checkpoint))
#        saver.restore(sess, latest_checkpoint)
#        
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    
    episode_rewards = np.zeros(num_episodes)

    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()
        
        episode = []
        
        # One step in the environment
        for t in itertools.count():
            
            # Take a step
            action_probs = policy_estimator.predict(state.reshape([batch_size,len(state)]))[0]
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done = env.step(action)
            
            # Keep track of the transition
            episode.append(Transition(
              state=state, action=action, reward=reward, next_state=next_state, done=done))
            
            # Update statistics
            episode_rewards[i_episode] += reward
            
            # Calculate TD Target
            value_next = value_estimator.predict(next_state.reshape([batch_size,len(next_state)]))
            td_target = reward + gamma * value_next
            td_error = td_target - value_estimator.predict(state.reshape([batch_size,len(state)]))
            
            # Update the value estimator
            value_estimator.update(state.reshape([batch_size,len(state)]), td_target)
            
            # Update the policy estimator
            # using the td error as our advantage estimate
            policy_estimator.update(state.reshape([batch_size,len(state)]), td_error, action)
            
            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                    t, i_episode + 1, num_episodes, episode_rewards[i_episode - 1]), end="")

            if done:
                break
                
            state = next_state

    # Save the variables to disk.
    meta_graph_def = tf.train.export_meta_graph(filename='/tmp/my-model.meta')
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)


    





    
    
    
