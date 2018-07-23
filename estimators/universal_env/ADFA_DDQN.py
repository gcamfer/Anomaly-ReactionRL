'''
Type anomaly detection file
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras import optimizers
import keras.backend as K
import json
import sys
import time

from network_classification import NetworkClassificationEnv




def huber_loss(y_true, y_pred, clip_value=1):
    assert clip_value > 0.

    x = y_true - y_pred
    if np.isinf(clip_value):
        # Spacial case for infinity since Tensorflow does have problems
        # if we compare `K.abs(x) < np.inf`.
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        if hasattr(tf, 'select'):
            return tf.select(condition, squared_loss, linear_loss)
        else:
            return tf.where(condition, squared_loss, linear_loss)
    elif K.backend() == 'theano':
        from theano import tensor as T
        return T.switch(condition, squared_loss, linear_loss)
    else:
        raise RuntimeError('Unknown backend "{}".'.format(K.backend()))

import keras.losses
keras.losses.huber_loss = huber_loss



class QNetwork():
    """
    Q-Network Estimator
    Represents the global model for the table
    """
    


    def __init__(self,obs_size,num_actions,hidden_size = 100,
                 hidden_layers = 1,learning_rate=.02):
        """
        Initialize the network with the provided shape
        """
        # Network arquitecture
        self.model = Sequential()
        # Add imput layer
        self.model.add(Dense(hidden_size, input_shape=(obs_size,),
                             activation='relu'))
        # Add hidden layers
        for layers in range(hidden_layers):
            self.model.add(Dense(hidden_size, activation='relu'))
        # Add output layer    
        self.model.add(Dense(num_actions))
        
#        optimizer = optimizers.SGD(learning_rate)
        optimizer = optimizers.Adam(0.0003)
        # optimizer = optimizers.AdaGrad(learning_rate)
        # optimizer = optimizers.RMSpropGraves(learning_rate, 0.95, self.momentum, 1e-2)
        
        # Compilation of the model with optimizer and loss
        self.model.compile(loss=huber_loss,optimizer=optimizer)

    def predict(self,state,batch_size=1):
        """
        Predicts action values.
        """
        return self.model.predict(state,batch_size=batch_size)

    def update(self, states, q):
        """
        Updates the estimator with the targets.

        Args:
          states: Target states
          q: Estimated values

        Returns:
          The calculated loss on the batch.
        """
        loss = self.model.train_on_batch(states, q)
        return loss

    def copy_model(model):
        """Returns a copy of a keras model."""
        model.save('tmp_model')
        return keras.models.load_model('tmp_model')





#Policy interface
class Policy:
    def __init__(self, num_actions, estimator):
        self.num_actions = num_actions
        self.estimator = estimator
    
class Epsilon_greedy(Policy):
    def __init__(self,estimator ,num_actions,epsilon,decay_rate, epoch_length):
        Policy.__init__(self, num_actions, estimator)
        self.name = "Epsilon Greedy"
        if (epsilon is None or epsilon < 0 or epsilon > 1):
            print("EpsilonGreedy: Invalid value of epsilon", flush = True)
            sys.exit(0)
        self.epsilon = epsilon
        self.step_counter = 0
        self.epoch_length = epoch_length
        self.decay_rate = decay_rate
        
#        # if epsilon set to 1, it will be decayed over time
#        if self.epsilon == 1:
#            self.epsilon_decay = True
#        else:
#            self.epsilon_decay = False
        # Always decay
        self.epsilon_decay = True
        
    
    def get_actions(self,states):
        # get next action
        if np.random.rand() <= self.epsilon:
            actions = np.random.randint(0, self.num_actions,states.shape[0])
        else:
            self.Q = self.estimator.predict(states,states.shape[0])
            # TODO: fix performance in this loop
            actions = []
            for row in range(self.Q.shape[0]):
                best_actions = np.argwhere(self.Q[row] == np.amax(self.Q[row]))
                actions.append(best_actions[np.random.choice(len(best_actions))].item())
            
        self.step_counter += 1 
        # decay epsilon after each epoch
        if self.epsilon_decay:
            if self.step_counter % self.epoch_length == 0:
                self.epsilon = max(.01, self.epsilon * self.decay_rate**self.step_counter)
            
        return actions
    



'''
Reinforcement learning Agent definition
'''
class Agent(object):   
    def __init__(self, actions,obs_size, policy="EpsilonGreedy", **kwargs):
        self.actions = actions
        self.num_actions = len(actions)
        self.obs_size = obs_size
        
        self.epsilon = kwargs.get('epsilon', 1)
        self.gamma = kwargs.get('gamma', .001)
        self.minibatch_size = kwargs.get('minibatch_size', 2)
        self.epoch_length = kwargs.get('epoch_length', 100)
        self.decay_rate = kwargs.get('decay_rate',0.99)
        self.ExpRep = kwargs.get('ExpRep',True)
        if self.ExpRep:
            self.memory = ReplayMemory(self.obs_size, kwargs.get('mem_size', 10))
        
        self.ddqn_time = 100
        self.ddqn_update = self.ddqn_time

        
        self.model_network = QNetwork(self.obs_size, self.num_actions,
                                      kwargs.get('hidden_size', 100),
                                      kwargs.get('hidden_layers',1),
                                      kwargs.get('learning_rate',.1))
        
        self.target_model_network = QNetwork(self.obs_size, self.num_actions,
                                      kwargs.get('hidden_size', 100),
                                      kwargs.get('hidden_layers',1),
                                      kwargs.get('learning_rate',.1))
        self.target_model_network.model = QNetwork.copy_model(self.model_network.model)
        
        if policy == "EpsilonGreedy":
            self.policy = Epsilon_greedy(self.model_network,len(actions),
                                         self.epsilon,self.decay_rate,
                                         self.epoch_length)
        
    def act(self,states):
        # Get actions under the policy
        actions = self.policy.get_actions(states)
        return actions
    
    def learn(self, states, actions,next_states, rewards, done):
        if self.ExpRep:
            self.memory.observe(states, actions, rewards, done)
        else:
            self.states = states
            self.actions = actions
            self.next_states = next_states
            self.rewards = rewards
            self.done = done


    def update_model(self):
        if self.ExpRep:
            (states, actions, rewards, next_states, done) = self.memory.sample_minibatch(self.minibatch_size)
        else:
            states = self.states
            rewards = self.rewards
            next_states = self.next_states
            actions = self.actions
            done = self.done
            
        next_actions = []
        # Compute Q targets
        Q_prime = self.model_network.predict(next_states,self.minibatch_size)
        # TODO: fix performance in this loop
        for row in range(Q_prime.shape[0]):
            best_next_actions = np.argwhere(Q_prime[row] == np.amax(Q_prime[row]))
            next_actions.append(best_next_actions[np.random.choice(len(best_next_actions))].item())
        sx = np.arange(len(next_actions))
        # Compute Q(s,a)
        Q = self.target_model_network.predict(states,self.minibatch_size)
        # Q-learning update
        # target = reward + gamma * max_a'{Q(next_state,next_action))}
        targets = rewards.reshape(Q[sx,actions].shape) + \
                  self.gamma * Q_prime[sx,next_actions] * \
                  (1-done.reshape(Q[sx,actions].shape))   
        Q[sx,actions] = targets  
        
        loss = self.model_network.model.train_on_batch(states,Q)#inputs,targets  
        
        # timer to ddqn update
        self.ddqn_update -= 1
        if self.ddqn_update == 0:
            self.ddqn_update = self.ddqn_time
#            self.target_model_network.model = QNetwork.copy_model(self.model_network.model)
            self.target_model_network.model.set_weights(self.model_network.model.get_weights()) 
            
        
        return loss    
        
      
        
    
class ReplayMemory(object):
    """Implements basic replay memory"""

    def __init__(self, observation_size, max_size):
        self.observation_size = observation_size
        self.num_observed = 0
        self.max_size = max_size
        self.samples = {
                 'obs'      : np.zeros(self.max_size * 1 * self.observation_size,
                                       dtype=np.float32).reshape(self.max_size, self.observation_size),
                 'action'   : np.zeros(self.max_size * 1, dtype=np.int16).reshape(self.max_size, 1),
                 'reward'   : np.zeros(self.max_size * 1).reshape(self.max_size, 1),
                 'terminal' : np.zeros(self.max_size * 1, dtype=np.int16).reshape(self.max_size, 1),
               }

    def observe(self, state, action, reward, done):
        index = self.num_observed % self.max_size
        self.samples['obs'][index, :] = state
        self.samples['action'][index, :] = action
        self.samples['reward'][index, :] = reward
        self.samples['terminal'][index, :] = done

        self.num_observed += 1

    def sample_minibatch(self, minibatch_size):
        max_index = min(self.num_observed, self.max_size) - 1
        sampled_indices = np.random.randint(max_index, size=minibatch_size)

        s      = np.asarray(self.samples['obs'][sampled_indices, :], dtype=np.float32)
        s_next = np.asarray(self.samples['obs'][sampled_indices+1, :], dtype=np.float32)

        a      = self.samples['action'][sampled_indices].reshape(minibatch_size)
        r      = self.samples['reward'][sampled_indices].reshape((minibatch_size, 1))
        done   = self.samples['terminal'][sampled_indices].reshape((minibatch_size, 1))

        return (s, a, r, s_next, done)

    




if __name__ == "__main__":
  
    train_path = '../../datasets/ADFA/UNSW_NB15_training-set.csv'
    test_path = '../../datasets/ADFA/UNSW_NB15_testing-set.csv'

    formated_train_path = "../../datasets/formated/formated_train_ADFA.data"
    formated_test_path = "../../datasets/formated/formated_test_ADFA.data"

    model_path = "models/typeAD_tf"
    
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
    
    column_names = ['id_drop','dur','proto','service_logarithm','state_logarithm',
                    'spkts_logarithm','dpkts_logarithm','sbytes_logarithm',
                    'dbytes_logarithm','rate_logarithm','sttl_logarithm',
                    'dttl_logarithm','sload_logarithm','dload_logarithm',
                    'sloss_logarithm','dloss_logarithm','sinpkt_logarithm',
                    'dinpkt_logarithm','sjit_logarithm','djit_logarithm',
                    'swin_logarithm','stcpb_logarithm','dtcpb_logarithm',
                    'dwin_logarithm','tcprtt_logarithm','synack','ackdat','smean',
                    'dmean','trans_depth','response_body_len','ct_srv_src',
                    'ct_state_ttl','ct_dst_ltm','ct_src_dport_ltm',
                    'ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login',
                    'ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm','ct_srv_dst',
                    'is_sm_ips_ports','labels','labels_drop']
    
    ##########################################################################
    



    # Valid actions = '0' supose no attack, '1' supose attack
    epsilon = 1  # exploration
    
    # Train batch
    batch_size = 1
    # batch of memory ExpRep
    minibatch_size = 100
    ExpRep = True
    
    iterations_episode = 100

    #3max_memory = 100
    decay_rate = 0.99
    gamma = 0.001
    
    
    hidden_size = 100
    hidden_layers = 3
    
    
    # Initialization of the enviroment
    env = NetworkClassificationEnv(
            'train',
            attack_map,
            column_names=column_names,
            train_path=train_path,test_path=test_path,
            formated_train_path = formated_train_path,
            formated_test_path = formated_test_path,
            batch_size = batch_size,
            iterations_episode = iterations_episode
            )


#    num_episodes = int(env.data_shape[0]/(iterations_episode)/10)
    num_episodes = 200
    valid_actions = list(range(len(env.attack_types))) # only detect type of attack
    num_actions = len(valid_actions)
    
    # Initialization of the Agent
    obs_size = env.observation_len
    
    agent = Agent(valid_actions,obs_size,"EpsilonGreedy",
                          epoch_length = iterations_episode,
                          epsilon = epsilon,
                          decay_rate = decay_rate,
                          gamma = gamma,
                          hidden_size=hidden_size,
                          hidden_layers=hidden_layers,
                          minibatch_size=minibatch_size,
                          mem_size = 10000,ExpRep=ExpRep)    
    
    
    # Statistics
    reward_chain = []
    loss_chain = []
    

    
    # Main loop
    for epoch in range(num_episodes):
        start_time = time.time()
        loss = 0.
        total_reward_by_episode = 0
        # Reset enviromet, actualize the data batch
        states = env.reset()
        
        done = False
       
        true_labels = np.zeros(len(env.attack_types))
        estimated_labels = np.zeros(len(env.attack_types))
        # Iteration in one episode
        for i_iteration in range(iterations_episode):
            

            # Get actions for actual states following the policy
            actions = agent.act(states)
            
            # Update dialog
            estimated_labels[actions] += 1
            true_labels[env.labels] += 1
            
            
            #Enviroment actuation for this actions
            next_states, reward, done = env.step(actions)
            # If the epoch*batch_size*iterations_episode is largest than the df

            agent.learn(states,actions,next_states,reward,done)
            
            # Train network, update loss after at least minibatch_learns
            if ExpRep and epoch*iterations_episode + i_iteration >= minibatch_size:
                loss += agent.update_model()
            elif not ExpRep:
                loss += agent.update_model()
            
            update_end_time = time.time()

            # Update the state
            states = next_states
            
            
            # Update statistics
            total_reward_by_episode += np.sum(reward,dtype=np.int32)

        # Update user view
        reward_chain.append(total_reward_by_episode)    
        loss_chain.append(loss) 
        
        
        end_time = time.time()
        print("\r|Epoch {:03d}/{:03d} | Loss {:4.4f} |" 
                "Tot reward in ep {:03d}| time: {:2.2f}|"
                .format(epoch, num_episodes 
                ,loss, total_reward_by_episode,(end_time-start_time)))
        print("\r|Estimated: {}|Labels: {}".format(estimated_labels,true_labels))
        
    # Save trained model weights and architecture, used in test
    agent.model_network.model.save_weights("models/ADFA_DDQN.h5", overwrite=True)
    with open("models/ADFA_DDQN.json", "w") as outfile:
        json.dump(agent.model_network.model.to_json(), outfile)
        
    
    # Plot training results
    plt.figure(1)
    plt.subplot(211)
    plt.plot(np.arange(len(reward_chain)),reward_chain)
    plt.title('Total reward by episode')
    plt.xlabel('n Episode')
    plt.ylabel('Total reward')
    
    plt.subplot(212)
    plt.plot(np.arange(len(loss_chain)),loss_chain)
    plt.title('Loss by episode')
    plt.xlabel('n Episode')
    plt.ylabel('loss')
    plt.tight_layout()
    #plt.show()
    plt.savefig('results/train_type_improved.eps', format='eps', dpi=1000)




