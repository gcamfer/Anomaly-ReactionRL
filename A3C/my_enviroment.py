'''
Reinforcement learning Enviroment Definition
'''

import numpy as np
import tensorflow as tf
from data_preprocessing import data_cls

class my_env(data_cls):
    def __init__(self,train_test,**kwargs):
        data_cls.__init__(self,train_test,**kwargs)
        self.data_shape = self.get_shape()
        self.batch_size = kwargs.get('batch_size',1) # experience replay -> batch = 1
        self.iterations_episode = kwargs.get('iterations_episode',10)
        self.action_space = len(self.attack_types) # Number of posible actions
        self.observation_space = self.data_shape[1]-self.action_space
        self.counter = 0

    def _update_state(self):
        self.states,self.labels = self.get_batch(self.batch_size)
        
    '''
    Returns:
        + Observation of the enviroment
    '''
    def reset(self):
        self.state_numb = 0
        
        #self.states,self.labels = data_cls.get_sequential_batch(self,self.batch_size)
        self.states,self.labels = data_cls.get_batch(self,self.batch_size)
        
        
        return self.states
   
    '''
    Returns:
        State: Next state for the game
        Reward: Actual reward
        done: If the game ends (no end in this case)
    '''    
    def step(self,actions):
        # Clear previous rewards        
        self.reward = 0
        
        # Actualize new rewards == get_reward
        if actions == np.argmax(self.labels.values):
            self.reward = 1
        # Get new state and new true values
        self._update_state()
        
#        self.done = False
        if self.counter >= 100:
            self.done = True
            
        else:
            self.done = False
        self.counter += 1
            
        return self.states, self.reward, self.done
    
