'''
Reinforcement learning Enviroment Definition
'''

import numpy as np
import tensorflow as tf


class env(data_cls):
    def __init__(self,path,train_test,**kwargs):
        data_cls.__init__(self,path,train_test,**kwargs)
        self.data_shape = data_cls.get_shape(self)
        self.batch_size = 1 # experience replay -> batch = 1

    def _update_state(self):
        self.states,self.labels = data_cls.get_batch(self,self.batch_size)
        
        # Update statistics
        self.true_labels += np.sum(self.labels).values

    '''
    Returns:
        + Observation of the enviroment
    '''
    def reset(self):
        # Statistics
        self.true_labels = np.zeros(len(env.attack_types),dtype=int)
        self.estimated_labels = np.zeros(len(env.attack_types),dtype=int)
        
        self.state_numb = 0
        
        #self.states,self.labels = data_cls.get_sequential_batch(self,self.batch_size)
        self.states,self.labels = data_cls.get_batch(self,self.batch_size)
        
        # Update statistics
        self.true_labels += np.sum(self.labels).values
        
        self.total_reward = 0
        self.steps_in_episode = 0
        return self.states.values 
   
    '''
    Returns:
        State: Next state for the game
        Reward: Actual reward
        done: If the game ends (no end in this case)
    '''    
    def act(self,actions):
        # Clear previous rewards        
        self.reward = 0
        
        # Actualize new rewards == get_reward
        if actions == np.argmax(self.labels.values):
            self.reward = 1
        self.estimated_labels[actions] +=1
        # Get new state and new true values
        self._update_state()
        
        # Done allways false in this continuous task       
        self.done = False
            
        return self.states, self.reward, self.done
    
