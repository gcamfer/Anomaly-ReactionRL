'''
Reinforcement learning Enviroment Definition for Network anomaly detection
'''
import logging.config
import gym
from gym import error, spaces, utils
from gym.utils import seeding
#from helpers_data_preprocessing import data_cls
from gym_network_classification_env.envs.helpers_data_preprocessing import data_cls
class NetworkClassificationEnv(gym.Env,data_cls):
    '''
        Environment class definition. It represents the dynamics of the AD
        Args needed by data_cls:
            col_names: list of all parameters in the dataset. Must contain 'label'
            train_path: Relative path in wich the train dataset is located
            test_path: Relative path in wich the test dataset is located
            train_test: If train it'll load formated train, if test it load formated test
            formated_train_path: Relative path in wich the train dataset will be stored
            formated_test_path: Relative path in wich the test dataset will be stored
            attack_types: Names for the final class labels
            attack_map: Maps the labels to each class
            
        Args in my_env:
            batch_size: It represents the size of states and labels to be returned
            fails_episode: Number of fails allowed in an episode before it's done
    '''
    def __init__(self):
        self.__version__ = "0.0.1"
        logging.info("NetworkClassificationEnv - Version {}".format(self.__version__))
        

    def _update_state(self):
        '''
        Next enviroment observation
        Returns:
            states: array like of the new observation
            labels: category of the new attacks
        '''
        self.states,self.labels = self.get_batch(self.batch_size)
        

    def reset(self,train_test,attack_map,**kwargs):
        '''
        Reset the environment and send first observation
            states: array like of the new observation
        '''
        data_cls.__init__(self,train_test,attack_map,**kwargs)
        self.data_shape = self.get_shape()
        self.batch_size = kwargs.get('batch_size',1) # experience replay -> batch = 1
        self.fails_episode = kwargs.get('fails_episode',10) 
        
        # Gym spaces
        self.action_space = spaces.Discrete(len(self.attack_types))
        self.observation_space = spaces.Discrete(self.data_shape[0])
        
        self.observation_len = self.data_shape[1]-1
        
        self.counter = 0
        
        self.states,self.labels = self.get_batch(self.batch_size)
        
        return self.states
    
    def _get_rewards(self,actions):
        # Clear previous rewards
        self.reward = 0
         # Actualize new rewards == get_reward
        if actions == self.labels:
            self.reward = 1
            
#            # Define individual rewards
#            ######################
#            if actions == 4:
#                self.reward = 3
#            if actions == 3:
#                self.reward = 2
#            ######################3
        # Update fails counter
        else: #fails ++
            self.counter += 1

    def step(self,actions):
        '''
        Execute one step ahead in the environment
        
        Parameters
        ----------
        action : (int)
        
        Returns
        -------
            State (list): Next state for the game
            Reward (int): Actual reward
            done (bool): If the game ends (no end in this case)
        '''

       # Actualize rewards and fail counter
        self._get_rewards(actions)
            

        # Get new state and new true values
        self._update_state()

        # Calculate the end of the episode by fails counter
        if self.counter >= self.fails_episode:
            self.done = True
            
        else:
            self.done = False
            
        return self.states, self.reward, self.done
    
