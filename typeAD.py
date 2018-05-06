'''
Multiple anomaly detection file
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras import optimizers
import json
from sklearn.utils import shuffle
import os
import sys
import time




'''
Data class processing
'''

class data_cls:
    def __init__(self, path,train_test,**kwargs):
        col_names = ["duration","protocol_type","service","flag","src_bytes",
            "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
            "logged_in","num_compromised","root_shell","su_attempted","num_root",
            "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
            "is_host_login","is_guest_login","count","srv_count","serror_rate",
            "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
            "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
            "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
            "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels"]
        self.index = 0
        # Data formated path and test path. 
        self.formated_path = "../datasets/formated/formated_data_type.data"
        self.test_path = "../datasets/formated/test_data_type.data"
        self.loaded = False
        self.train_test = train_test
        self.second_path = kwargs.get('join_path', '../datasets/corrected')

        
        self.attack_types = ['normal','DoS','Probe','R2L','U2R']
        self.attack_map =   { 'normal.': 'normal',
                        
                        'back.': 'DoS',
                        'land.': 'DoS',
                        'neptune.': 'DoS',
                        'pod.': 'DoS',
                        'smurf.': 'DoS',
                        'teardrop.': 'DoS',
                        'mailbomb.': 'DoS',
                        'apache2.': 'DoS',
                        'processtable.': 'DoS',
                        'udpstorm.': 'DoS',
                        
                        'ipsweep.': 'Probe',
                        'nmap.': 'Probe',
                        'portsweep.': 'Probe',
                        'satan.': 'Probe',
                        'mscan.': 'Probe',
                        'saint.': 'Probe',
                    
                        'ftp_write.': 'R2L',
                        'guess_passwd.': 'R2L',
                        'imap.': 'R2L',
                        'multihop.': 'R2L',
                        'phf.': 'R2L',
                        'spy.': 'R2L',
                        'warezclient.': 'R2L',
                        'warezmaster.': 'R2L',
                        'sendmail.': 'R2L',
                        'named.': 'R2L',
                        'snmpgetattack.': 'R2L',
                        'snmpguess.': 'R2L',
                        'xlock.': 'R2L',
                        'xsnoop.': 'R2L',
                        'worm.': 'R2L',
                        
                        'buffer_overflow.': 'U2R',
                        'loadmodule.': 'U2R',
                        'perl.': 'U2R',
                        'rootkit.': 'U2R',
                        'httptunnel.': 'U2R',
                        'ps.': 'U2R',    
                        'sqlattack.': 'U2R',
                        'xterm.': 'U2R'
                    }
        
        # If path is not provided system out error
        if (not path):
            print("Path: not path name provided", flush = True)
            sys.exit(0)
        formated = False     
        
        
        if os.path.exists(self.formated_path) and train_test=="train":
            formated = True
        elif os.path.exists(self.test_path) and train_test=="test":
            formated = True
        elif os.path.exists(self.test_path) and os.path.exists(self.formated_path) and (train_test == 'full' or train_test=='join'):
            formated = True
            
       
        # If the formatted data path exists, is not needed to process it again
        if os.path.exists(self.formated_path):
            formated = True
            

        # If it does not exist, it's needed to format the data
        if not formated:
            ''' Formating the dataset for ready-2-use data'''
            self.df = pd.read_csv(path,sep=',',names=col_names)
            if 'dificulty' in self.df.columns:
                self.df.drop('dificulty', axis=1, inplace=True) #in case of difficulty     
                
            if train_test == 'join':
                data2 = pd.read_csv(self.second_path,sep=',',names=col_names)
                if 'dificulty' in data2:
                    del(data2['dificulty'])
                train_indx = self.df.shape[0]
                frames = [self.df,data2]
                self.df = pd.concat(frames)
            # Data now is in RAM
            self.loaded = True
            
            # Dataframe processing
            self.df = pd.concat([self.df.drop('protocol_type', axis=1), pd.get_dummies(self.df['protocol_type'])], axis=1)
            self.df = pd.concat([self.df.drop('service', axis=1), pd.get_dummies(self.df['service'])], axis=1)
            self.df = pd.concat([self.df.drop('flag', axis=1), pd.get_dummies(self.df['flag'])], axis=1)
              
            
            # 1 if ``su root'' command attempted; 0 otherwise 
            self.df['su_attempted'] = self.df['su_attempted'].replace(2.0, 0.0)
            
            # Normalization of the df
            #normalized_df=(df-df.mean())/df.std()
            for indx,dtype in self.df.dtypes.iteritems():
                if dtype == 'float64' or dtype == 'int64':
                    if self.df[indx].max() == 0 and self.df[indx].min()== 0:
                        self.df[indx] = 0
                    else:
                        self.df[indx] = (self.df[indx]-self.df[indx].min())/(self.df[indx].max()-self.df[indx].min())
                    
                      
            # One-hot-Encoding for reaction.  
            all_labels = self.df['labels'] # Get all labels in df
            mapped_labels = np.vectorize(self.attack_map.get)(all_labels) # Map attacks
            self.df = self.df.reset_index(drop=True)
            self.df = pd.concat([self.df.drop('labels', axis=1),pd.get_dummies(mapped_labels)], axis=1)
            
             # Save data
            # suffle data: if join shuffled before in order to save train/test
            if train_test != 'join':
                self.df = shuffle(self.df,random_state=np.random.randint(0,100))            
            
           
            if train_test == 'train':
                self.df.to_csv(self.formated_path,sep=',',index=False)
            elif train_test == 'test':
                self.df.to_csv(self.test_path,sep=',',index=False)
            elif train_test == 'full':
            # 70% train 30% test
                train_indx = np.int32(self.df.shape[0]*0.7)
                test_df = self.df.iloc[train_indx:self.df.shape[0]]
                self.df = self.df[:train_indx]
                test_df.to_csv(self.test_path,sep=',',index=False)
                self.df.to_csv(self.formated_path,sep=',',index=False)
            else: #join: index calculated before
                test_df = self.df.iloc[train_indx:self.df.shape[0]]
                test_df = shuffle(test_df,random_state=np.random.randint(0,100))
                self.df = self.df[:train_indx]
                self.df = shuffle(self.df,random_state=np.random.randint(0,100))
                test_df.to_csv(self.test_path,sep=',',index=False)
                self.df.to_csv(self.formated_path,sep=',',index=False)
            
            
        
    ''' Get n-row batch from the dataset
        Return: df = n-rows
                labels = correct labels for detection 
    Sequential for largest datasets
    '''
    def get_sequential_batch(self, batch_size=100):
        if self.loaded is False:
            self.df = pd.read_csv(self.formated_path,sep=',', nrows = batch_size)
            self.loaded = True
        else:
            self.df = pd.read_csv(self.formated_path,sep=',', nrows = batch_size,
                         skiprows = self.index)
        
        self.index += batch_size

        labels = self.df[self.attack_types]
        for att in self.attack_types:
            del(self.df[att])
        return self.df,labels
    
    
    ''' Get n-rows from loaded data 
        The dataset must be loaded in RAM
    '''
    def get_batch(self, batch_size=100):
        
        if self.loaded is False:
            self.df = pd.read_csv(self.formated_path,sep=',') # Read again the csv
            self.loaded = True
            #self.headers = list(self.df)
        
        batch = self.df.iloc[self.index:self.index+batch_size]
        self.index += batch_size
        labels = batch[self.attack_types]
        
        for att in self.attack_types:
            del(batch[att])
        return batch,labels
    
    
    
    def get_shape(self):
        if self.loaded is False:
            self._load_df()
        
        self.data_shape = self.df.shape
        # stata + labels
        return self.data_shape
    
    def _load_df(self):
        if self.train_test == 'train' or self.train_test == 'full':
            self.df = pd.read_csv(self.formated_path,sep=',') # Read again the csv
        else:
            self.df = pd.read_csv(self.test_path,sep=',')
        self.loaded = True




class QNetwork():
    """
    Q-Network Estimator
    Represents the global model for the table
    """

    def __init__(self,obs_size,num_actions,batch_size=1,hidden_size = 100,
                 hidden_layers = 1,learning_rate=.2):
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
        
        optimizer = optimizers.SGD(learning_rate)
        # optimizer = optimizers.Adam(alpha=learning_rate)
        # optimizer = optimizers.AdaGrad(learning_rate)
        # optimizer = optimizers.RMSpropGraves(learning_rate, 0.95, self.momentum, 1e-2)
        
        # Compilation of the model with optimizer and loss
        self.model.compile(optimizer,"mse")

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




#Policy interface
class Policy:
    def __init__(self, num_actions, estimator,batch_size):
        self.num_actions = num_actions
        self.estimator = estimator
        self.batch_size = batch_size
    
class Epsilon_greedy(Policy):
    def __init__(self,estimator ,num_actions ,batch_size,epsilon,decay_rate, epoch_length):
        Policy.__init__(self, num_actions, estimator,batch_size)
        self.name = "Epsilon Greedy"
        if (epsilon is None or epsilon < 0 or epsilon > 1):
            print("EpsilonGreedy: Invalid value of epsilon", flush = True)
            sys.exit(0)
        self.epsilon = epsilon
        self.step_counter = 0
        self.epoch_length = epoch_length
        self.batch_size = batch_size
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
            actions = np.random.randint(0, self.num_actions,self.batch_size)
        else:
            self.Q = self.estimator.predict(states,self.batch_size)
            actions = np.argmax(self.Q,axis=1)
            
        self.step_counter += 1 
        # decay epsilon after each epoch
        if self.epsilon_decay:
            if self.step_counter % self.epoch_length == 0:
                self.epsilon = max(.001, self.epsilon * self.decay_rate**self.step_counter)
            
        return actions
    



'''
Reinforcement learning Agent definition
'''

class Agent(object):  
        
    def __init__(self, actions,obs_size):
        self.actions = actions
        self.num_actions = len(actions)
        self.obs_size = obs_size
        

    def act(self, state,policy):
        raise NotImplementedError

#class AttackerAgent(Agent):
#
#    def __init__(self, actions, obs_size, **kwargs):
#        super().__init__(actions)


class DefenderAgent(Agent):      
    def __init__(self, actions, obs_size, policy="EpsilonGreedy", **kwargs):
        super().__init__(actions,obs_size)
        
        self.epsilon = kwargs.get('epsilon', .01)
        self.gamma = kwargs.get('gamma', .001)
        self.batch_size = kwargs.get('batch_size', 1)
        self.epoch_length = kwargs.get('epoch_length', 100)
        self.decay_rate = kwargs.get('decay_rate',0.99)
        
        
        self.model_network = QNetwork(self.obs_size, self.num_actions,
                                      kwargs.get('batch_size',1),
                                      kwargs.get('hidden_size', 100),
                                      kwargs.get('hidden_layers',1),
                                      kwargs.get('learning_rate',.2))
        if policy == "EpsilonGreedy":
            self.policy = Epsilon_greedy(self.model_network,len(actions),
                                         self.batch_size,self.epsilon,
                                         self.decay_rate,self.epoch_length)
        
        
    def act(self,states):
        # Get actions under the policy
        actions = self.policy.get_actions(states)
        return actions
    
    def update_model(self,states,actions,next_states,reward):
        # Compute Q targets
        Q_prime = self.model_network.predict(next_states)
        indx = np.argmax(Q_prime,axis=1)
        sx = np.arange(len(indx))
        # Compute Q(s,a)
        Q = self.model_network.predict(states)
        
        # Q-learning update
        targets = reward + self.gamma * Q[sx,indx]   
        Q[sx,indx] = targets  
        
        loss = self.model_network.model.train_on_batch(states,Q)
        
        return loss
        
        

'''
Reinforcement learning Enviroment Definition
'''
class RLenv(data_cls):
    def __init__(self,path,train_test,batch_size = 10,**kwargs):
        data_cls.__init__(self,path,train_test,**kwargs)
        self.batch_size = batch_size
        self.data_shape = data_cls.get_shape(self)

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
        self.reward = np.zeros(self.batch_size)
        
        # Actualize new rewards == get_reward
        for indx,a in enumerate(actions):
            self.estimated_labels[a] += 1              
            if a == np.argmax(self.labels.iloc[indx].values):
                self.reward[indx] = 1
        
        # Get new state and new true values
        self._update_state()
        
        # Done allways false in this continuous task       
        self.done = False
            
        return self.states, self.reward, self.done
    



if __name__ == "__main__":
  
    kdd_10_path = '../datasets/kddcup.data_10_percent_corrected'
    kdd_path = '../datasets/kddcup.data'

    # Valid actions = '0' supose no attack, '1' supose attack
    epsilon = 1  # exploration

    batch_size = 10

    #3max_memory = 100
    decay_rate = 0.99
    gamma = 0.001
    
    
    hidden_size = 100
    hidden_layers = 3
    

    # Initialization of the enviroment
    env = RLenv(kdd_path,'train',batch_size,join_path='../datasets/corrected')
    
    iterations_episode = 100
    num_episodes = int(env.data_shape[0]/(iterations_episode*batch_size)/10)
	
    valid_actions = list(range(len(env.attack_types))) # only detect type of attack
    num_actions = len(valid_actions)
    
    # Initialization of the Agent
    obs_size = env.data_shape[1]-len(env.attack_types)
    
    agent = DefenderAgent(valid_actions,obs_size,"EpsilonGreedy",
                          batch_size=batch_size,
                          epoch_length = iterations_episode,
                          epsilon = epsilon,
                          decay_rate = decay_rate,
                          gamma = gamma,
                          hidden_size=hidden_size,
                          hidden_layers=hidden_layers)    
    
    
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
       

        # Iteration in one episode
        for i_iteration in range(iterations_episode):
            

            # Get actions for actual states following the policy
            actions = agent.act(states)
            #Enviroment actuation for this actions
            next_states, reward, done = env.act(actions)
            # If the epoch*batch_size*iterations_episode is largest than the df
            if next_states.shape[0] != batch_size:
                break # finished df
            
            
            # Train network, update loss
            loss += agent.update_model(states,actions,next_states,reward)
            
            update_end_time = time.time()

            # Update the state
            states = next_states
            
            
            # Update statistics
            total_reward_by_episode += int(sum(reward))
        
        if next_states.shape[0] != batch_size:
                break # finished df
        # Update user view
        reward_chain.append(total_reward_by_episode)    
        loss_chain.append(loss) 
        
        end_time = time.time()
        print("\r|Epoch {:03d}/{:03d} | Loss {:4.4f} |" 
                "Tot reward in ep {:03d}| time: {:2.2f}|"
                .format(epoch, num_episodes 
                ,loss, total_reward_by_episode,(end_time-start_time)))
        print("\r|Estimated: {}|Labels: {}".format(env.estimated_labels,env.true_labels))
        
    # Save trained model weights and architecture, used in test
    agent.model_network.model.save_weights("models/type_model.h5", overwrite=True)
    with open("models/type_model.json", "w") as outfile:
        json.dump(agent.model_network.model.to_json(), outfile)
        
    # Save test dataset deleting the data used to train
    print("Shape: ",env.data_shape)
    print("Used: ",num_episodes*iterations_episode*batch_size)
    #env.save_test()
    
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
    plt.savefig('results/train_type.eps', format='eps', dpi=1000)




