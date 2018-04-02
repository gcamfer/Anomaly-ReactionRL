'''
Multiple anomaly detection file
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
import json
from sklearn.utils import shuffle
import os
import sys
import time






class data_cls:
    def __init__(self, path):
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
        self.formated_path = "../datasets/formated_multiple_data.data"
        self.test_path = "../datasets/test_multiple_data.data"
        self.loaded = False
        
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
       
        # If the formatted data path exists, is not needed to process it again
        if os.path.exists(self.formated_path):
            formated = True
            

        # If it does not exist, it's needed to format the data
        if not formated:
            ''' Formating the dataset for ready-2-use data'''
            df = pd.read_csv(path,sep=',',names=col_names)
            
            # Data now is in RAM
            self.loaded = True
            
            # Dataframe processing
            df = pd.concat([df.drop('protocol_type', axis=1), pd.get_dummies(df['protocol_type'])], axis=1)
            df = pd.concat([df.drop('service', axis=1), pd.get_dummies(df['service'])], axis=1)
            df = pd.concat([df.drop('flag', axis=1), pd.get_dummies(df['flag'])], axis=1)
              
            
            # 1 if ``su root'' command attempted; 0 otherwise 
            df['su_attempted'] = df['su_attempted'].replace(2.0, 0.0)
            
            # Normalization of the df
            #normalized_df=(df-df.mean())/df.std()
            for indx,dtype in df.dtypes.iteritems():
                if dtype == 'float64' or dtype == 'int64':
                    if df[indx].max() == 0 and df[indx].min()== 0:
                        df[indx] = 0
                    else:
                        df[indx] = (df[indx]-df[indx].min())/(df[indx].max()-df[indx].min())
                    
                      
            # One-hot-Encoding for reaction.  
            all_labels = df['labels'] # Get all labels in df
            mapped_labels = np.vectorize(self.attack_map.get)(all_labels) # Map attacks
            df = pd.concat([df.drop('labels', axis=1),
                            pd.get_dummies(mapped_labels)], axis=1)
            
            # suffle data
            df = shuffle(df,random_state=np.random.randint(0,100))
            # Save data
            df.to_csv(self.formated_path,sep=',',index=False)
            
    ''' Get n-row batch from the dataset
        Return: df = n-rows
                labels = correct labels for detection 
    Sequential for largest datasets
    '''
    def get_sequential_batch(self, batch_size=100):
        if self.loaded is False:
            df = pd.read_csv(self.formated_path,sep=',', nrows = batch_size)
            self.loaded = True
        else:
            df = pd.read_csv(self.formated_path,sep=',', nrows = batch_size,
                         skiprows = self.index)
        
        self.index += batch_size

        labels = df[self.attack_types]
        for att in self.attack_types:
            del(df[att])
        return df,labels
    
    
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
    
    def save_test(self):
        test_df = self.df.iloc[self.index:self.data_shape[0] + 1]
        test_df.to_csv(self.test_path,sep=',',index=False)
    
    def get_full(self):
        self.df = pd.read_csv(self.data_path,sep=',')        
        self.labels = self.df['labels']
        self.loaded = True
        del(self.df['labels'])
        
    def get_shape(self):
        if self.loaded is False:
            self.df = pd.read_csv(self.formated_path,sep=',') # Read again the csv
            self.loaded = True
        
        self.data_shape = self.df.shape
        # stata + labels
        return self.data_shape


'''
Definition
'''
class RLenv(data_cls):
    def __init__(self,path,batch_size = 10):
        data_cls.__init__(self,path)
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
    epsilon = .1  # exploration
    num_episodes = 300
    iterations_episode = 100
    
    #3max_memory = 100
    decay_rate = 0.99
    gamma = 0.001
    
    
    hidden_size_1 = 100
    hidden_size_2 = 300
    batch_size = 10

    # Initialization of the enviroment
    env = RLenv(kdd_path,batch_size)

    valid_actions = list(range(len(env.attack_types))) # only detect type of attack
    num_actions = len(valid_actions)
    
    # Network arquitecture
    model = Sequential()
    model.add(Dense(hidden_size_1, input_shape=(env.data_shape[1]-len(env.attack_types),),
                    batch_size=batch_size, activation='relu'))
    model.add(Dense(hidden_size_1, activation='relu'))
    model.add(Dense(hidden_size_2, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.2), "mse")
    
    
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
       
        # Define exploration to improve performance
        exploration = 1
        # Define q to avoid not defined in the q update
        q = np.zeros([batch_size,num_actions])
        # Iteration in one episode
        for i_iteration in range(iterations_episode):
            
            # get next action
            if exploration > 0.001:
                exploration = epsilon*decay_rate**(epoch*i_iteration)            
            if np.random.rand() <= exploration:
                actions = np.random.randint(0, num_actions,batch_size)
            else:
                q = model.predict(states)
                actions = np.argmax(q,axis=1)
            
            
            # apply actions, get rewards and new state
            act_time = time.time()
            next_states, reward, done = env.act(actions)
            act_end_time = time.time()
            
            q_prime = model.predict(next_states)
            indx = np.argmax(q_prime,axis=1)
            sx = np.arange(len(indx))
            # Update q values
            targets = reward + gamma * q[sx,indx]   
            q[sx,indx] = targets         
            
            update_end_time = time.time()
            
            # Train network, update loss
            loss += model.train_on_batch(states, q)
            # Update the state
            states = next_states
            
            
            # Update statistics
            total_reward_by_episode += int(sum(reward))
        
        # Update user view
        app_time = time.time()
        reward_chain.append(total_reward_by_episode)    
        loss_chain.append(loss) 
        app_end_time = time.time() - app_time
        
        end_time = time.time()
        print("\r|Epoch {:03d}/{:03d} | Loss {:4.4f} |" 
                "Tot reward in ep {:03d}| time: {:2.2f}|"
                .format(epoch, num_episodes 
                ,loss, total_reward_by_episode,(end_time-start_time)))
        print("\r|Estimated: {}|Labels: {}".format(env.estimated_labels,env.true_labels))
        
    # Save trained model weights and architecture, used in test
    model.save_weights("multi_model.h5", overwrite=True)
    with open("multi_model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)
        
    # Save test dataset deleting the data used to train
    print("Shape: ",env.data_shape)
    print("Used: ",num_episodes*iterations_episode*batch_size)
    env.save_test()
    
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



