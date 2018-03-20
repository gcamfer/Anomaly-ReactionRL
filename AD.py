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
        self.headers = None
        
        if (not path):
            print("Path: not path name provided", flush = True)
            sys.exit(0)
        formated = False     
        # Search for a previous formated data:
        #if (not os.path.exists('../datasets')):
        #    os.makedirs('../datasets')
        #    formated = False
        self.data_path = '../datasets/formated_data.data'
        if os.path.exists(self.data_path):
            formated = True
        # If it does not exist, it's needed to format the data
        if not formated:
            ''' Formating the dataset for ready-2-use data'''
            df = pd.read_csv(path,sep=',',names=col_names)
            
            # Dataframe processing
            df = pd.concat([df.drop('protocol_type', axis=1), pd.get_dummies(df['protocol_type'])], axis=1)
            df = pd.concat([df.drop('service', axis=1), pd.get_dummies(df['service'])], axis=1)
            df = pd.concat([df.drop('flag', axis=1), pd.get_dummies(df['flag'])], axis=1)
              
            #normalized_df=(df-df.mean())/df.std()
            
            # 1 if ``su root'' command attempted; 0 otherwise 
            df['su_attempted'] = df['su_attempted'].replace(2.0, 0.0)
            
            # Normalization of the df
            for indx,dtype in df.dtypes.iteritems():
                if dtype == 'float64' or dtype == 'int64':
                    if df[indx].max() == 0 and df[indx].min()== 0:
                        df[indx] = 0
                    else:
                        df[indx] = (df[indx]-df[indx].min())/(df[indx].max()-df[indx].min())
                    
            # One-hot-Encoding for reaction. 4 detection binary label 
            # labels = pd.get_dummies(df['labels'])
        
            # Only save one column for atack
            # '0' if the data is normal '1' if atack
            df = pd.concat([df.drop('labels', axis=1),
                            1 - pd.get_dummies(df['labels'])['normal.']], axis=1)
            #Only detectign label as normal = 0 atack = 1 -> reanaming column
            df.rename(columns={'normal.': 'labels'},inplace=True)
            # suffle data
            df = shuffle(df,random_state=np.random.randint(0,100))
            df.to_csv(self.data_path,sep=',',index=False)
            
    ''' Get n-rows from the dataset'''
    def get_batch(self, batch_size=100):
        if self.headers is None:
            df = pd.read_csv(self.data_path,sep=',', nrows = batch_size)
            self.headers = list(df)
        else:
            df = pd.read_csv(self.data_path,sep=',', nrows = batch_size,
                         skiprows = self.index,names = self.headers)
        
        self.index += batch_size

        labels = df['labels']
        del(df['labels'])
        return df,labels
        
    def get_size(self):
        df = pd.read_csv(self.data_path,sep=',',nrows=1)
        # stata + labels
        return df.shape[1]-1


'''
Definition
'''
class RLenv(data_cls):
    def __init__(self,path,batch_size = 100):
        data_cls.__init__(self,path)
        self.batch_size = batch_size
        self.state_size = data_cls.get_size(self)

    def _update_state(self):
        self.states,self.labels = data_cls.get_batch(self,self.batch_size)
    '''
    Returns:
        + Observation of the enviroment
    '''
    def reset(self):
        self.state_numb = 0
        self.states,self.labels = data_cls.get_batch(self,self.batch_size)
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
        self.reward = np.zeros(self.batch_size)
        for indx,a in enumerate(actions):
            if a == self.labels[indx]:
                self.reward[indx] = 1
        
        # Get new state and new true values
        self._update_state()
        
        # Done allways false in this continuous task       
        self.done = False
            
        return self.states, self.reward, self.done
    



if __name__ == "__main__":
  
    kdd_10_path = '../datasets/kddcup.data_10_percent_corrected'
    micro_kdd = '../datasets/micro_kddcup.data'
    # Valid actions = '0' supose no atack, '1' supose atack
    valid_actions = [0, 1]
    num_actions = len(valid_actions)
    epsilon = .1  # exploration
    num_episodes = 100
    iterations_episode = 100
    
    #3max_memory = 100
    decay_rate = 0.99
    gamma = 0.001
    
    
    hidden_size = 100
    batch_size = 10

    # Initialization of the enviroment
    env = RLenv(kdd_10_path,batch_size)

    
    # Network arquitecture
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(env.state_size,),batch_size=batch_size, activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.2), "mse")
    
    
    reward_chain = []
    loss_chain = []
    

    
    # Main loop
    for epoch in range(num_episodes):
        loss = 0.
        total_reward_by_episode = 0
        # Reset enviromet, actualize the data batch
        states = env.reset()
        
        done = False
        # Get control of the actions taken
        ones = 0
        zeros = 0
        
        # Iteration in one episode
        for i_iteration in range(iterations_episode):
            
            # get next action
            exploration = epsilon*decay_rate**epoch
            if np.random.rand() <= exploration:
                actions = np.random.randint(0, num_actions,batch_size)
            else:
                q = model.predict(states)
                actions = np.argmax(q,axis=1)
            
            # apply actions, get rewards and new state
            next_states, reward, done = env.act(actions)
            
            q_prime = model.predict(next_states)
            indx = np.argmax(q_prime,axis=1)
            sx = np.arange(len(indx))
            # Update q values
            targets = reward + gamma * q[sx,indx]   
            q[sx,indx] = targets         
            
            # Train network, update loss
            loss += model.train_on_batch(states, q)
            
            # Update the state
            states = next_states
            
            
            ones += int(sum(actions))
            zeros += batch_size - int(sum(actions))
            total_reward_by_episode += int(sum(reward))
            
            reward_chain.append(total_reward_by_episode)    
            loss_chain.append(loss)
            
        print("\rEpoch {:03d}/{:03d} | Loss {:4.4f} | Tot reward x episode {:03d}| Ones/Zeros: {}/{} ".format(epoch,
              num_episodes ,loss, total_reward_by_episode,ones,zeros))
        
        
    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)
    
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



