import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
import json
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time





def get_data(file_name):
        
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
    
    
    df =pd.read_csv(file_name,sep=',',names=col_names)
    

    
    # Dataframe processing
    #protocol_type = pd.get_dummies(df['protocol_type'])
    #service = pd.get_dummies(df['service'])
    #flag = pd.get_dummies(df['flag'])
    
    df = pd.concat([df.drop('protocol_type', axis=1), pd.get_dummies(df['protocol_type'])], axis=1)
    df = pd.concat([df.drop('service', axis=1), pd.get_dummies(df['service'])], axis=1)
    df = pd.concat([df.drop('flag', axis=1), pd.get_dummies(df['flag'])], axis=1)
    
    
    
    # 1 if ``su root'' command attempted; 0 otherwise 
    df['su_attempted'] = df['su_attempted'].replace(2.0, 0.0)
    
    # Normalization of the df
    for indx,dtype in df.dtypes.iteritems():
        if dtype == 'float64' or dtype == 'int64':
            if df[indx].max() == 0 and df[indx].min()== 0:
                df[indx] = 0
            else:
                df[indx] = (df[indx]-df[indx].min())/(df[indx].max()-df[indx].min())
                
    #normalized_df=(df-df.mean())/df.std()
    
    # One-hot-Encoding for reaction. 4 detection binary label 
    # labels = pd.get_dummies(df['labels'])
    
    # '0' if the data is normal '1' if atack
    labels = 1 - pd.get_dummies(df['labels'])['normal.']
    del df['labels']
    
    return df,labels





'''
Definition
'''
class RLenv(object):
    def __init__(self,path = '../datasets/kddcup.data_10_percent_corrected'):
        self.path = path
        self.data,self.labels = get_data(self.path)
        self.reset()
        
        
    def _update_state(self):
        self.state_numb += 1
        self.state = self.data.iloc[[self.state_numb]].values
        
    '''
    Returns:
        + Observation of the enviroment
    '''
    def reset(self):
        self.state_numb = 0
        self.data, self.labels = shuffle(self.data,self.labels,
                                         random_state=np.random.randint(0,100))
        self.total_reward = 0
        self.steps_in_episode = 0


        return self.data.iloc[[self.state_numb]].values  
    
    def act(self,action):
        self._update_state()
        self.steps_in_episode += 1
        if self.labels.iloc[[self.state_numb]].values == action:
            self.reward = 1
            self.total_reward += 1
            self.done = False
        else:
            self.reward = 0
            self.done = True
                
            
        #if(self.steps_in_episode>=50):
        #if(abs(self.total_reward)>=50 or self.steps_in_episode>=200):
        #    self.done = True
        #else:
        #    self.done = False
        
        return self.state, self.reward, self.done
    
        
        

class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, done):
        self.memory.append([states, done])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state, action, reward, nex_state = self.memory[idx][0]
            done = self.memory[idx][1]

            inputs[i:i+1] = state
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state)[0]
            Q_sa = np.max(model.predict(next_state)[0])
            if done:  # if done is True
                targets[i, action] = reward
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets



if __name__ == "__main__":
  
    kdd_10_path = '../datasets/kddcup.data_10_percent_corrected'
    micro_kdd = '../datasets/micro_kddcup.data'
    # Valid actions = '0' supose no atack, '1' supose atack
    valid_actions = [0, 1]
    num_actions = len(valid_actions)
    epsilon = .1  # exploration
    num_episodes = 1000
    #3max_memory = 100
    decay_rate = 0.99
    discount_factor = 0.9
    
    
    hidden_size = 100
    batch_size = 50

    # Initialization of the enviroment
    env = RLenv(kdd_10_path)

    
    # Network arquitecture
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(env.data.shape[1],), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.2), "mse")
    
    
    reward_chain = []
    loss_chain = []
    # Initialize a win-counter
    for i_episode in range(num_episodes):
        
        loss = 0.
        total_reward_by_episode = 0
        state = env.reset()
        done = False
        ones = 0
        zeros = 0
        
        # Iteration in one episode
        while not done:
                # get next action
            exploration = epsilon*decay_rate**i_episode
            if np.random.rand() <= exploration:
                action = np.random.randint(0, num_actions)
            else:
                q = model.predict(state)
                action = np.argmax(q[0])
        
            # apply action, get rewards and new state
            next_state, reward, done = env.act(action)
            
            # Test
            if(action==0):
                zeros += 1
            else:
                ones += 1
            
            total_reward_by_episode += reward
            
                        
            targets = model.predict(state)
            Q_sa = np.max(model.predict(next_state))
            if done:  # if done is True
                targets[0][action] = reward
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[0][action] = reward + discount_factor * Q_sa            
            
            
            
            
            loss += model.train_on_batch(state, targets)
            
            # Update the state
            state = next_state
            
            reward_chain.append(total_reward_by_episode)    
            loss_chain.append(loss)
            
        print("\rEpoch {:03d}/{:03d} | Loss {:4.4f} | Tot reward x episode {:03d}| Ones/Zeros: {}/{} ".format(i_episode,
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



