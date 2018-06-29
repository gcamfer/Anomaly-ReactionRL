'''
Type anomaly detection file
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
import os
import sys
import time

import tensorflow as tf
import tensorflow.contrib.layers as layer

import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import  confusion_matrix




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



'''
Data class processing
'''

class data_cls:
    def __init__(self,train_test,**kwargs):
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

        self.loaded = False
        self.train_test = train_test
        self.train_path = kwargs.get('train_path', '../../datasets/NSL/KDDTrain+.txt')
        self.test_path = kwargs.get('test_path','../../datasets/NSL/KDDTest+.txt')
        
        self.formated_train_path = kwargs.get('formated_train_path', 
                                              "../../datasets/formated/formated_train_type.data")
        self.formated_test_path = kwargs.get('formated_test_path',
                                             "../../datasets/formated/formated_test_type.data")



        
        self.attack_types = ['normal','DoS','Probe','R2L','U2R']
        self.attack_map =   { 'normal': 'normal',
                        
                        'back': 'DoS',
                        'land': 'DoS',
                        'neptune': 'DoS',
                        'pod': 'DoS',
                        'smurf': 'DoS',
                        'teardrop': 'DoS',
                        'mailbomb': 'DoS',
                        'apache2': 'DoS',
                        'processtable': 'DoS',
                        'udpstorm': 'DoS',
                        
                        'ipsweep': 'Probe',
                        'nmap': 'Probe',
                        'portsweep': 'Probe',
                        'satan': 'Probe',
                        'mscan': 'Probe',
                        'saint': 'Probe',
                    
                        'ftp_write': 'R2L',
                        'guess_passwd': 'R2L',
                        'imap': 'R2L',
                        'multihop': 'R2L',
                        'phf': 'R2L',
                        'spy': 'R2L',
                        'warezclient': 'R2L',
                        'warezmaster': 'R2L',
                        'sendmail': 'R2L',
                        'named': 'R2L',
                        'snmpgetattack': 'R2L',
                        'snmpguess': 'R2L',
                        'xlock': 'R2L',
                        'xsnoop': 'R2L',
                        'worm': 'R2L',
                        
                        'buffer_overflow': 'U2R',
                        'loadmodule': 'U2R',
                        'perl': 'U2R',
                        'rootkit': 'U2R',
                        'httptunnel': 'U2R',
                        'ps': 'U2R',    
                        'sqlattack': 'U2R',
                        'xterm': 'U2R'
                    }
        
        


        formated = False
        
        # Test formated data exists
        if os.path.exists(self.formated_train_path) and os.path.exists(self.formated_test_path):
            formated = True
               

        # If it does not exist, it's needed to format the data
        if not formated:
            ''' Formating the dataset for ready-2-use data'''
            self.df = pd.read_csv(self.train_path,sep=',',names=col_names,index_col=False)
            if 'dificulty' in self.df.columns:
                self.df.drop('dificulty', axis=1, inplace=True) #in case of difficulty     
                
            data2 = pd.read_csv(self.test_path,sep=',',names=col_names,index_col=False)
            if 'dificulty' in data2:
                del(data2['dificulty'])
            train_indx = self.df.shape[0]
            frames = [self.df,data2]
            self.df = pd.concat(frames)
            
            
            # Dataframe processing
            self.df = pd.concat([self.df.drop('protocol_type', axis=1), pd.get_dummies(self.df['protocol_type'])], axis=1)
            self.df = pd.concat([self.df.drop('service', axis=1), pd.get_dummies(self.df['service'])], axis=1)
            self.df = pd.concat([self.df.drop('flag', axis=1), pd.get_dummies(self.df['flag'])], axis=1)
              
            # 1 if ``su root'' command attempted; 0 otherwise 
            self.df['su_attempted'] = self.df['su_attempted'].replace(2.0, 0.0)
            
            
            # One-hot-Encoding for reaction.  
            all_labels = self.df['labels'] # Get all labels in df
            mapped_labels = np.vectorize(self.attack_map.get)(all_labels) # Map attacks
            self.df = self.df.reset_index(drop=True)
            self.df = pd.concat([self.df.drop('labels', axis=1),pd.get_dummies(mapped_labels)], axis=1)
            
            
            # Normalization of the df
            #self.df = (self.df-self.df.mean())/(self.df.max()-self.df.min())
            for indx,dtype in self.df.dtypes.iteritems():
                if dtype == 'float64' or dtype == 'int64':
                    if self.df[indx].max() == 0 and self.df[indx].min()== 0:
                        self.df[indx] = 0
                    else:
                        self.df[indx] = (self.df[indx]-self.df[indx].min())/(self.df[indx].max()-self.df[indx].min())
                    
                      
            
             # Save data
            test_df = self.df.iloc[train_indx:self.df.shape[0]]
            test_df = shuffle(test_df,random_state=np.random.randint(0,100))
            self.df = self.df[:train_indx]
            self.df = shuffle(self.df,random_state=np.random.randint(0,100))
            test_df.to_csv(self.formated_test_path,sep=',',index=False)
            self.df.to_csv(self.formated_train_path,sep=',',index=False)
            
            
        
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
            self._load_df()
            
        indexes = list(range(self.index,self.index+batch_size))    
        if max(indexes)>self.data_shape[0]-1:
            dif = max(indexes)-self.data_shape[0]
            indexes[len(indexes)-dif-1:len(indexes)] = list(range(dif+1))
            self.index=batch_size-dif
            batch = self.df.iloc[indexes]
        else: 
            batch = self.df.iloc[indexes]
            self.index += batch_size    
            

        labels = batch[self.attack_types]
        
        for att in self.attack_types:
            del(batch[att])
        return batch,labels
    
    def get_full(self):
        if self.loaded is False:
            self._load_df()
            
        batch = self.df        
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
        if self.train_test == 'train':
            self.df = pd.read_csv(self.formated_train_path,sep=',') # Read again the csv
        else:
            self.df = pd.read_csv(self.formated_test_path,sep=',')
        self.index=0
        self.loaded = True



class DuelingQnetwork():
    def __init__(self,scope="estimator",h_size=512,summaries_dir=None):    
        self.scope = scope
        self.h_size = h_size
        # Writes Tensorboard summaries to disk
        self.summary_writer = False
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)
        
    def _build_model(self):   
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        self.Input =  tf.placeholder(shape=[None,env.obs_size],dtype=tf.float32)
        
        
        self.out1 = layer.fully_connected(inputs=self.Input,num_outputs=100)
        self.out2 = layer.fully_connected(inputs=self.out1,num_outputs=100)
        self.out3 = layer.fully_connected(inputs=self.out2,num_outputs=self.h_size)
        
        #We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC,self.streamVC = tf.split(self.out3,2,1)
        
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([self.h_size//2,env.num_actions]))
        self.VW = tf.Variable(xavier_init([self.h_size//2,1]))
        self.Advantage = tf.matmul(self.streamAC,self.AW)
        self.Value = tf.matmul(self.streamVC,self.VW)

        
        #Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keepdims=True))
        self.predictions = tf.argmax(self.Qout,1)
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,env.num_actions,dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
#        self.loss = tf.reduce_mean(self.td_error)
        self.loss=tf.losses.huber_loss(self.targetQ,self.Q)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.00025)
#        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
#        self.trainer = tf.train.RMSPropOptimizer(0.0001, 0.999, 0.0, 1e-6)
        self.updateModel = self.trainer.minimize(self.loss,global_step=tf.train.get_global_step())
        
        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("td_error", self.td_error),
            tf.summary.histogram("q_values_hist", self.Qout),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.Qout))
            ])

    def predict(self,sess,states):
        """
        Predicts action values.
        """
        return sess.run(self.Qout,
                        feed_dict={self.Input:states})

    def update(self,sess,states,actions,targets):
        """
        Updates the estimator with the targets.

        Args:
          states: Target states
          q: Estimated values

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { self.Input: states, self.actions: actions, self.targetQ: targets }
        summaries,global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.updateModel, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)        
        
        return loss

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
        
    
    def get_actions(self,sess,states):
        # get next action
        if np.random.rand() <= self.epsilon:
            actions = np.random.randint(0, self.num_actions,states.shape[0])
        else:
            self.Q = self.estimator.predict(sess,states)
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

        
#        self.model_network = QNetwork(self.obs_size, self.num_actions,
#                                      kwargs.get('hidden_size', 100),
#                                      kwargs.get('hidden_layers',1),
#                                      kwargs.get('learning_rate',.2))
#        
#        self.target_model_network = QNetwork(self.obs_size, self.num_actions,
#                                      kwargs.get('hidden_size', 100),
#                                      kwargs.get('hidden_layers',1),
#                                      kwargs.get('learning_rate',.2))
#        self.target_model_network.model = QNetwork.copy_model(self.model_network.model)
        
        h_size = 512
        self.model_network = DuelingQnetwork(scope='q',h_size=h_size,summaries_dir='sumaries/ddqn')
        self.target_model_network = DuelingQnetwork(scope='target_q',h_size=h_size)
        
        
        if policy == "EpsilonGreedy":
            self.policy = Epsilon_greedy(self.model_network,len(actions),
                                         self.epsilon,self.decay_rate,
                                         self.epoch_length)
        
    def act(self,sess,states):
        # Get actions under the policy
        actions = self.policy.get_actions(sess,states)
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


    def update_model(self,sess):
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
        Q_prime = self.model_network.predict(sess,next_states)
        # TODO: fix performance in this loop
        for row in range(Q_prime.shape[0]):
            best_next_actions = np.argwhere(Q_prime[row] == np.amax(Q_prime[row]))
            next_actions.append(best_next_actions[np.random.choice(len(best_next_actions))].item())
        Q_prime_target = self.target_model_network.predict(sess,next_states)
        
        # Compute Q(s,a)
        # Q-learning update
#        doubleQ = Q_prime_target[range(self.minibatch_size),next_actions]
#        targetQ =  rewards + (self.gamma*doubleQ * (1-done))  
#        
        sx = np.arange(len(next_actions))
        targets = rewards.reshape(Q_prime_target[sx,next_actions].shape) + \
                  self.gamma * Q_prime_target[sx,next_actions] * \
                  (1-done.reshape(Q_prime_target[sx,next_actions].shape))   
        Q_prime_target[sx,actions] = targets  
        
        loss = self.model_network.update(sess,states,actions,targets)

        
        
        # timer to ddqn update
        self.ddqn_update -= 1
        if self.ddqn_update == 0:
            copy_model_parameters(sess,self.model_network,self.target_model_network)
            self.ddqn_update = self.ddqn_time
        
        return loss    
        


def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)
    
    
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

    

'''
Reinforcement learning Enviroment Definition
'''
class RLenv(data_cls):
    def __init__(self,train_test,**kwargs):
        data_cls.__init__(self,train_test,**kwargs)
        self.data_shape = data_cls.get_shape(self)
        self.batch_size = kwargs.get('batch_size',1) # experience replay -> batch = 1
        self.iterations_episode = kwargs.get('iterations_episode',10)
        if self.batch_size=='full':
            self.batch_size = int(self.data_shape[0]/iterations_episode)
        
        self.valid_actions = list(range(len(self.attack_types))) # only detect type of attack
        self.num_actions = len(self.valid_actions)
        self.obs_size = self.data_shape[1]-len(self.attack_types)

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
        self.reward = np.zeros(len(actions))
        
        # Actualize new rewards == get_reward
        self.reward = (actions == self.labels.values.argmax(axis=1)).astype(np.int32)
        labels,counts = np.unique(actions,return_counts=True)
        self.estimated_labels[labels] += counts
        # Get new state and new true values
        self._update_state()
        
        # Done allways false in this continuous task       
        self.done = False
            
        return self.states, self.reward, self.done
    

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

if __name__ == "__main__":
  
    kdd_20_path = '../../datasets/NSL/KDDTrain+_20Percent.txt'
    kdd_train = '../../datasets/NSL/KDDTrain+.txt'
    kdd_test = '../../datasets/NSL/KDDTest+.txt'

    formated_train_path = "../../datasets/formated/formated_train_type.data"
    formated_test_path = "../../datasets/formated/formated_test_type.data"

    model_path = "models/typeAD_tf"

    load_model = False
    
    tf.reset_default_graph()

    # Valid actions = '0' supose no attack, '1' supose attack
    epsilon = 1  # exploration
    
    # Train batch
    batch_size = 1
    # batch of memory ExpRep
    minibatch_size = 500
    ExpRep = True
    
    iterations_episode = 100

    #3max_memory = 100
    decay_rate = 0.999
    gamma = 0.001
    
    
    hidden_size = 100
    hidden_layers = 3

    # Initialization of the enviroment
    env = RLenv('train',train_path=kdd_train,test_path=kdd_test,
                formated_train_path = formated_train_path,
                formated_test_path = formated_test_path,batch_size=batch_size,
                iterations_episode=iterations_episode)
    

    
 
#    num_episodes = int(env.data_shape[0]/(iterations_episode)/10)
    num_episodes = 200

    # Initialization of the Agent
    
    agent = Agent(env.valid_actions,env.obs_size,"EpsilonGreedy",
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
    
        
     
    
    trainables = tf.trainable_variables()
    # Create a glboal step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)
    

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        
        
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
#                actions = agent.act(states)
                actions = agent.act(sess,states)
                #Enviroment actuation for this actions
                next_states, reward, done = env.act(actions)
                # If the epoch*batch_size*iterations_episode is largest than the df
    
                agent.learn(states,actions,next_states,reward,done)
                
                # Train network, update loss after at least minibatch_learns
                if ExpRep and epoch*iterations_episode + i_iteration >= minibatch_size:
                    loss += agent.update_model(sess)
#                    updateTarget(targetOps,sess) #Update the target network toward the primary network.
                    
                elif not ExpRep:
                    loss += np.sqrt(agent.update_model(sess))
                
                update_end_time = time.time()
    
                # Update the state
                states = next_states
                
                
                # Update statistics
                total_reward_by_episode += np.sum(reward,dtype=np.int32)
    
            # Update user view
            reward_chain.append(total_reward_by_episode)    
            loss_chain.append(loss) 
            
            # Correcting next states labels
            env.true_labels -= np.sum(env.labels).values
            
            end_time = time.time()
            print("\r|Epoch {:03d}/{:03d} | Loss {:4.4f} |" 
                    "Tot reward in ep {:03d}| time: {:2.2f}|"
                    .format(epoch, num_episodes 
                    ,loss, total_reward_by_episode,(end_time-start_time)))
            print("\r|Estimated: {}|Labels: {}".format(env.estimated_labels,env.true_labels))
            
        
        save_path = saver.save(sess, model_path+'.ckpt')
        print("Model saved in path: %s" % save_path)
        
#        # Save trained model weights and architecture, used in test
#        agent.model_network.model.save_weights("models/type_model.h5", overwrite=True)
#        with open("models/type_model.json", "w") as outfile:
#            json.dump(agent.model_network.model.to_json(), outfile)
            
    
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
        plt.savefig('results/train_dueling_network.eps', format='eps', dpi=1000)







        #TEST
        env = RLenv('test',formated_test_path = formated_test_path) 
        total_reward = 0    
        epochs = int(env.data_shape[0]/env.batch_size/1)
    
        true_labels = np.zeros(len(env.attack_types),dtype=int)
        estimated_labels = np.zeros(len(env.attack_types),dtype=int)
        estimated_correct_labels = np.zeros(len(env.attack_types),dtype=int)
        
        
        #states , labels = env.get_sequential_batch(test_path,batch_size = env.batch_size)
        states , labels = env.get_full()
        
        Q = agent.model_network.predict(sess,states)
        # TODO: fix performance in this loop
        actions = []
        for row in range(Q.shape[0]):
            best_actions = np.argwhere(Q[row] == np.amax(Q[row]))
            actions.append(best_actions[np.random.choice(len(best_actions))].item())
        
        
        
        true_labels += np.sum(labels).values

        for indx,a in enumerate(actions):
            estimated_labels[a] +=1              
            if a == np.argmax(labels.iloc[indx].values):
                total_reward += 1
                estimated_correct_labels[a] += 1
        
        


        Accuracy = estimated_correct_labels / true_labels
        Mismatch = estimated_labels - true_labels
    
        print('\r\nTotal reward: {} | Number of samples: {} | Accuracy = {}%'.format(total_reward,
              len(states),float(100*total_reward/len(states))))
        outputs_df = pd.DataFrame(index = env.attack_types,columns = ["Estimated","Correct","Total","Acuracy"])
        for indx,att in enumerate(env.attack_types):
           outputs_df.iloc[indx].Estimated = estimated_labels[indx]
           outputs_df.iloc[indx].Correct = estimated_correct_labels[indx]
           outputs_df.iloc[indx].Total = true_labels[indx]
           outputs_df.iloc[indx].Acuracy = Accuracy[indx]*100
           outputs_df.iloc[indx].Mismatch = abs(Mismatch[indx])
    

    
    print(outputs_df)
    
    #%%
    
    fig, ax = plt.subplots()
    width = 0.35
    pos = np.arange(len(true_labels))
    p1 = plt.bar(pos, estimated_correct_labels,width,color='g')
    p1 = plt.bar(pos+width,
                 (np.abs(estimated_correct_labels-true_labels)),width,
                 color='r')
    p2 = plt.bar(pos+width,np.abs(estimated_labels-estimated_correct_labels),width,
                 bottom=(np.abs(estimated_correct_labels-true_labels)),
                 color='b')

    
    ax.set_xticks(pos+width/2)
    ax.set_xticklabels(env.attack_types,rotation='vertical')
    #ax.set_yscale('log')

    #ax.set_ylim([0, 100])
    ax.set_title('Test set scores, Acc = {:.2f}'.format(100*total_reward/len(states)))
    plt.legend(('Correct estimated','False negative','False positive'))
    #plt.show()
    plt.savefig('results/test_dueling_network.svg', format='svg', dpi=1000)

    #%% Agregated precision

    aggregated_data_test = np.argmax(labels.values,axis=1)
    
    print('Performance measures on Test data')
    print('Accuracy =  {}'.format(accuracy_score( aggregated_data_test,actions)))
    print('F1 =  {}'.format(f1_score(aggregated_data_test,actions, average='weighted')))
    print('Precision_score =  {}'.format(precision_score(aggregated_data_test,actions, average='weighted')))
    print('recall_score =  {}'.format(recall_score(aggregated_data_test,actions, average='weighted')))
    
    cnf_matrix = confusion_matrix(aggregated_data_test,actions)
    np.set_printoptions(precision=2)
    plt.figure()
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=env.attack_types, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig('results/confusion_matrix_type_dueling.svg', format='svg', dpi=1000)









