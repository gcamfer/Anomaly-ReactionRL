import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import itertools

from my_enviroment import my_env

from estimators import ValueEstimator, PolicyEstimator

kdd_train = '../../datasets/NSL/KDDTrain+.txt'
kdd_test = '../../datasets/NSL/KDDTest+.txt'

formated_train_path = "../../datasets/formated/formated_train_type.data"
formated_test_path = "../../datasets/formated/formated_test_type.data"

#TEST
batch_size = 1
env = my_env('test',formated_test_path = formated_test_path,batch_size=batch_size) 
total_reward = 0    
epochs = int(env.data_shape[0]/env.batch_size/1)

true_labels = np.zeros(len(env.attack_types),dtype=int)
estimated_labels = np.zeros(len(env.attack_types),dtype=int)
estimated_correct_labels = np.zeros(len(env.attack_types),dtype=int)

with tf.Session() as sess:
    model = tf.train.import_meta_graph('/tmp/my-model.meta')
    
    
    for e in range(epochs):
        #states , labels = env.get_sequential_batch(test_path,batch_size = env.batch_size)
        states , labels = env.get_batch(batch_size)
        
        # TODO: fix performance in this loop
        action_probs = policy_estimator.predict(state.reshape([batch_size,len(state)]))[0]
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        reward = np.zeros(env.batch_size)
        
        true_labels += np.sum(labels).values
    
        estimated_labels[action] +=1              
        if action == np.argmax(labels.values):
            reward = 1
            estimated_correct_labels[action] += 1
        
        
        total_reward += reward
        print("\rEpoch {}/{} | Tot Rew -- > {}".format(e,epochs,total_reward), end="")
        
    Accuracy = estimated_correct_labels / true_labels
    Mismatch = estimated_labels - true_labels
    
    print('\r\nTotal reward: {} | Number of samples: {} | Accuracy = {}%'.format(total_reward,
          int(epochs*env.batch_size),float(100*total_reward/(epochs*env.batch_size))))
    outputs_df = pd.DataFrame(index = env.attack_types,columns = ["Estimated","Correct","Total","Acuracy"])

for indx,att in enumerate(env.attack_types):
   outputs_df.iloc[indx].Estimated = estimated_labels[indx]
   outputs_df.iloc[indx].Correct = estimated_correct_labels[indx]
   outputs_df.iloc[indx].Total = true_labels[indx]
   outputs_df.iloc[indx].Acuracy = Accuracy[indx]*100
   outputs_df.iloc[indx].Mismatch = abs(Mismatch[indx])


    
print(outputs_df)

#%%



ind = np.arange(1,len(env.attack_types)+1)
fig, ax = plt.subplots()
width = 0.35
p1 = plt.bar(ind, estimated_correct_labels,width,color='g')
p2 = plt.bar(ind, 
             (np.abs(estimated_correct_labels-true_labels)\
              +np.abs(estimated_labels-estimated_correct_labels)),width,
             bottom=estimated_correct_labels,color='r')

    
ax.set_xticks(ind)
ax.set_xticklabels(env.attack_types,rotation='vertical')
#ax.set_yscale('log')

#ax.set_ylim([0, 100])
ax.set_title('Test set scores')
plt.legend((p1[0], p2[0]), ('Correct estimated', 'Incorrect estimated'))
plt.tight_layout()
#plt.show()
plt.savefig('../results/AC_test_type.eps', format='eps', dpi=1000)
