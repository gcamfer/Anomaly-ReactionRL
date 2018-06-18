import json
import numpy as np
import pandas as pd
from keras.models import model_from_json
from adversarialAD import RLenv
import matplotlib.pyplot as plt
from adversarialAD import huber_loss

from sklearn.metrics import f1_score


batch_size = 10
formated_test_path = "../../datasets/formated/formated_test_adv.data"

with open("models/defender_agent_model.json", "r") as jfile:
    model = model_from_json(json.load(jfile))
model.load_weights("models/defender_agent_model.h5")

model.compile(loss=huber_loss,optimizer="sgd")

# Define environment, game, make sure the batch_size is the same in train
env = RLenv('test',formated_test_path = formated_test_path)


total_reward = 0    


true_labels = np.zeros(len(env.attack_types),dtype=int)
estimated_labels = np.zeros(len(env.attack_types),dtype=int)
estimated_correct_labels = np.zeros(len(env.attack_types),dtype=int)

#states , labels = env.get_sequential_batch(test_path,batch_size = env.batch_size)
states , labels = env.get_full()
q = model.predict(states)
actions = np.argmax(q,axis=1)        

maped=[]
for indx,label in labels.iterrows():
    maped.append(env.attack_types.index(env.attack_map[label.idxmax()]))

labels,counts = np.unique(maped,return_counts=True)
true_labels[labels] += counts



for indx,a in enumerate(actions):
    estimated_labels[a] +=1              
    if a == maped[indx]:
        total_reward += 1
        estimated_correct_labels[a] += 1


action_dummies = pd.get_dummies(actions)
posible_actions = np.arange(len(env.attack_types))
for non_existing_action in posible_actions:
    if non_existing_action not in action_dummies.columns:
        action_dummies[non_existing_action] = np.uint8(0)
labels_dummies = pd.get_dummies(maped)

normal_f1_score = f1_score(labels_dummies[0].values,action_dummies[0].values)
dos_f1_score = f1_score(labels_dummies[1].values,action_dummies[1].values)
probe_f1_score = f1_score(labels_dummies[2].values,action_dummies[2].values)
r2l_f1_score = f1_score(labels_dummies[3].values,action_dummies[3].values)
u2r_f1_score = f1_score(labels_dummies[4].values,action_dummies[4].values)
    

Accuracy = [normal_f1_score,dos_f1_score,probe_f1_score,r2l_f1_score,u2r_f1_score]
Mismatch = estimated_labels - true_labels

acc = float(100*total_reward/len(states))
print('\r\nTotal reward: {} | Number of samples: {} | Accuracy = {:.2f}%'.format(total_reward,
      len(states),acc))
outputs_df = pd.DataFrame(index = env.attack_types,columns = ["Estimated","Correct","Total","F1_score"])
for indx,att in enumerate(env.attack_types):
   outputs_df.iloc[indx].Estimated = estimated_labels[indx]
   outputs_df.iloc[indx].Correct = estimated_correct_labels[indx]
   outputs_df.iloc[indx].Total = true_labels[indx]
   outputs_df.iloc[indx].F1_score = Accuracy[indx]*100
   outputs_df.iloc[indx].Mismatch = abs(Mismatch[indx])

##############





        

    #%%

print(outputs_df)

fig, ax = plt.subplots()
width = 0.35
pos = np.arange(len(true_labels))
p1 = plt.bar(pos, estimated_correct_labels,width,color='g')
p1 = plt.bar(pos+width,
             (np.abs(estimated_correct_labels-true_labels)\
              +np.abs(estimated_labels-estimated_correct_labels)),width,
             color='r')


ax.set_xticks(pos+width/2)
ax.set_xticklabels(env.attack_types,rotation='vertical')
#ax.set_yscale('log')

#ax.set_ylim([0, 100])
ax.set_title('Test set scores')
plt.legend(('Correct estimated', 'Incorrect estimated'))
plt.tight_layout()
#plt.show()
plt.savefig('results/test_adv.eps', format='eps', dpi=1000)
