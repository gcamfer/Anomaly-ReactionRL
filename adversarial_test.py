import json
import numpy as np
import pandas as pd
from keras.models import model_from_json
from adversarialAD import RLenv
import matplotlib.pyplot as plt
from adversarialAD import huber_loss



if __name__ == "__main__":
    batch_size = 10
    formated_test_path = "../datasets/formated/formated_test_adv.data"

    with open("models/defender_agent_model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("models/defender_agent_model.h5")
    
    model.compile(loss=huber_loss,optimizer="sgd")

    # Define environment, game, make sure the batch_size is the same in train
    env = RLenv('test',formated_test_path = formated_test_path,batch_size=batch_size)
    

    total_reward = 0    
    epochs = int(env.data_shape[0]/env.batch_size/1)
    
    
    true_labels = np.zeros(len(env.attack_types),dtype=int)
    estimated_labels = np.zeros(len(env.attack_types),dtype=int)
    estimated_correct_labels = np.zeros(len(env.attack_types),dtype=int)
    
    for e in range(epochs):
        #states , labels = env.get_sequential_batch(test_path,batch_size = env.batch_size)
        states , labels = env.get_batch(batch_size = env.batch_size)
        q = model.predict(states)
        actions = np.argmax(q,axis=1)        
        
        reward = np.zeros(env.batch_size)
        maped=[]
        for indx,label in labels.iterrows():
            maped.append(env.attack_types.index(env.attack_map[label.idxmax()]))
        
        labels,counts = np.unique(maped,return_counts=True)
        true_labels[labels] += counts
        


        for indx,a in enumerate(actions):
            estimated_labels[a] +=1              
            if a == maped[indx]:
                reward[indx] = 1
                estimated_correct_labels[a] += 1
        
        
        total_reward += int(sum(reward))
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
    plt.savefig('results/test_type.eps', format='eps', dpi=1000)
