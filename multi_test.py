import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import model_from_json
from mAD import RLenv


if __name__ == "__main__":
    
    kdd_10_path = 'datasets/kddcup.data_10_percent_corrected'


    with open("multi_model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("multi_model.h5")
    model.compile("sgd", "mse")

    # Define environment, game
    env = RLenv(kdd_10_path)
    

    total_reward = 0    
    #epochs = int(env.state_shape[0]/env.batch_size)
    epochs = 10
    true_labels = np.zeros(len(env.attack_names))
    estimated_labels = np.zeros(len(env.attack_names))
    
    for e in range(epochs):
        states , labels = env.get_batch(batch_size = env.batch_size)
        q = model.predict(states)
        actions = np.argmax(q,axis=1)        
        
        reward = np.zeros(env.batch_size)
        
        for indx,a in enumerate(actions):
            true_labels[indx] +=1
            if a == np.argmax(labels.iloc[indx].values):
                reward[indx] = 1
                estimated_labels[indx] +=1              
        
       
        total_reward += int(sum(reward))
        
        print("\rEpoch {}/{} | Tot Rew -- > {}".format(e,epochs,total_reward), end="")
 
    print('\r\nTotal reward: {} | Number of samples: {} | Acuracy = {}%'.format(total_reward,
          int(epochs*env.batch_size),float(100*total_reward/(epochs*env.batch_size))))
    outputs_df = pd.DataFrame(index = env.attack_names,columns = ["Clasification","Acuracy"])
    for indx,att in enumerate(env.attack_names):
       outputs_df.iloc[indx].Clasification = "{}/{}".format(estimated_labels[indx],true_labels[indx])
       outputs_df.iloc[indx].Acuracy = 100*estimated_labels[indx]/true_labels[indx]

        
    print(outputs_df)
    
        #plt.imshow(input_t.reshape((grid_size,)*2),
        #           interpolation='none', cmap='gray')
        #plt.savefig("error.png")

