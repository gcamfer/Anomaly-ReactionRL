import json
import numpy as np
import pandas as pd
from keras.models import model_from_json
from mAD import RLenv


if __name__ == "__main__":
    batch_size = 10
    kdd_10_path = 'datasets/kddcup.data_10_percent_corrected'
    kdd_path = '../datasets/kddcup.data'
    test_path = '../datasets/formated_multiple_test_data.data'


    with open("multi_model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("multi_model.h5")
    model.compile("sgd", "mse")

    # Define environment, game, make sure the batch_size is the same in train
    env = RLenv(test_path,batch_size)
    

    total_reward = 0    
    epochs = int(env.data_shape[0]/env.batch_size/2)
    
    true_labels = np.zeros(len(env.attack_names),dtype=int)
    estimated_labels = np.zeros(len(env.attack_names),dtype=int)
    estimated_correct_labels = np.zeros(len(env.attack_names),dtype=int)
    
    for e in range(epochs):
        states , labels = env.get_batch(batch_size = env.batch_size)
        q = model.predict(states)
        actions = np.argmax(q,axis=1)        
        
        reward = np.zeros(env.batch_size)
        
        true_labels += np.sum(labels).values

        for indx,a in enumerate(actions):
            estimated_labels[a] +=1              
            if a == np.argmax(labels.iloc[indx].values):
                reward[indx] = 1
                estimated_correct_labels[a] += 1
        
        
        total_reward += int(sum(reward))
        print("\rEpoch {}/{} | Tot Rew -- > {}".format(e,epochs,total_reward), end="")
        
    Err = estimated_correct_labels / true_labels
    print('\r\nTotal reward: {} | Number of samples: {} | Error = {}%'.format(total_reward,
          int(epochs*env.batch_size),float(100*total_reward/(epochs*env.batch_size))))
    outputs_df = pd.DataFrame(index = env.attack_names,columns = ["Clasification","Acuracy"])
    for indx,att in enumerate(env.attack_names):
       outputs_df.iloc[indx].Clasification = "{}/{}".format(estimated_correct_labels[indx],true_labels[indx])
       outputs_df.iloc[indx].Acuracy = Err[indx]*100

        
    print(outputs_df)
    
        #plt.imshow(input_t.reshape((grid_size,)*2),
        #           interpolation='none', cmap='gray')
        #plt.savefig("error.png")

