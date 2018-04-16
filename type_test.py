import json
import numpy as np
import pandas as pd
from keras.models import model_from_json
from typeAD import RLenv


if __name__ == "__main__":
    batch_size = 10
    test_path = '../datasets/formated/test_data_type.data'


    with open("models/type_model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("models/type_model.h5")
    model.compile("sgd", "mse")

    # Define environment, game, make sure the batch_size is the same in train
    env = RLenv(test_path,'test',batch_size)
    

    total_reward = 0    
    epochs = int(env.data_shape[0]/env.batch_size/4)
    
    
    true_labels = np.zeros(len(env.attack_types),dtype=int)
    estimated_labels = np.zeros(len(env.attack_types),dtype=int)
    estimated_correct_labels = np.zeros(len(env.attack_types),dtype=int)
    
    for e in range(epochs):
        #states , labels = env.get_sequential_batch(test_path,batch_size = env.batch_size)
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
    
        #plt.imshow(input_t.reshape((grid_size,)*2),
        #           interpolation='none', cmap='gray')
        #plt.savefig("error.png")

