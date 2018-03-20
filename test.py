import json
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json
from AD import RLenv


if __name__ == "__main__":
    
    kdd_10_path = 'datasets/kddcup.data_10_percent_corrected'


    with open("model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("model.h5")
    model.compile("sgd", "mse")

    # Define environment, game
    env = RLenv(kdd_10_path)
    
    state = env.reset()
    ones = 0
    zeros = 0
    total_reward = 0    
    for e in range(env.data.shape[0]):
        loss = 0.
        
        q = model.predict(state)
        action = np.argmax(q[0])
        if action == 1:
            ones += 1
        else:
            zeros += 1
        next_state, reward, done = env.act(action)
        if reward == 1:
            total_reward +=1
        
        print("\rEpoch {}/{} | Ones/Zeros: {}/{}  Tot Rew -- > {}".format(e,env.data.shape[0],ones,zeros,total_reward), end="")
 
    print('\r\nTotal reward: {} | Number of samples: {} | Probability: {}'.format(total_reward,
          env.data.shape[0],float(total_reward/env.data.shape[0])))
        #plt.imshow(input_t.reshape((grid_size,)*2),
        #           interpolation='none', cmap='gray')
        #plt.savefig("error.png")

