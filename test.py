import json
import numpy as np
from keras.models import model_from_json
from AD import RLenv


if __name__ == "__main__":
    
    test_path = '../datasets/corrected'
    #test_path = '../datasets/KDDTest+.txt'

    with open("model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("model.h5")
    model.compile("sgd", "mse")

    batch_size = 10
    # Define environment, game
    env = RLenv(test_path,'test',batch_size)
    
    
    ones = 0
    zeros = 0
    total_reward = 0    
    epochs = int(env.state_shape[0]/env.batch_size)
    for e in range(epochs):
        states , labels = env.get_batch(batch_size = env.batch_size)
        q = model.predict(states)
        actions = np.argmax(q,axis=1)        
        
        reward = np.zeros(env.batch_size)
        for indx,a in enumerate(actions):
            if a == labels[indx]:
                reward[indx] = 1
        
        ones += int(sum(actions))
        zeros += env.batch_size - int(sum(actions))
        total_reward += int(sum(reward))
        
        print("\rEpoch {}/{} | Ones/Zeros: {}/{}  Tot Rew -- > {}".format(e,epochs,ones,zeros,total_reward), end="")
 
    print('\r\nTotal reward: {} | Number of samples: {} | Acuracy = {}%'.format(total_reward,
          int(epochs*env.batch_size),float(100*total_reward/(epochs*env.batch_size))))
        #plt.imshow(input_t.reshape((grid_size,)*2),
        #           interpolation='none', cmap='gray')
        #plt.savefig("error.png")

