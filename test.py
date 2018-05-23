import json
import numpy as np
from keras.models import model_from_json
from AD import RLenv
import matplotlib.pyplot as plt



if __name__ == "__main__":
    
    test_path = '../datasets/corrected'
    #test_path = '../datasets/KDDTest+.txt'

    with open("models/model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("models/model.h5")
    model.compile("sgd", "mse")

    batch_size = 10
    # Define environment, game
    env = RLenv(test_path,'test',batch_size)
    
    
    true_ones = 0
    true_zeros = 0
    false_ones = 0
    false_zeros = 0
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
                if a == 0:
                    true_zeros +=1
                else:
                    true_ones +=1
            else:
                if a==0:
                    false_zeros +=1
                else:
                    false_ones +=1

        total_reward += int(sum(reward))
        
        print("\rEpoch {}/{} | Ones/Zeros: {}/{}  Tot Rew -- > {}".format(e,epochs,true_ones,true_zeros,total_reward), end="")
 
    print('\r\nTotal reward: {} | Number of samples: {} | Acuracy = {}%'.format(total_reward,
          int(epochs*env.batch_size),float(100*total_reward/(epochs*env.batch_size))))
        #plt.imshow(input_t.reshape((grid_size,)*2),
        #           interpolation='none', cmap='gray')
        #plt.savefig("error.png")

    ind = np.arange(1,5)
    fig, ax = plt.subplots()
    
    t_o, f_o, t_z ,f_z = plt.bar(ind, (true_ones,false_ones,true_zeros,false_zeros))
    
    t_o.set_facecolor('g')
    f_o.set_facecolor('r')
    t_z.set_facecolor('g')
    f_z.set_facecolor('r')
    
    ax.set_xticks(ind)
    ax.set_xticklabels(['True ones', 'False ones', 'True zeros', 'False zeros'])
   # ax.set_yscale('log')

    #ax.set_ylim([0, 100])
    #ax.set_ylabel('Percent usage')
    ax.set_title('Test set')
    plt.tight_layout()
    #plt.show()
    plt.savefig('results/test_simple_nat.eps', format='eps', dpi=1000)
