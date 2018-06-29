import json
import numpy as np
import pandas as pd
from keras.models import model_from_json
from typeAD import RLenv
import matplotlib.pyplot as plt
from typeAD import huber_loss

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




if __name__ == "__main__":
    formated_test_path = "../../datasets/formated/formated_test_type.data"


    with open("models/type_model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("models/type_model.h5")

    
    model.compile(loss=huber_loss,optimizer="sgd")

    env = RLenv('test',formated_test_path = formated_test_path)
    
    total_reward = 0
    
    true_labels = np.zeros(len(env.attack_types),dtype=int)
    estimated_labels = np.zeros(len(env.attack_types),dtype=int)
    estimated_correct_labels = np.zeros(len(env.attack_types),dtype=int)
    
    states , labels = env.get_full()
    q = model.predict(states)
    actions = np.argmax(q,axis=1)        
    
    
    
    
    
    
    true_labels += np.sum(labels).values

    for indx,a in enumerate(actions):
        estimated_labels[a] +=1              
        if a == np.argmax(labels.iloc[indx].values):
            total_reward += 1
            estimated_correct_labels[a] += 1
    
    
    action_dummies = pd.get_dummies(actions)
    posible_actions = np.arange(len(env.attack_types))
    for non_existing_action in posible_actions:
        if non_existing_action not in action_dummies.columns:
            action_dummies[non_existing_action] = np.uint8(0)
    

    normal_f1_score = f1_score(labels['normal'].values,action_dummies[0].values)
    dos_f1_score = f1_score(labels['DoS'].values,action_dummies[1].values)
    probe_f1_score = f1_score(labels['Probe'].values,action_dummies[2].values)
    r2l_f1_score = f1_score(labels['R2L'].values,action_dummies[3].values)
    u2r_f1_score = f1_score(labels['U2R'].values,action_dummies[4].values)
        
    Accuracy = [normal_f1_score,dos_f1_score,probe_f1_score,r2l_f1_score,u2r_f1_score]
    Mismatch = abs(estimated_correct_labels - true_labels)+abs(estimated_labels-estimated_correct_labels)

    print('\r\nTotal reward: {} | Number of samples: {} | Accuracy = {}%'.format(total_reward,
          len(states),float(100*total_reward/len(states))))
    outputs_df = pd.DataFrame(index = env.attack_types,columns = ["Estimated","Correct","Total","F1_score","Mismatch"])
    for indx,att in enumerate(env.attack_types):
       outputs_df.iloc[indx].Estimated = estimated_labels[indx]
       outputs_df.iloc[indx].Correct = estimated_correct_labels[indx]
       outputs_df.iloc[indx].Total = true_labels[indx]
       outputs_df.iloc[indx].F1_score = Accuracy[indx]*100
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
    plt.tight_layout()
    #plt.show()
    plt.savefig('results/test_type_improved.svg', format='svg', dpi=1000)

    #%% Agregated precision

    aggregated_data_test = np.argmax(labels.values,axis=1)
    
    print('Performance measures on Test data')
    print('Accuracy =  {:.2f}'.format(accuracy_score( aggregated_data_test,actions)))
    print('F1 =  {:.2f}'.format(f1_score(aggregated_data_test,actions, average='weighted')))
    print('Precision_score =  {:.2f}'.format(precision_score(aggregated_data_test,actions, average='weighted')))
    print('recall_score =  {:.2f}'.format(recall_score(aggregated_data_test,actions, average='weighted')))
    
    cnf_matrix = confusion_matrix(aggregated_data_test,actions)
    np.set_printoptions(precision=2)
    plt.figure()
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=env.attack_types, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig('results/confusion_matrix_type_imp.svg', format='svg', dpi=1000)