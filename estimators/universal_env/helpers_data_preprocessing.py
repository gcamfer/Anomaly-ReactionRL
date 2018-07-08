'''
Data class processing
'''
import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


class data_cls:
    """
    Process the data in order to get states and labels for RL env.
    Args:
        colum_names: list of all parameters in the dataset. Must contain 'label'
        train_path: Relative path in wich the train dataset is located
        test_path: Relative path in wich the test dataset is located
        train_test: If train it'll load formated train, if test it load formated test
        formated_train_path: Relative path in wich the train dataset will be stored
        formated_test_path: Relative path in wich the test dataset will be stored
        attack_types: Names for the final class labels
        attack_map: Maps the labels to each class
    """
    def __init__(self,train_test,attack_map,**kwargs):
        self.column_names = kwargs.get('column_names', ["duration","protocol_type","service","flag","src_bytes",
            "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
            "logged_in","num_compromised","root_shell","su_attempted","num_root",
            "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
            "is_host_login","is_guest_login","count","srv_count","serror_rate",
            "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
            "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
            "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
            "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels"])
        self.index = 0
        self.loaded = False
        
        # Data formated path and test path.
        self.train_test = train_test
        self.train_path = kwargs.get('train_path', '../../datasets/NSL/KDDTrain+.txt')
        self.test_path = kwargs.get('test_path','../../datasets/NSL/KDDTest+.txt')

        # Definitions for the formated outuputs:
        self.formated_train_path = kwargs.get('formated_train_path', 
                                              "../../datasets/formated/uni_formated_train.data")
        self.formated_test_path = kwargs.get('formated_test_path',
                                             "../../datasets/formated/uni_formated_test.data")
        

#        self.attack_types = kwargs.get('attack_types',
#                                       ['normal','DoS','Probe','R2L','U2R'])
        self.attack_map =   attack_map 
        self.attack_types = list(set(attack_map.values()))

        
        formated = False
        
        # Test formated data exists
        if os.path.exists(self.formated_train_path) and os.path.exists(self.formated_test_path):
            formated = True
           
        self.formated_dir = "../../datasets/formated/"
        if not os.path.exists(self.formated_dir):
            os.makedirs(self.formated_dir)
               

        # If it does not exist, it's needed to format the data
        if not formated:
            ''' Formating the dataset for ready-2-use data'''
            self.df = pd.read_csv(self.train_path,sep=',',names=self.column_names,index_col=False,header=0)
            test = pd.read_csv(self.test_path,sep=',',names=self.column_names,index_col=False,header=0)

            train_indx = self.df.shape[0]
            frames = [self.df,test]
            self.df = pd.concat(frames)
            
            
            # Remove labels column
            labels = self.df['labels']
            self.df = self.df.drop('labels',axis=1)
            
            # Remove all solicided columns
            drop_names = self.df.filter(like='drop').columns
            self.df = self.df.drop(drop_names,axis=1)

            # Processing categorical and numerical columns
            num_cols = list(self.df._get_numeric_data().columns)
            cat_cols = list(set(self.df.columns)-set(num_cols))
            

            
            for name_col in cat_cols:
                self.df = pd.concat([self.df.drop(name_col, axis=1), pd.get_dummies(self.df[name_col])], axis=1)
           
            
            # Normalization of the df
            log_cols = self.df.filter(like='logarithm').columns
            nat_cols =  list(set(self.df.columns)-set(log_cols))
#            aux_df = (self.df-self.df.mean())/(self.df.max()-self.df.min())
#            self.df = (self.df)/(self.df.max()-self.df.min())
            self.df[nat_cols] = self.df[nat_cols]/(self.df[nat_cols].max()-self.df[nat_cols].min())
            self.df[log_cols] = np.log(self.df[log_cols])
            self.df[log_cols] = self.df[log_cols]/(self.df[log_cols].max()-self.df[log_cols].min())

            # If na max and min = 0 so delete column
            self.df = self.df.dropna(axis=1)

            # Add labels again
            self.df['labels']=labels
            
            
             # Save data
            test_df = self.df.iloc[train_indx:self.df.shape[0]]
            test_df = shuffle(test_df,random_state=np.random.randint(0,100))
            self.df = self.df[:train_indx]
            self.df = shuffle(self.df,random_state=np.random.randint(0,100))
            test_df.to_csv(self.formated_test_path,sep=',',index=False)
            self.df.to_csv(self.formated_train_path,sep=',',index=False)
            
        
    
    def get_batch(self, batch_size=100):
        
        if self.loaded is False:
            self._load_df()
            
        indexes = list(range(self.index,self.index+batch_size))    
        if max(indexes)>self.data_shape[0]-1:
            dif = max(indexes)-self.data_shape[0]
            indexes[len(indexes)-dif-1:len(indexes)] = list(range(dif+1))
            self.index=batch_size-dif
            batch = self.df.iloc[indexes]
        else: 
            batch = self.df.iloc[indexes]
            self.index += batch_size    
        
        map_type = pd.Series(index=self.attack_types,data=np.arange(len(self.attack_types))).to_dict()
        labels = batch['labels'].map(self.attack_map).map(map_type).values
        del(batch['labels'])
            
        return np.array(batch),labels
    
    def get_full(self):

        self._load_df()
        
        batch = self.df
        map_type = pd.Series(index=self.attack_types,data=np.arange(len(self.attack_types))).to_dict()
        labels = batch['labels'].map(self.attack_map).map(map_type).values
        
        del(batch['labels'])
        
        return np.array(batch),labels
    
    def get_shape(self):
        if self.loaded is False:
            self._load_df()
        
        self.data_shape = self.df.shape
        # stata + labels
        return self.data_shape
    
    def _load_df(self):
        if self.train_test == 'train':
            self.df = pd.read_csv(self.formated_train_path,sep=',') # Read again the csv
        else:
            self.df = pd.read_csv(self.formated_test_path,sep=',')
        self.index=np.random.randint(0,self.df.shape[0]-1,dtype=np.int32)
        self.loaded = True





