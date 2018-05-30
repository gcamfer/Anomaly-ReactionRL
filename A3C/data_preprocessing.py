'''
Data class processing
'''

class data_cls:
    def __init__(self, path,train_test,**kwargs):
        col_names = ["duration","protocol_type","service","flag","src_bytes",
            "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
            "logged_in","num_compromised","root_shell","su_attempted","num_root",
            "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
            "is_host_login","is_guest_login","count","srv_count","serror_rate",
            "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
            "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
            "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
            "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels"]
        self.index = 0
        # Data formated path and test path. 
        self.formated_path = "../../datasets/formated/formated_data_type.data"
        self.test_path = "../../datasets/formated/test_data_type.data"
        self.loaded = False
        self.train_test = train_test
        self.second_path = kwargs.get('join_path', '../../datasets/corrected')

        
        self.attack_types = ['normal','DoS','Probe','R2L','U2R']
        self.attack_map =   { 'normal.': 'normal',
                        
                        'back.': 'DoS',
                        'land.': 'DoS',
                        'neptune.': 'DoS',
                        'pod.': 'DoS',
                        'smurf.': 'DoS',
                        'teardrop.': 'DoS',
                        'mailbomb.': 'DoS',
                        'apache2.': 'DoS',
                        'processtable.': 'DoS',
                        'udpstorm.': 'DoS',
                        
                        'ipsweep.': 'Probe',
                        'nmap.': 'Probe',
                        'portsweep.': 'Probe',
                        'satan.': 'Probe',
                        'mscan.': 'Probe',
                        'saint.': 'Probe',
                    
                        'ftp_write.': 'R2L',
                        'guess_passwd.': 'R2L',
                        'imap.': 'R2L',
                        'multihop.': 'R2L',
                        'phf.': 'R2L',
                        'spy.': 'R2L',
                        'warezclient.': 'R2L',
                        'warezmaster.': 'R2L',
                        'sendmail.': 'R2L',
                        'named.': 'R2L',
                        'snmpgetattack.': 'R2L',
                        'snmpguess.': 'R2L',
                        'xlock.': 'R2L',
                        'xsnoop.': 'R2L',
                        'worm.': 'R2L',
                        
                        'buffer_overflow.': 'U2R',
                        'loadmodule.': 'U2R',
                        'perl.': 'U2R',
                        'rootkit.': 'U2R',
                        'httptunnel.': 'U2R',
                        'ps.': 'U2R',    
                        'sqlattack.': 'U2R',
                        'xterm.': 'U2R'
                    }
        
        # If path is not provided system out error
        if (not path):
            print("Path: not path name provided", flush = True)
            sys.exit(0)
        formated = False     
        
        
        if os.path.exists(self.formated_path) and train_test=="train":
            formated = True
        elif os.path.exists(self.test_path) and train_test=="test":
            formated = True
        elif os.path.exists(self.test_path) and os.path.exists(self.formated_path) and (train_test == 'full' or train_test=='join'):
            formated = True
            
       
        # If the formatted data path exists, is not needed to process it again
        if os.path.exists(self.formated_path):
            formated = True
            

        # If it does not exist, it's needed to format the data
        if not formated:
            ''' Formating the dataset for ready-2-use data'''
            self.df = pd.read_csv(path,sep=',',names=col_names)
            if 'dificulty' in self.df.columns:
                self.df.drop('dificulty', axis=1, inplace=True) #in case of difficulty     
                
            if train_test == 'join':
                data2 = pd.read_csv(self.second_path,sep=',',names=col_names)
                if 'dificulty' in data2:
                    del(data2['dificulty'])
                train_indx = self.df.shape[0]
                frames = [self.df,data2]
                self.df = pd.concat(frames)
            # Data now is in RAM
            self.loaded = True
            
            # Dataframe processing
            self.df = pd.concat([self.df.drop('protocol_type', axis=1), pd.get_dummies(self.df['protocol_type'])], axis=1)
            self.df = pd.concat([self.df.drop('service', axis=1), pd.get_dummies(self.df['service'])], axis=1)
            self.df = pd.concat([self.df.drop('flag', axis=1), pd.get_dummies(self.df['flag'])], axis=1)
              
            
            # 1 if ``su root'' command attempted; 0 otherwise 
            self.df['su_attempted'] = self.df['su_attempted'].replace(2.0, 0.0)
            
            # Normalization of the df
            #normalized_df=(df-df.mean())/df.std()
            for indx,dtype in self.df.dtypes.iteritems():
                if dtype == 'float64' or dtype == 'int64':
                    if self.df[indx].max() == 0 and self.df[indx].min()== 0:
                        self.df[indx] = 0
                    else:
                        self.df[indx] = (self.df[indx]-self.df[indx].min())/(self.df[indx].max()-self.df[indx].min())
                    
                      
            # One-hot-Encoding for reaction.  
            all_labels = self.df['labels'] # Get all labels in df
            mapped_labels = np.vectorize(self.attack_map.get)(all_labels) # Map attacks
            self.df = self.df.reset_index(drop=True)
            self.df = pd.concat([self.df.drop('labels', axis=1),pd.get_dummies(mapped_labels)], axis=1)
            
             # Save data
            # suffle data: if join shuffled before in order to save train/test
            if train_test != 'join':
                self.df = shuffle(self.df,random_state=np.random.randint(0,100))            
            
           
            if train_test == 'train':
                self.df.to_csv(self.formated_path,sep=',',index=False)
            elif train_test == 'test':
                self.df.to_csv(self.test_path,sep=',',index=False)
            elif train_test == 'full':
            # 70% train 30% test
                train_indx = np.int32(self.df.shape[0]*0.7)
                test_df = self.df.iloc[train_indx:self.df.shape[0]]
                self.df = self.df[:train_indx]
                test_df.to_csv(self.test_path,sep=',',index=False)
                self.df.to_csv(self.formated_path,sep=',',index=False)
            else: #join: index calculated before
                test_df = self.df.iloc[train_indx:self.df.shape[0]]
                test_df = shuffle(test_df,random_state=np.random.randint(0,100))
                self.df = self.df[:train_indx]
                self.df = shuffle(self.df,random_state=np.random.randint(0,100))
                test_df.to_csv(self.test_path,sep=',',index=False)
                self.df.to_csv(self.formated_path,sep=',',index=False)
            
            
        
    ''' Get n-row batch from the dataset
        Return: df = n-rows
                labels = correct labels for detection 
    Sequential for largest datasets
    '''
    def get_sequential_batch(self, batch_size=100):
        if self.loaded is False:
            self.df = pd.read_csv(self.formated_path,sep=',', nrows = batch_size)
            self.loaded = True
        else:
            self.df = pd.read_csv(self.formated_path,sep=',', nrows = batch_size,
                         skiprows = self.index)
        
        self.index += batch_size

        labels = self.df[self.attack_types]
        for att in self.attack_types:
            del(self.df[att])
        return self.df,labels
    
    
    ''' Get n-rows from loaded data 
        The dataset must be loaded in RAM
    '''
    def get_batch(self, batch_size=100):
        
        if self.loaded is False:
            self.df = pd.read_csv(self.formated_path,sep=',') # Read again the csv
            self.loaded = True
            #self.headers = list(self.df)
        
        batch = self.df.iloc[self.index:self.index+batch_size]
        self.index += batch_size
        labels = batch[self.attack_types]
        
        for att in self.attack_types:
            del(batch[att])
        return batch,labels
    
    
    
    def get_shape(self):
        if self.loaded is False:
            self._load_df()
        
        self.data_shape = self.df.shape
        # stata + labels
        return self.data_shape
    
    def _load_df(self):
        if self.train_test == 'train' or self.train_test == 'full':
            self.df = pd.read_csv(self.formated_path,sep=',') # Read again the csv
        else:
            self.df = pd.read_csv(self.test_path,sep=',')
        self.loaded = True





