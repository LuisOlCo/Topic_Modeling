import pandas as pd
import torch
import re
import string
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import os
from tqdm import tqdm

MAIN_FOLDER = '/home/luis/Data/depression_dataset/'
USERS_TABLE_PATH_FILE = '/home/luis/Data/depression_dataset/users_info_table_all.csv'
DEPRESSED_TWEET_PATH_FILE = '/home/luis/Data/depression_dataset/depressed_users/Users_tweets'
CONTROL_TWEET_PATH_FILE = '/home/luis/Data/depression_dataset/control_users/Users_tweets'


class TotalData():
    '''
    Class object containing all the data. This object feeds once at time the
    data corresponding to the period of time to the model
    @Input: path of the csv file containing all the tweets
    @Output: Pytorch dataset object with all the tweets within a certain period of time
    '''
    def __init__(self,path_file):

        self.date_ref = pd.Timestamp(2020,2,1)
        self.path_file = path_file
        self.data = pd.read_csv(self.path_file)

        #self.data = self.data[self.data['English']==True]
        #self.data = self.data.drop(labels=['Unnamed: 0'],axis=1)

        # Delete rows with nan in date at text columns
        self.data = self.data[self.data['date'].notna()]
        self.data = self.data[self.data['text'].notna()]
        self.data['period'] = self.data['date'].apply(lambda row: row.split('-')[1] + '_' + row.split('-')[0])

        self.data['date'] = self.data['date'].apply(lambda cur_date: cur_date.split(' ')[0])
        self.data['date'] = pd.to_datetime(self.data['date'],format='%Y-%m-%d')

        # Filter out those tweets before the reference date
        self.data = self.data[self.data['date'] > self.date_ref]

        # List of all period groups in our dataset
        self.periods = self.data['period'].unique()

    def get_chunk_tweets(self,period):
        '''
        Returns a pandas dataframe with all the tweets of the current period that we are studying
        '''
        data_period = self.data[self.data['period'] == period]
        data_period = data_period.reset_index(drop=True)
        return data_period


class DatasetPandas2Torch(Dataset):
    '''
    Class object Dataset to pass data from pandas to Pytorch Dataloader
    @Input: Pandas dataframe
    @Output: DataLoader calls with method __getitem__ to retrieve samples of the dataframe
    '''
    def __init__(self,data):
        #self.data = pd.read_csv(data, index_col = 0)
        self.data = data

    def __len__(self):
        return (len(self.data))

    #remove the emoji
    def deEmojify(self,inputString):
        return inputString.encode('ascii', 'ignore').decode('ascii')

    def preprocess(self,tweet):
        '''
        Method with a set of commands to clean tweets
        '''
        tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())
        # Convert all text to lowercase
        #data['tweet'] = data['tweet'].apply(lambda sentence: sentence.lower())
        # remove symbols, exclamation marks... --> '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        tweet = re.sub('[%s]' % re.escape(string.punctuation), '', tweet)
        # remove line breaks special characters
        tweet = re.sub('[\t\n\r\f\v]' , '', str(tweet))
        tweet = tweet.replace(u'\xa0', u' ')
        # Remove URL from tweets
        tweet = re.sub(r"http\S+", "", tweet)
        # Remove hashtags
        tweet = re.sub(r"#\S+", "", tweet)
        # Remove pic links
        tweet = re.sub(r'pic.twitter.com/[\w]*',"", tweet)
        # Remove emojis
        tweet = self.deEmojify(tweet)
        # Substitute multiple white spaces characters for one
        tweet = re.sub(' +' , ' ', tweet)
        return tweet.strip()

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.preprocess(self.data['text'][idx]), self.data['tweet_id'][idx]
