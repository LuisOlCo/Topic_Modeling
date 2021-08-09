import json
import os
from datetime import datetime


def create_experiment(period,name):
    '''
    Creates a new folder to store all the sentence embeddings of the new experiment configuration
    '''
    path = os.path.join(os.getcwd(), 'Experiments')
    path_new_folder_name = name +  '_' + period
    new_experiment_path = os.path.join(path,path_new_folder_name)
    if not os.path.isdir(new_experiment_path):
        print('New Experiment, creating folder...')
        os.mkdir(new_experiment_path)
    return new_experiment_path

def save_sentence_embeddings(experiment_path,embeddings, period):
    '''
    Save sentence embeddings in json file
    '''
    name = transfrom_timestamp_2_string(period)
    file_name = name + '.json'
    full_path_file = os.path.join(experiment_path,file_name)
    with open(full_path_file,'w') as file:
        json.dump(embeddings,file)
    return "Files saved"

def transfrom_timestamp_2_string(period):
    '''
    Transform the timestamp to string to storage it easily
    '''
    date_start = "_".join([period[0].strftime("%Y"),period[0].strftime("%m"),period[0].strftime("%d")])
    date_end = "_".join([period[1].strftime("%Y"),period[1].strftime("%m"),period[1].strftime("%d")])
    return (date_start + '__' + date_end)

def load_sentence_embeddings(file):
    '''
    Load a json file with the sentence embeddings, it should receive the file path
    '''
    with open(file, 'r') as fp:
        data = json.load(fp)
    return data

def save_tweet_id_cluster_df(experiment_path,df,period):
    '''
    Saves the dataframe with the tweets and their corresponding cluster to be used later on the topic modeling
    '''
    name = transfrom_timestamp_2_string(period) + '.csv'
    file_path = os.path.join(experiment_path,name)
    df.to_csv(file_path)
    return 'CSV file saved'

def save_topics(experiment_path,topics):
    '''
    Save topics in json file
    '''
    full_path_file = os.path.join(experiment_path,'topics.json')
    with open(full_path_file,'w') as file:
        json.dump(topics,file)
    return "Files saved"
