import json
import os
from datetime import datetime


def create_experiment(name):
    '''
    Creates a new folder to store all the sentence embeddings of the new experiment configuration
    '''

    new_experiment_path = os.path.join('/home/luis/Topic_Modeling/Sentence_BERT', 'Experiments', name)
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
    Transform the timestamp to string format
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

def save_tweet_id_cluster_df(experiment_path,df,period,group):
    '''
    Saves the dataframe with the tweets and their corresponding cluster to be used later on the topic modeling
    '''
    if group != None:
        name = period + '_' + group + '_' + '.csv'
        file_path = os.path.join(experiment_path,name)
    else:
        name = period + '.csv'
        file_path = os.path.join(experiment_path,name)

    df.to_csv(file_path)
    return 'CSV file saved'

def get_bow_corpus(token_to_idx,chapter_counter):
    '''
    Transforms the chapter_counter dictionary into a list of tuple lists,
    this is the format required by Gensim to compute TD-IDF or LDA

    @Input: 1.- Dictionary tokens with their respectives index
            2.- Dictionary with the frequency of each word in each chapter
    '''
    bow_corpus = []
    for chapter in chapter_counter:
        chapter_list = []
        for word in chapter_counter[chapter]:
            chapter_list.append((token_to_idx[word],chapter_counter[chapter][word]))
        bow_corpus.append(chapter_list)
    return bow_corpus

def relevant_words_TDIDF_per_chapter(corpus_tfidf,idx_to_token,top_n_relevant_words=10):
    '''
    Returns back the top n most TD-IDF relevant words in a chapter
    @Input: 1.- number of most relevant words
            2.- TD-IDF values for each chapters
            3.- idx_to_token dictionary to identify the word
    '''
    most_relevant_words = []
    corpus_tfidf.sort(key = lambda x: x[1], reverse = True)
    for item in corpus_tfidf:
        most_relevant_words.append(idx_to_token[item[0]])
        if len(most_relevant_words) > (top_n_relevant_words-1):
            break

    return most_relevant_words
