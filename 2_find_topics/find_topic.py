import pickle
import os
import pandas as pd
from tqdm import tqdm
import re
import string

import spacy
from gensim import corpora, models

from tokenizer_potts import TweetTokenizer
from vocabulary import *


class Topics():
    '''
    Class object that find the topics in each of the clusters
    Creates a BoW with the most relevant words representing the cluster
    '''

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        #self.all_stopwords = self.nlp.Defaults.stop_words
        self.all_stopwords = self._get_stopwords()
        self.topics = {}
        self.topics_matrix = {}
        self.tokenizer = TweetTokenizer()

    def _get_stopwords(self):
        '''Loads personalize list of stopwords mixed of spacy stopwords and experience'''
        stopwords = []
        with open('stopwords.txt','r') as f:
            for line in f:
                stopwords.append(line.strip())
        return stopwords

    def compute_topics_tfidf(self,data,period):
        '''
        Given a set of tweet for a determined period of time and with their cluster_id, computes the topics for each of the clusters
        @Input: csv file with all the tweets within a period of time and labeled with its cluster id
        '''
        self.topics_matrix[period] = {}
        vocab = Vocabulary()
        lemma = []

        # Tokenize all the tweets in the file
        print('Tokenizing tweets')
        data['lemma_tok'] = data['text'].apply(lambda tweet: self.tokenizer.tokenize(self._preprocess(str(tweet.encode('unicode-escape'))))[1:])

        # Create the vocabulary for each cluster
        print('Total number of topics: ',len(data['cluster_id'].unique()))
        for cluster in data['cluster_id'].unique():
            mask = data['cluster_id'] == cluster
            data_cluster = data[mask]
            data_cluster['lemma_tok'].apply(lambda sentence: [vocab.update_vocabulary(token,cluster) for token in sentence if token not in self.all_stopwords])

        token_to_idx = vocab.token_to_idx
        idx_to_token = vocab.idx_to_token
        cluster_counter = vocab.cluster_counter

        bow_corpus = self._get_bow_corpus(token_to_idx,cluster_counter)
        tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]

        for i_topic in range(len(corpus_tfidf)):
            self.topics_matrix[period][i_topic] = corpus_tfidf[i_topic]

        #for i_topic in range(len(corpus_tfidf)):
        #    self.topics_matrix[period][i_topic] = self._relevant_words_TDIDF_per_cluster(corpus_tfidf[i_topic],idx_to_token)
        self.topics_matrix[period]['idex_to_token'] = idx_to_token

        return 'All topics computed and saved'

    def _preprocess(self,tweet):
        '''
        Method with a set of commands to clean tweets
        '''
        # Convert all text to lowercase
        #data['tweet'] = data['tweet'].apply(lambda sentence: sentence.lower())
        tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())
        # remove symbols, exclamation marks... --> '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        tweet = re.sub('[%s]' % re.escape(string.punctuation), '', tweet)
        # remove line breaks special characters
        tweet = re.sub('[\t\n\r\f\v]' , '', str(tweet))
        tweet = tweet.replace(u'\xa0', u' ')
        # Remove URL from tweets
        tweet = re.sub(r"http\S+", "", tweet)
        # Remove hashtags
        #tweet = re.sub(r"#\S+", "", tweet)
        # Remove pic links
        tweet = re.sub(r'pic.twitter.com/[\w]*',"", tweet)
        # Substitute multiple white spaces characters for one
        tweet = re.sub(' +' , ' ', tweet)
        return tweet.strip()

    def save_topics(self,path_file):
        '''
        Save the dictionry to make the matrix with the idx_token/topic TF-IDF weights and a dictionary
        to transform index to words
        '''
        full_path_file = os.path.join(path_file,'tf_idf_weights_potts.pickle')
        with open(full_path_file,'wb') as handle:
            pickle.dump(self.topics_matrix,handle)

    def _get_bow_corpus(self,token_to_idx,cluster_counter):
        '''
        Transforms the chapter_counter dictionary into a list of tuple lists,
        this is the format required by Gensim to compute TD-IDF or LDA

        @Input: 1.- Dictionary tokens with their respectives index
                2.- Dictionary with the frequency of each word in each cluster
        '''
        bow_corpus = []
        for cluster in cluster_counter:
            cluster_list = []
            for word in cluster_counter[cluster]:
                cluster_list.append((token_to_idx[word],cluster_counter[cluster][word]))
            bow_corpus.append(cluster_list)
        return bow_corpus

    def _relevant_words_TDIDF_per_cluster(self,corpus_tfidf,idx_to_token,top_n_relevant_words=20):
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
