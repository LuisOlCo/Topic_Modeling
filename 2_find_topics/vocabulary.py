from collections import Counter
import numpy as np

class Vocabulary():

    def __init__(self):
        self.counter_global = {}
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.length = 0
        self.cluster_counter = {}
        self.most_common_words_dict = {}

    def update_vocabulary(self,token,cluster):
        '''
        Updates the vocabulary and the counters given one token
        '''
        self._add_token(token)
        self._update_counter(token,self.counter_global)
        self._update_cluster_counter(token,cluster)

    def _add_token(self,token):
        '''
        Builds a dictionary with all the words in the dataset
        '''
        if token not in self.token_to_idx:
            self.token_to_idx.update({token:self.length})
            self.idx_to_token.update({self.length:token})
            self.length += 1

    def _update_counter(self,token,counter):
        '''
        Tracks the frequency of each word in the dataset
        '''
        if token in counter:
            counter[token] += 1
        else:
            counter.update({token:1})

    def _update_cluster_counter(self,token,cluster):
        '''
        Builds a dictionary tracking the frequency of words in each chapter
        '''
        if cluster in self.cluster_counter:
            self._update_counter(token,self.cluster_counter[cluster])
        else:
            self.cluster_counter.update({cluster:{}})

    def most_common_words_per_chapter(self,clusters,n):
        '''
        Returns a pie chart for every cluster with the top n most common words
        '''
        # list of tuples with the most common word and their frequencies
        for cluster in clusters:
            words = Counter(self.cluster_counter[cluster]).most_common(n)
            words_dict = {}
            for word in words:
                words_dict[word[0]] = word[1]

            # from the tuple to a dictionary
            top_words = []
            freqs = []
            for x, y in words_dict.items():
                top_words.append(x)
                freqs.append(y)

            # plot
            plt.figure(figsize=(30,10))
            plt.title('Cluster: %d' %cluster, fontsize=30)
            plt.pie(freqs, labels=top_words,wedgeprops=dict(width=.7), autopct='%1.0f%%', startangle= -20, textprops={'fontsize': 30})
            plt.axis('equal')
            plt.show()
