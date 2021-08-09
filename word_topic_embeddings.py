import os
import argparse
import pickle
import numpy as np

import gensim.models


def load_w2v_model(w2v_file_path):
    '''Load pre-trained word embedding in the our twitter dataset'''
    return gensim.models.Word2Vec.load(w2v_file_path)

def load_tf_idf_weights(tf_idf_weights_path_file):
    '''
    Return TF IDF weights for each of the topics
    Structure --> weights = {period:{topic1:{word1:tfidf_weight,word2:tfidf_weight,...},
                                     topic2:{word1:tfidf_weight,word2:tfidf_weight,...},
                                     ...
                                     idex_to_token: {idx1:word1,idx2:word2,...}},
                            period:{...}}
    '''
    with open(tf_idf_weights_path_file,'rb') as handle:
        tf_idf_weights = pickle.load(handle)
    return tf_idf_weights

def save_topics(path_file,embeddings):
    '''Save the embedding dictionary'''
    full_path_file = os.path.join(path_file,'topic_embeddings.pickle')
    with open(full_path_file,'wb') as handle:
        pickle.dump(embeddings,handle)
    handle.close()

def main(w2v_file_path,tf_idf_weights_path_file,new_embed_path_file,topn,dim):

    print('Loading Word2Vec model trained on dataset...')
    w2v_model = load_w2v_model(w2v_file_path)

    print('Loading TF IDF weights for each of the topics...')
    tfidf_weights =  load_tf_idf_weights(tf_idf_weights_path_file)
    error_words = []
    embeddings = {}
    for period in (tfidf_weights.keys()):
        # Each period has a idx2token dictionary
        idx2token = tfidf_weights[period]['idex_to_token']
        # for each period we compute the embedding for each of the topics in the period
        for topic in tfidf_weights[period]:
            if topic != 'idex_to_token':
                topic_embedding = np.zeros(dim)
                # get the idx and the tfidf weight of the topn words for each topic
                topic_weights = sorted(tfidf_weights[period][topic],key = lambda x: x[1],reverse=True)[:topn]
                # Compute the topic embedding
                for i,(idx,weight) in enumerate(topic_weights):
                    word = idx2token[idx]
                    try:
                        # Create the topic embedding in combination of the topn words
                        if i < topn:
                            topic_embedding += weight*w2v_model.wv[word]
                        # Fill the embedding model 3 times the words used for creating the topic embedding
                        embeddings[word] = w2v_model.wv[word]
                    except:
                        error_words.append(word)
                    if i > 3*topn:
                        break
                # Add the topic embedding once we have evaluated all the words for a topic
                embeddings[period + '_' + str(topic)] = topic_embedding
    # Once all the period have been evaluated we save the embeddings in a pickle file
    save_topics(new_embed_path_file,embeddings)
    print(error_words)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Compute Topic Embeddings')
    parser.add_argument('--embed_path_file', '-epf', default='/home/luis/Topic_Modeling/Sentence_BERT/own_word_embeddings/word_embeddings_tweet_model', type=str, help='TPath where the word2vec is saved')
    parser.add_argument('--tf_idf_weights_path_file', '-tfidfpf', default='/home/luis/Topic_Modeling/Sentence_BERT/Experiments/groups_separated/tf_idf_weights_potts.pickle',type=str, help='Name of the experiment')
    parser.add_argument('--new_embed_path_file', '-nepf', default='/home/luis/Topic_Modeling/Sentence_BERT/Experiments/groups_separated/',type=str, help='Path file to save the topic embeddings')
    parser.add_argument('--top_n_words', '-topn', default=10,type=int, help='Top n of words to consider for creating the topic embedding')
    parser.add_argument('--embed_dim', '-dim', default=100,type=int, help='Dimension of the from the pre-trained word embeddings')
    args = parser.parse_args()

    main(args.embed_path_file,args.tf_idf_weights_path_file,args.new_embed_path_file,args.top_n_words,args.embed_dim)
