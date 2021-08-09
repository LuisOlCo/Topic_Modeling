import pandas as pd
import os
import pickle
from tqdm import tqdm
import argparse


def get_topic_words(df,n=10):
    with open('tf_idf_weights.pickle','rb') as handle:
        tf_idf_weights = pickle.load(handle)

    df['topic_words'] = None
    for i in tqdm(range(len(df))):
        period = df.loc[i]['period']
        cur_topic = int(df.loc[i]['topic'])

        tf_idf_weights[period][cur_topic].sort(key=lambda x:x[1],reverse=True)
        topic_words_idx_vals = tf_idf_weights[period][cur_topic][:n]
        # we have to transform from index to token
        topic_words = [tf_idf_weights[period]['idex_to_token'][idx[0]] for idx in topic_words_idx_vals]
        df.at[i,'topic_words'] = topic_words

    return df


def get_distributions():

    files = os.listdir()
    dist = pd.DataFrame()
    for file in tqdm(files):

        if '.csv' in file:
            data = pd.read_csv(file)
            period = file.replace('.csv','')
            data = data[['cluster_id','label']]
            clusters = data['cluster_id'].unique()
            n_depressed = len(data[data['label']==1])
            n_control = len(data[data['label']==0])

            for cluster in clusters:
                data_fil = data[data['cluster_id']==cluster]
                data_fil_depressed = data_fil[data_fil['label']==1]
                data_fil_control = data_fil[data_fil['label']==0]
                row_cluster = {}
                row_cluster['topic'] = cluster
                if n_depressed > 0:
                    row_cluster['depressed'] = len(data_fil_depressed)/n_depressed
                else:
                    row_cluster['depressed'] = 0
                if n_control > 0:
                    row_cluster['control'] = len(data_fil_control)/n_control
                else:
                    row_cluster['control'] = 0
                row_cluster['period'] = period
                dist = dist.append(row_cluster,ignore_index=True)

    return dist

def main():
    return None


if __name__='__main__':

    parser = argparse.ArgumentParser('Sentece-BERT Topic Model Evaluation')
    parser.add_argument('--data_path_file',type=str,default='',help='Path where all the csv files for each period are and the tf_idf_weight matrix is')
    parser.add_argument('--aux_data_path_file',type=str,default='',help='Folder containing the dictionary and docs files')
    parser.add_argument('--embed_path_file',type=str,default='',help='Folder containing the embedding file')
    parser.add_argument('--topk_topic_words',type=int,default=10,help='Top k number of topic words representing a topic')
    args = parser.parse_args()

    main()
