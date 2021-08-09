import os
import pandas as pd
from preprocess_data import *
from utils_v2 import *
from find_topic import *
import pickle
import argparse


def main(FILE_PATH):
    # create the object that finds the topics
    topics = Topics()

    for file in os.listdir(FILE_PATH):
        if '.csv' in file:
            period = file.replace('_.csv','')
            print('Computing topics for period: ',period)
            path_file = os.path.join(FILE_PATH,file)
            data = preprocess_data(path_file)
            data = data[['text','cluster_id']]
            # compute the topics for the given file, every file contains all the tweets that
            # were post in a given period, generally one natural month
            topics.compute_topics_tfidf(data,period)

    topics.save_topics(FILE_PATH)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Find topics based in most relevant words')
    parser.add_argument('--experiment_path', '-p', type=str, help='Path to the experiment folder')

    args = parser.parse_args()

    main(args.experiment_path)
