from transformers import AutoTokenizer, AutoModel

from dataset_v4 import *
from utils_v4 import *
from clusters_v4 import *
from embedding_storage_v4 import *

import argparse
from tqdm import tqdm


'''
Analyze depressed and control users topics separately

'''


def load_model(device,model_checkpoint = "sentence-transformers/bert-base-nli-mean-tokens"):
    '''
    Loads the pre-trained model from hugging face
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModel.from_pretrained(model_checkpoint)
    model.eval()
    model.to(device)
    return model, tokenizer

def mean_pooling(model_output, attention_mask):
    '''
    Transforms the BERT's output into a sentence embedding by doing an average of the token embeddings
    '''
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def compute_sentence_embeddings(model,tokenizer,tweets,device):
    '''
    Computes the sentence embeddings given a tuple of tweets, the model's input demands a list
    instread of a tuple, so it is needed to transform it
    '''
    if len(tweets) > 0:
        tweets = list(tweets)

    torch.cuda.empty_cache()
    encoded_input = tokenizer(list(tweets), padding=True, truncation=True, max_length=300, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input.to(device))

    return mean_pooling(model_output, encoded_input['attention_mask'])

def main(experiment_path, data_file_path, device, batch_size):

    print('Loading data...')
    total_dataset = TotalData(data_file_path)
    print('Loading models...')
    model,tokenizer = load_model(device)

    for period in total_dataset.periods:
        data_period_all = total_dataset.get_chunk_tweets(period)

        # Analyze Control and Depression users separately
        for label in data_period_all['label'].unique():
            data_period = data_period_all[data_period_all['label']==label]
            data_period = data_period.reset_index()
            data_period = data_period.drop(labels=['index'],axis=1)

            # Check if this period of time in this experiment has been already computed
            if period + '.csv' not in os.listdir(experiment_path):
                data_loader = torch.utils.data.DataLoader(DatasetPandas2Torch(data_period), batch_size = batch_size)
                # Load the data corresponding to the current period into the model
                print('----SENTENCE EMBEDDINGS----')
                print('Generating sentence embeddings for the period: ', period)
                embed_set = EmbeddingStorage()

                for sample in tqdm(data_loader):
                    #tweets, tweets_id = torch.Tensor(sample[0]).to(device), torch.Tensor(sample[1]).to(device)
                    tweets, tweets_id = sample[0], sample[1]
                    # Compute Sentence embeddings for the tweets in this batch
                    sentence_embeddings = compute_sentence_embeddings(model,tokenizer,tweets,device)
                    # Store the sentence embedding
                    embed_set.add_embeddings_period(sentence_embeddings.tolist(),tweets_id.tolist())

                print('----CLUSTERS----')
                embeddings = embed_set.embeddings['embeddings']
                indexes = embed_set.embeddings['indexes']
                if device == 'cpu':
                    cls = ComputeClusters(embeddings,indexes,device)
                else:
                    cls = ComputeClusters(embeddings,indexes)

                # Dataframe with the tweet_id and the id_cluster for the current period
                df_tweet_id_cluster = cls.compute()

                # set up index to join both dataframes
                df_tweet_id_cluster = df_tweet_id_cluster.set_index('tweet_id')
                data_period = data_period.set_index('tweet_id')
                # Create combinig dataframe
                df_combined = data_period.join(df_tweet_id_cluster, how="inner")
                # save the csv file
                if label == 1:
                    group = 'depressed'
                elif label == 0:
                    group = 'control'
                else:
                    group = None
                save_tweet_id_cluster_df(experiment_path,df_combined,period,group)

            else:
                print('This period has been analyzed already')



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Compute the sentence embeddings')
    parser.add_argument('--batch_size', '-b', default=16, type=int, help='The batch size.')
    parser.add_argument('--name', '-n', type=str, help='Name of the experiment')
    parser.add_argument('--cuda_unit', '-c',  default=0,type=str, help='Cuda unit to use')
    args = parser.parse_args()


    cuda_unit = 'cuda:' + args.cuda_unit
    device = torch.device(cuda_unit if torch.cuda.is_available() else 'cpu')

    print('Batch size--> ',args.batch_size)
    print('Device--> ',device)
    print('Experiment name--> ',args.name)


    main(experiment_path = create_experiment(args.name),
    data_file_path = '/home/luis/Data/depression_dataset/All_original_tweets.csv',
    device = device,
    batch_size = args.batch_size)
