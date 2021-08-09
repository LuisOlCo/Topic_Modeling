import pandas as pd
import os
import re
import string

def preprocess_data(path_file):
    '''
    Preprocess column text containing the tweets
    '''

    data = pd.read_csv(path_file)
    data = data.drop_duplicates()
    data['text'] = data['text'].apply(lambda sentence: ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",sentence).split()))
    # Convert all text to lowercase
    data['text'] = data['text'].apply(lambda sentence: str(sentence).lower().strip())
    # remove symbols, exclamation marks... --> '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    data['text'] = data['text'].apply(lambda sentence: re.sub('[%s]' % re.escape(string.punctuation), '', sentence))
    # remove numbers
    data['text'] = data['text'].apply(lambda sentence: re.sub('[0-9]', '', sentence))
    # remove line breaks special characters
    data['text'] = data['text'].apply(lambda sentence: re.sub('[\t\n\r\f\v]' , '', sentence))
    # Substitute multiple white spaces characters for one
    data['text'] = data['text'].apply(lambda sentence: re.sub(' +' , ' ', sentence))
    # From analysing the data afterwards the symbol ° has shown up several time
    data['text'] = data['text'].apply(lambda sentence: sentence.replace('°',''))
    # delete retweet mark
    data['text'] = data['text'].apply(lambda sentence: sentence.replace('rt ',''))
    # delete retweet mark
    data['text'] = data['text'].apply(lambda sentence: sentence.replace('…',''))

    data['text'] = data['text'].apply(lambda sentence: sentence.replace('"',''))

    data['text'] = data['text'].apply(lambda sentence: sentence.replace(u'\xa0', u' '))
    # Remove URL from tweets
    data['text'] = data['text'].apply(lambda sentence: re.sub(r"http\S+", "", sentence))
    # Remove hashtags
    data['text'] = data['text'].apply(lambda sentence: re.sub(r"#\S+", "", sentence))
    # Remove pic links
    data['text'] = data['text'].apply(lambda sentence: re.sub(r'pic.twitter.com/[\w]*',"", sentence))
    # Remove emojis
    data['text'] = data['text'].apply(lambda sentence: deEmojify(sentence))
    # Substitute multiple white spaces characters for one
    data['text'] = data['text'].apply(lambda sentence: re.sub(' +' , ' ', sentence))

    return data

#remove the emoji
def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')


def remove_emoji(string):
    '''
    Method to remove all emojis in one tweet
    @Input: tweet in string format
    @Output: returns string without emojis
    '''
    emoji_pattern = re.compile("["
                              u"\U0001F600-\U0001F64F"  # emoticons
                              u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                              u"\U0001F680-\U0001F6FF"  # transport & map symbols
                              u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                              u"\U00002500-\U00002BEF"  # chinese char
                              u"\U00002702-\U000027B0"
                              u"\U00002702-\U000027B0"
                              u"\U000024C2-\U0001F251"
                              u"\U0001f926-\U0001f937"
                              u"\U00010000-\U0010ffff"
                              u"\u2640-\u2642"
                              u"\u2600-\u2B55"
                              u"\u200d"
                              u"\u23cf"
                              u"\u23e9"
                              u"\u231a"
                              u"\ufe0f"  # dingbats
                              u"\u3030"
                              "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)
