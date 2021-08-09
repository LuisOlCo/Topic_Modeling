# Topic words
# docs --> /home/luis/Topic_Modeling/ETM_Tweets_version/data/depression_dataset_2020/PERIOD
# dictionary --> /home/luis/Topic_Modeling/ETM_Tweets_version/data/depression_dataset_2020/PERIOD



def calc_topic_diversity(topic_words):
    '''
    Topic diversity: percentage of unique words in the top 25 words of all topics.
    Score close to 0 indicates redundant topics;
    Score close to 1 indicates more varied topics.
    topic_words is in the form of [[w11,w12,...],[w21,w22,...]]
    '''
    # Vocabulary with all the topic words
    vocab = set(sum(topic_words,[]))
    # Total number of words summing all the topics words of each topic
    n_total = len(topic_words) * len(topic_words[0])
    topic_div = len(vocab) / n_total
    return topic_div

def calc_topic_coherence(topic_words,docs,dictionary,emb_path=None,taskname=None,sents4emb=None,calc4each=False):
    '''
    Compute all the Topic Coherence scores:
    - C_V
    - C_W2V
    - C_UCI
    - C_NPMI
    '''
    # emb_path: path of the pretrained word2vec weights, in text format.
    # sents4emb: list/generator of tokenized sentences.

    # Computing the C_V score
    cv_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_v')
    cv_per_topic = cv_coherence_model.get_coherence_per_topic() if calc4each else None
    cv_score = cv_coherence_model.get_coherence()

    # Computing the C_W2V score
    try:
        emb_path = os.path.join(os.getcwd(),'GoogleNews-vectors-negative300.bin.gz')

        if emb_path!=None and os.path.exists(emb_path):
            keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(emb_path,binary=True)

        else:
            raise Exception("C_w2v score isn't available for the missing of training corpus (sents4emb=None).")

        print('Computing C_W2V score...')
        w2v_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_w2v',keyed_vectors=keyed_vectors)
        w2v_per_topic = w2v_coherence_model.get_coherence_per_topic() if calc4each else None
        w2v_score = w2v_coherence_model.get_coherence()

    except Exception as e:
        print('It did not work out')
        print(e)
        #In case of OOV Error
        w2v_per_topic = [None for _ in range(len(topic_words))]
        w2v_score = None

    # Computing the C_UCI score
    c_uci_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_uci')
    c_uci_per_topic = c_uci_coherence_model.get_coherence_per_topic() if calc4each else None
    c_uci_score = c_uci_coherence_model.get_coherence()


    # Computing the C_NPMI score
    c_npmi_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_npmi')
    c_npmi_per_topic = c_npmi_coherence_model.get_coherence_per_topic() if calc4each else None
    c_npmi_score = c_npmi_coherence_model.get_coherence()

    return (cv_score,w2v_score,c_uci_score, c_npmi_score),(cv_per_topic,w2v_per_topic,c_uci_per_topic,c_npmi_per_topic)

def co_occur(word2docs,w1,w2):
    return len(word2docs[w1].intersection(word2docs[w2]))

def mimno_topic_coherence(topic_words,docs):
    '''
    Mimno Topic Coherence by:
    Optimizing Semantic Coherence in Topic Models - Mimno, David et al. 2011 ACL
    '''
    tword_set = set([w for wlst in topic_words for w in wlst])
    word2docs = {w:set([]) for w in tword_set}
    scores = []

    # Create a dictionary with the top words for each topic and the indices of the documents where this word is present
    for docid,doc in enumerate(docs):
        doc = set(doc)
        for word in tword_set:
            if word in doc:
                word2docs[word].add(docid)

    for wlst in topic_words:
        s = 0
        for i in range(1,len(wlst)):
            for j in range(0,i):
                s += np.log((co_occur(word2docs,wlst[i],wlst[j])+1.0)/len(word2docs[wlst[j]]))
        scores.append(s)
    return np.mean(s)

def evaluate_topic_quality(topic_words, test_data, taskname=None, calc4each=False):
    '''
    Computes different topic scores to evaluate the performance of the process
    - Topic Diversity
    - Topic Coherence
    - Mimno Topic Coherence
    '''
    # TOPIC DIVERSITY
    td_score = calc_topic_diversity(topic_words)
    print(f'topic diversity:{td_score}')

    # TOPIC COHERENCE
    (c_v, c_w2v, c_uci, c_npmi),\
        (cv_per_topic, c_w2v_per_topic, c_uci_per_topic, c_npmi_per_topic) = \
        calc_topic_coherence(topic_words=topic_words, docs=test_data.docs, dictionary=test_data.dictionary,
                             emb_path=None, taskname=taskname, sents4emb=test_data, calc4each=calc4each)

    print('c_v:{}, c_w2v:{}, c_uci:{}, c_npmi:{}'.format(c_v, c_w2v, c_uci, c_npmi))
    scrs = {'c_v':cv_per_topic,'c_w2v':c_w2v_per_topic,'c_uci':c_uci_per_topic,'c_npmi':c_npmi_per_topic}

    if calc4each:
        for scr_name,scr_per_topic in scrs.items():
            print(f'{scr_name}:')
            for t_idx, (score, twords) in enumerate(zip(scr_per_topic, topic_words)):
                print(f'topic.{t_idx+1:>03d}: {score} {twords}')

    # MIMNO TOPIC COHERENCE
    mimno_tc = mimno_topic_coherence(topic_words, test_data.docs)
    print('mimno topic coherence:{}'.format(mimno_tc))
    if calc4each:
        return (c_v, c_w2v, c_uci, c_npmi, mimno_tc, td_score), (cv_per_topic, c_w2v_per_topic, c_uci_per_topic, c_npmi_per_topic)
    else:
        return c_v, c_w2v, c_uci, c_npmi, mimno_tc, td_score
