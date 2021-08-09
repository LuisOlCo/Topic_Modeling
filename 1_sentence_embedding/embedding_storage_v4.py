

class EmbeddingStorage():
    '''
    Data Strucutre to store all the sentence embeddings arranged by thier period
    '''
    def __init__(self):
        self.embeddings = {}

    def add_embeddings_period(self,embeddings,ids):
        '''
        Save the embeddings on the period that we are studying
        '''
        if 'embeddings' in self.embeddings:
            self.embeddings['embeddings'] += list(embeddings)
            self.embeddings['indexes'] += list(ids)
        else:
            self.embeddings = {'embeddings':list(embeddings),'indexes':list(ids)}

    def _embeddings_ids_dictionary(self,embeddings,ids):
        '''
        Converts all the embedding and their indexes into one dictionary
        '''
        dicto = {}
        for embedd,index in zip (embeddings,ids):
            dicto[embedd] = index
        return dicto
