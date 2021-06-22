from re import sub

from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.models.fasttext import load_facebook_model
from core.IndoStopword import STOP_WORDS

nltk_stop_words = STOP_WORDS

class NotReadyError(Exception):
    pass

class SemanticMeasure:
    default_model = "../cc.id.300.bin"
    model_ready = False
    
    def __init__(self, stopwords=None, verbose=False):
        self.verbose = verbose
        
        if self.verbose: 
            print('Init')

        if stopwords is None:
            self.stopwords = nltk_stop_words
        else:
            self.stopwords = stopwords

        self._load_model()

    def _load_model(self):
        self._setup_model()

    def _setup_model(self):
        if self.verbose: 
            print('Loading model')

        loaded_model = load_facebook_model(self.default_model)
        self.model = loaded_model.wv

        if self.verbose: 
            print('Model loaded')

        self.similarity_index = WordEmbeddingSimilarityIndex(self.model)
        self.model_ready = True

    def preprocess(self, doc: str):
        doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
        doc = sub(r'<[^<>]+(>|$)', " ", doc)
        doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
        doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
        
        return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in self.stopwords]

    def similarity_query(self, answer: str, key: str):
        if len(answer) == 0 or len(key) == 0:
            return False

        if self.model_ready:
            documents = [answer, key]
            
            if self.verbose:
                print(f'{len(documents)} documents loaded and ready to preprocess')

            corpus = [self.preprocess(document) for document in documents]
            
            if self.verbose:
                print(f'{len(corpus)} documents loaded into corpus')
            
            dictionary = Dictionary(corpus)
            tfidf = TfidfModel(dictionary=dictionary)
            similarity_matrix = SparseTermSimilarityMatrix(self.similarity_index, dictionary, tfidf)

            answer_bow = dictionary.doc2bow(self.preprocess(answer))
            key_bow = dictionary.doc2bow(self.preprocess(key))
            
            # Measure soft cosine similarity
            scores = similarity_matrix.inner_product(answer_bow, key_bow, normalized=True)

            return scores

        else:
            raise NotReadyError('Word embedding model is not ready.')