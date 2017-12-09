import pandas as pd
import nltk
import gzip
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import random
from matplotlib import pyplot
from pylab import rcParams
from IPython.display import HTML, display
from multiprocessing import Pool
from timeit import default_timer as timer

rcParams['figure.figsize'] = 15, 10
np.random.seed(42)



def filterTags(w1, w2):
    """
    w1: First tagged word
    w2: Second tagged word

    Returns true if the bigram collocation is of type adjective-noun, adverb-past participle or verb-adverb
    """
    _, tag1 = nltk.pos_tag(nltk.word_tokenize(w1))[0]
    _, tag2 = nltk.pos_tag(nltk.word_tokenize(w2))[0]

    return (tag1.startswith('JJ') and tag2.startswith('NN')) or \
           (tag1.startswith('RB') and tag2.startswith('VBN')) or \
           (tag1.startswith('VB') and tag2 == "JJ")


def getCollocations(text):
    """
    text: The concatenation of all the review for a given product

    Returns the collocations of words larger 
    """
    ignored_words = nltk.corpus.stopwords.words('english')

    tokens = nltk.word_tokenize(text)
    bigram_measures = nltk.collocations.BigramAssocMeasures()

    finder = nltk.BigramCollocationFinder.from_words(tokens)

    # Ignore bigram that are infrequent
    finder.apply_freq_filter(3)

    # Retrieves the 10 most common bigrams
    res = finder.nbest(bigram_measures.pmi, 10)

    res = [(x, round(finder.score_ngram(bigram_measures.pmi, x[0], x[1]), 2)) for x in res if filterTags(x[0], x[1])]

    if (len(res) > 0):
        return res
    else:
        return np.nan


if __name__ == "__main__":

    df_elec = pd.read_pickle('data/electronics_serialized.pickle')
    df_product = df_elec.groupby(["asin"])['reviewText'].agg(lambda x:''.join(set(x.str.lower()))).reset_index()

    N_THREADS = 8

    def parallelized_collocations(grouped_reviews: pd.Series):
        print("Test")
        responses = [None] * len(grouped_reviews)
        print(len(grouped_reviews))
        processes = []

        with Pool(N_THREADS) as thread_pool:
            responses = thread_pool.map(getCollocations, grouped_reviews.tolist())

        return pd.Series(responses)



    start = timer()
    df_product["reviewText"] = parallelized_collocations(df_product["reviewText"])
    end = timer()

    df_product.to_pickle("elec_collocations.pickle")

    print(end - start)