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
from collections import Counter
import re, string

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


def getCollocations(tokens):
    """
    text: The concatenation of all the review for a given product
    Returns the collocations of words larger
    """

    bigram_measures = nltk.collocations.BigramAssocMeasures()

    finder = nltk.BigramCollocationFinder.from_words(tokens)

    # Ignore bigram that are infrequent
    finder.apply_freq_filter(3)
    bigram_res = finder.nbest(bigram_measures.pmi, None)
    res = [(x,round(finder.score_ngram(bigram_measures.raw_freq, x[0], x[1]),5)*200)
                  for x in bigram_res if filterTags(x[0],x[1]) and not is_compound_word(x[0]+"_"+x[1])]

    if (len(res) > 0):
        return res
    else:
        return np.nan


compound_dictionnary = pd.read_pickle("data/filtered_compound_words.pickle.gz")

def is_compound_word(word):
    i = compound_dictionnary.searchsorted(word)

    if i == len(compound_dictionnary):
        return False

    x = compound_dictionnary.iloc[i].values
    return x[0] == word


def isAdjective(token):
    """
    w: Tagged word

    Returns true if the part-of-speech is an adjective
    """
    _, tag = nltk.pos_tag([token])[0]

    return tag == 'JJ' or tag == 'JJS'


pattern = re.compile('[\\\*\.]+')

def getAdjectives(tokens):
    """
    text: The concatenation of all the reviews for a given product

    Returns a list of adjectives and their frequency
    """

    print("AdjExt "+tokens[0])
    num_tokens = len(tokens)

    adjectives = []

    for i, token in enumerate(tokens):

        # Extract adjectives
        if isAdjective(token):

            one_after = min(i + 1, num_tokens-1)

            if not is_compound_word(tokens[i] + "_" + tokens[one_after]):
                one_before = max(i - 1, 0)
                two_before = max(i - 2, 0)

                # Identify negations
                if tokens[one_before] == 'not':
                    adjectives.append(tokens[one_before] + " " + tokens[i])
                elif tokens[two_before] == 'not':
                    adjectives.append(tokens[two_before] + " " + tokens[i])
                else:
                    adjectives.append(token)

    return Counter(adjectives).most_common(20)

if __name__ == "__main__":
    df_elec = pd.read_pickle('data/electronics_serialized.pickle')
    df_product = df_elec.groupby(["asin"])['reviewText'].agg(lambda x: ''.join(set(x.str.lower()))).reset_index()

    print(len(df_product))

    N_THREADS = 8

    def parallelized_apply(thread_pool, func, grouped_reviews: pd.Series):
        return pd.Series(thread_pool.map(func, grouped_reviews.tolist()))


    with Pool(N_THREADS) as thread_pool:

        tokenized_text = df_product["reviewText"].apply(nltk.word_tokenize)

        adjs = parallelized_apply(thread_pool, getAdjectives, tokenized_text)
        colls = parallelized_apply(thread_pool, getCollocations, tokenized_text)

        df_product["rawAdjectives"] = adjs
        df_product["reviewText"] = colls

        df_product.to_pickle("elec_collocations_adjectives.pickle")

        print(df_product)
