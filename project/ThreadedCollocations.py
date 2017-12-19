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
import datetime
from collections import OrderedDict
import itertools
import re, string

rcParams['figure.figsize'] = 15, 10
np.random.seed(42)

black_list_t1 = ['few', 'am', 'is', 'was', "'m", 'different', 'being', 'other', 'same', 'last', 'upper']
black_list_t2 = ['reviews', 'engineer', 'days', 'months', 'years']

def filterBigram(w1, w2):
    return  filterTags(w1, w2) and notInBlackList(w1, w2) and not is_compound_word(w1+"_"+w2)


def notInBlackList(w1, w2):
    return (w1 not in black_list_t1) and (w2 not in black_list_t2)

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
    print(text[:3])

    tokens = nltk.word_tokenize(text)
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = nltk.BigramCollocationFinder.from_words(tokens)

    # Ignore bigram that are infrequent
    finder.apply_freq_filter(3)

    bigram_res_likelihood = finder.nbest(bigram_measures.likelihood_ratio, None)
    bigram_res_pmi = finder.nbest(bigram_measures.pmi, None)

    res_likelihood = [(x, round(finder.score_ngram(bigram_measures.raw_freq, x[0], x[1]), 5) * 200)
                      for x in bigram_res_likelihood if filterBigram(x[0], x[1])]

    res_pmi = [(x, round(finder.score_ngram(bigram_measures.raw_freq, x[0], x[1]), 5) * 200)
                      for x in bigram_res_pmi if filterBigram(x[0], x[1])]

    # Convert back to list
    res = list(
        # Eliminate duplicates
        OrderedDict.fromkeys(
            # Flatten list of tuples
            list(
                itertools.chain.from_iterable(
                    # Assemble pmi and likelihood scores
                    zip(res_pmi, res_likelihood)
                )
            )
        )
    )


    if (len(res) > 0):
        return res[:10]
    else:
        return np.nan


compound_dictionnary = pd.read_pickle("data/filtered_compound_words.pickle.gz")

def is_compound_word(word):
    """
    
    :param word: The word to be searched in the dictionnary
    :return: True if the word was found in the dictionnary, false otherwise
    """
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

def getAdjectives(text):
    """
    text: The concatenation of all the reviews for a given product

    Returns a list of adjectives and their frequency
    """

    print("AdjExt "+text[:5])

    #text = re.sub(pattern, " ", text)

    tokens = nltk.word_tokenize(text)
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

    return Counter(adjectives).most_common(15)

if __name__ == "__main__":
    df_elec = pd.read_pickle('data/electronics_serialized.pickle')
    df_product = df_elec.groupby(["asin"])['reviewText'].agg(lambda x: ''.join(set(x.str.lower()))).reset_index()

    print(len(df_product))

    N_THREADS = 8

    def parallelized_apply(func, grouped_reviews: pd.Series):

        with Pool(N_THREADS) as thread_pool:
            responses = thread_pool.map(func, grouped_reviews.tolist())

        return pd.Series(responses)


    print(datetime.datetime.now())
    adjs = parallelized_apply(getAdjectives, df_product["reviewText"])
    colls = parallelized_apply(getCollocations, df_product["reviewText"])

    df_product["rawAdjectives"] = adjs
    df_product["reviewText"] = colls

    df_product.to_pickle("elec_collocations_adjectives_hybrid.pickle")

    print(df_product)
