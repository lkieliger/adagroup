import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk import word_tokenize

class SentimentAnalyser:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
    def _penn_to_wn(self, tag):
        """
        Convert between the PennTreebank tags to simple Wordnet tags
        """
        if tag.startswith('J'):
            return wn.ADJ
        elif tag.startswith('N'):
            return wn.NOUN
        elif tag.startswith('R'):
            return wn.ADV
        elif tag.startswith('V'):
            return wn.VERB
        return None

    def sentiment_for_tagged_word(self, tagged_word):
        """
        Compute the score for a given tagged word.
        The word is assumed to be tagged using the Penn Treebank Project's tags
        Return None for irrelevant words, a tuple (positive score, negative score) otherwise
        """
        word, tag = tagged_word
        
        wn_tag = self._penn_to_wn(tag)
        
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            return None
        
        lemma = self.lemmatizer.lemmatize(word, pos=wn_tag)
        
        if not lemma:
            return None
        
        synsets = wn.synsets(lemma, pos=wn_tag)
        
        if not synsets:
            return None
        
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        
        return swn_synset.pos_score(), swn_synset.neg_score()
    
    def sentiment_score_for_raw_sentence(self, raw_sentence):
        """
        Compute the sum of the differences in sentiment score for each word in the sentence
        """
        tagged_sentence = nltk.pos_tag(word_tokenize(raw_sentence))
        sum_deltas = 0

        for tagged_word in tagged_sentence:
            scores = self.sentiment_for_tagged_word(tagged_word)

            if scores is None:
                continue
                
            pos_score, neg_score = scores
            sum_deltas += (pos_score - neg_score)
        
        return sum_deltas