import os
import dill
import re
import string
import nltk
from nltk.corpus import wordnet
import spacy
from collections import defaultdict


def get_stopwords(from_file=False, filename='stopwords.pickle'):
    """
    Returns set of stopwords, strings to be removed from list of tokens
    """
    if from_file:
        with open(filename, 'rb') as handle:
            stopwords = dill.load(handle)
    else:
        spacy.load("es_core_news_sm", disable=["tagger", "parser", "ner"])

        stopwords = set.union(set(nltk.corpus.stopwords.words('spanish')),
                              spacy.lang.es.stop_words.STOP_WORDS)

    return stopwords


def store_stopwords(stopwords={}, filename='stopwords.pickle'):
    assert stopwords, 'Empty set, no stopwords to store'
    with open(filename, 'wb') as handle:
        dill.dump(stopwords, handle)


STOP_WORDS = get_stopwords()


class TextPreprocessor(object):
    """
    Class to define a text preprocessing pipeline.
    """
    def __init__(self, rm_linebreaks=True, keep_quotes=False, rm_HTML=True,
                 to_lower=True, to_tokens=True, nlp_lib='nltk',
                 spacy_lm="en_core_web_sm", to_lemmas=False,
                 rm_punctuation=True, rm_stopwords=False, join_text=False):

        # List of functions to apply
        self.pipeline = []

        if rm_linebreaks:
            self.pipeline.append(self.rm_linebreaks)

        if keep_quotes:
            self.pipeline.append(self.keep_quotes)

        if rm_HTML:
            self.pipeline.append(self.rm_HTML)

        if to_lower:
            self.pipeline.append(self.to_lower)

        if to_tokens:
            assert nlp_lib in {'nltk', 'spacy'}, 'Invalid NLP library'

            if nlp_lib == 'spacy':
                self.nlp = spacy.load(spacy_lm, disable=["parser", "ner"])

            if not (nlp_lib == 'spacy' and to_lemmas):
                self.tokenize = self.get_tokenizer(nlp_lib)
                self.pipeline.append(self.tokenize)

            # Default lemmatizer: nltk.stem.WordNetLemmatizer
            if to_lemmas:
                self.lemmatize = self.get_lemmatizer(nlp_lib)
                self.pipeline.append(self.lemmatize)

            if rm_punctuation:
                self.rm_punctuation = self.get_punct_remover(nlp_lib)
                self.pipeline.append(self.rm_punctuation)

            if rm_stopwords:
                self.stopwords = STOP_WORDS
                self.pipeline.append(self.rm_stopwords)

            # Join list of tokens into single string
            if join_text:
                self.pipeline.append(self.join_text)

    def preprocess(self, text: str):
        # Consecutively apply functions in preprocessing pipeline
        pp_text = text

        for func in self.pipeline:
            pp_text = func(pp_text)

        return pp_text

    def rm_linebreaks(self, text: str):
        # Remove linebreaks \n
        return re.sub("[\n]+", " ", text)

    def keep_quotes(self, text: str):
        # Escape quotation marks
        return re.sub(r"\\*'", r"\'", re.sub(r'\\*"', r'\"', text))

    def rm_HTML(self, text: str):
        # Remove HTML tags
        return re.sub("<.*?>", " ", text)

    def to_lower(self, text: str):
        return text.lower()

    def get_tokenizer(self, nlp_lib='nltk'):

        if nlp_lib == 'nltk':
            def tokenize(text: str):
                return [token for token in nltk.word_tokenize(text)]

        elif nlp_lib == 'spacy':
            def tokenize(text: str):
                # with self.nlp.disable_pipes("tagger"):
                return [token.text for token in self.nlp(text)]

        return tokenize

    def get_lemmatizer(self, nlp_lib='nltk'):
        # import nltk
        # from collections import defaultdict

        if nlp_lib == 'nltk':
            lemma = nltk.stem.WordNetLemmatizer()

            # WordNetLemmatizer needs POS tags to get the part of speech.
            tag_map = defaultdict(lambda: wordnet.NOUN)
            tag_map['J'] = wordnet.ADJ
            tag_map['V'] = wordnet.VERB
            tag_map['R'] = wordnet.ADV

            def lemmatize(token_list: list):
                # Takes list of tokens and converts them into lemmas
                lemma_list = [lemma.lemmatize(word, tag_map[tag[0]])
                              for word, tag in nltk.pos_tag(token_list)]
                return lemma_list

        elif nlp_lib == 'spacy':

            def lemmatize(text: str):
                # Takes a text and splits it into lemmas
                return [token.lemma_ for token in self.nlp(text)]

        return lemmatize

    def get_punct_remover(self, nlp_lib='nltk'):

        if nlp_lib in {'nltk', 'spacy'}:
            def rm_punctuation(token_list: list):
                new_list = []
                chars_to_strip = string.punctuation + ' '
                for token in token_list:
                    new = token.strip(chars_to_strip)
                    if not ''.__eq__(new):
                        new_list.append(new)
                return new_list

        return rm_punctuation

    def rm_stopwords(self, token_list: list):
        tr = str.maketrans('', '', string.punctuation.replace("'", ""))
        new_list = [token for token in token_list
                    if token.translate(tr) not in self.stopwords]
        return new_list

    def join_text(self, x: list):
        assert isinstance(x, list), "Argument must be list of tokens"
        return ' '.join(x)

    def dump_pickle(self, file_path: str):
        if not file_path.endswith('.pickle'):
            file_path += '.pickle'

        if os.path.isfile(file_path):
            os.remove(file_path)

        with open(file_path, 'wb') as handle:
            dill.dump(self, handle, protocol=dill.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, file_path: str):
        """
        Example usage:
        pp = TextPreprocessor.from_pickle('folder/TextPreprocessor.pickle')
        """
        if not file_path.endswith('.pickle'):
            file_path += '.pickle'
        assert os.path.isfile(file_path), 'Invalid file path.'

        with open(file_path, 'rb') as handle:
            new_obj = dill.load(handle)

        return new_obj


