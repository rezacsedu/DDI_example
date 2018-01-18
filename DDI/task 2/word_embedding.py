#!/usr/bin/env python3
"""0
    To preprocessing sentences and train a word embedding model
"""
from string import punctuation
from pathlib import Path

from lxml import etree
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec


def train_word2vec(sentences):
    """ Train a Word2Vec model """
    model = Word2Vec(sentences, size=50, workers=8, iter=100, min_count=3)
    return model


def tokenize_and_lemmatize(sents):
    """ Do tokenization and lematizatino on sentences """
    new_sents = []
    lemmatizer = WordNetLemmatizer()
    # stops = set(stopwords.words("english"))
    tokens = set()
    for sent in sents:
        # Tokenize (with lowrecased)
        new_sent = word_tokenize(sent.lower())
        # Remove punctuations
        new_sent = [tok for tok in new_sent if tok not in punctuation
                    # and tok not in stops
                   ] 
        new_sent = [lemmatizer.lemmatize(tok) for tok in new_sent]
        tokens.update(new_sent)
        new_sents.append(new_sent)

    return new_sents


def analyze_xml(name):
    """
        Analyze a .xml and return a list of sentences
    """
    with open(name, "rb") as fin:
        content = fin.read()

    root = etree.fromstring(content)
    sentences = []
    for child in root:
        if child.tag == "sentence":
            sentence = child.get("text")
            for subchild in child:
                if subchild.tag == "entity" and " " in subchild.get("text"):
                    old = subchild.get("text")
                    new = old.replace(" ", "__")
                    sentence = sentence.replace(old, new)
            sentences.append(sentence)

    return sentences


def get_xmls():
    """ To read all xml files under a specific directory hierarchy tree """
    def _get_xmls(path):
        """ Recursive search all .xml files """
        ret = []

        for child in path.iterdir():
            if child.is_dir():
                ret.extend(_get_xmls(child))
            elif child.name.endswith('.xml'):
                ret.append(str(child))

        return ret

    return _get_xmls(Path("Train/"))


def get_sentences():
    """ Parsing XML files and return a list of sentences """
    xmls = get_xmls()

    sentences = []
    for xml in xmls:
        sentences.extend(analyze_xml(xml))

    return sentences


def main():
    """ main routine """
    sentences = get_sentences()
    sentences = tokenize_and_lemmatize(sentences)
    model = train_word2vec(sentences)
    model.save_word2vec_format("we_embedding", "we_vocab")


if __name__ == "__main__":
    main()
