#!/usr/bin/env python
"""
    Get vector representation for relation extraction
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import xrange, cPickle

import re
import os
import traceback
from pathlib import Path
import string
import pickle

from lxml import etree
from nltk import WordNetLemmatizer, word_tokenize
from gensim.models import Word2Vec
import numpy as np

from sentence import Sentence


WORD_EMBEDDING_PATHS = ("we_embedding_300", "we_vocab_300")


class OffsetTooComplicated(Exception):
    def __init__(self, *arg, **kwargs):
        super().__init__("OffsetTooComplicated", *arg, **kwargs)


class EntityTangled(Exception):
    def __init__(self, *arg, **kwargs):
        super().__init__("EntityTangled", *arg, **kwargs)


def get_xmls():
    """ Gather all .xml files in Train folder """
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


def analyze_xml(name):
    """
        Analyze a .xml and return a list of dictionary
    about tokens and entity relationships
    """
    with open(name, "rb") as fin:
        content = fin.read()

    root = etree.fromstring(content)
    sentences = []
    bad_sentences = []
    for child in root:
        if child.tag == "sentence":
            try:
                sent = parse_sentence(child)
                sent.source = name
                sentences.append(sent)
            except OffsetTooComplicated as ex:
                traceback.print_exc()
                bad_sentences.append(ex.args[1])
            except EntityTangled as ex:
                traceback.print_exc()
                bad_sentences.append(ex.args[1])
                
    return sentences, bad_sentences


def tokenize_and_lemmatize(sent):
    """ Tokenize and normailize the sentence """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in word_tokenize(sent.lower())
            if token not in string.punctuation]


def replace_text(cent, entities):
    """ To replace text for entities """
    sorted_entities = sorted(entities.items(), key=lambda x: x[1][0])
    if len(sorted_entities) == 0:
        return cent

    parts = []
    last_pos = 0
    for key, (pos, _) in sorted_entities:
        parts.append(cent[last_pos:pos[0]])
        parts.append(key)
        last_pos = pos[1]
    parts.append(cent[last_pos:])

    return "".join(parts)


def get_range(s):
    """ Get the offset for text """
    if ";" in s:
        raise OffsetTooComplicated(s)

    beg, end = s.split("-")
    beg, end = int(beg), int(end) + 1
    return (beg, end)


def process_entity_text(text):
    text = text.lower()
    lemmatizer = WordNetLemmatizer()

    pattern = re.compile(r"[\s]")
    tokens = pattern.split(text)
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]

    return "__".join(tokens)


def parse_sentence(tag):
    """ Parse an XML tag for a sentence """

    text = tag.get("text")
    entities = {}
    pairs = []

    # For entities
    for child in tag:
        if child.tag == "entity":
            entitiy_id = "_" + child.get("id").lower() + "_"
            pos = get_range(child.get("charOffset"))
            entity_text = process_entity_text(child.get("text"))
            entities[entitiy_id] = (pos, entity_text)

    # Tokenize
    new_cent = replace_text(text, entities)
    tokens = tokenize_and_lemmatize(new_cent)
    
    # For entities (replace back)
    entity_offsets = {}

    tokens_copy = tokens[:]
    for entity in entities:
        count = 0
        for i, token_copy in enumerate(tokens_copy):
            if entity in token_copy:
                if count > 1 or tokens[i] != token_copy:
                    raise EntityTangled(text)
                entity_offsets[entity] = i
                tokens[i] = entities[entity][1]
                count += 1

    # For pairs
    for child in tag:
        if child.tag == "pair":
            entity1 = entity_offsets["_" + child.get("e1").lower() + "_"]
            entity2 = entity_offsets["_" + child.get("e2").lower() + "_"]
            ddi = child.get("type") if (child.get("ddi") == "true") else "false"
            pairs.append((entity1, entity2, ddi))

    return Sentence(tokens, list(entity_offsets.values()), pairs, "")


def get_sentence_relations():
    """ Get all sentences and relations in .xml """
    xmls =get_xmls()

    sentences = []
    bad_sentences = []
    for xml in xmls:
        sents, bad_sents = analyze_xml(xml)
        sentences.extend(sents)
        bad_sentences.extend(bad_sents)

    return sentences, bad_sentences


def word_embedding(sentence_relations):
    """ Train a word embedding model """
    sentences = [sent.text for sent in sentence_relations]
    model = Word2Vec(sentences, size=50, workers=8, iter=100, min_count=1, sg=1)
    return model


def train_word_embedding(sentence_relations):
    """ Load the word embedding model, or train one """
    if not all(os.path.exists(p) for p in WORD_EMBEDDING_PATHS):
        model = word_embedding(sentence_relations)
        model.save_word2vec_format(*WORD_EMBEDDING_PATHS)
    else:
        print("load saved model.")
        model = Word2Vec.load_word2vec_format(*WORD_EMBEDDING_PATHS)

    return model


def vectorize(sentence_relations, model):
    """ Get vector representations for relation extraction """
    features = []
    for sent_relations in sentence_relations:
        for e1, e2, ddi in sent_relations.relation:
            vectors = []
            for i, tok in enumerate(sent_relations.text):
                try:
                    vectors.append(np.hstack([model[tok], [i - e1, i - e2]]))
                except Exception as ex:
                    print(sent_relations.text, e1, e2, sent_relations.source)
                    raise 
            features.append([sent_relations.source, sent_relations.text, (e1, e2, ddi), np.vstack(vectors)])

    return features


def output_features(features):
    """ Output the feature file to pickle file """
    with open("features_300.pickle", "wb") as fout:
        cPickle.dump(features, fout, protocol=2)


def main():
    """ main routine """
    sentence_relations, _ = get_sentence_relations()
    model = train_word_embedding(sentence_relations) 
    features = vectorize(sentence_relations, model)
    output_features(features)


if __name__ == "__main__":
    main()
