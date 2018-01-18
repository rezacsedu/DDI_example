#!/usr/bin/env python3
"""
    File for Sentence class
"""

class Sentence:
    """
        A wrapper class for sentence, its entities and the relationship
        between these entities
    """
    def __init__(self, tokens, entities, relations, source):
        """ """
        self.text = tokens
        self.entities = entities
        self.relation = relations
        self.source = source
