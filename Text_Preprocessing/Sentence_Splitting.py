import en_core_web_sm
from pycorenlp.corenlp import StanfordCoreNLP
from nltk.tree import Tree
from Helper.DataOperations import flatten_list
import subprocess
import stanza
import re
import spacy

def _get_sentences_spacy(text):
    sentences = []
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for sent in doc.sents:
        sentences.append(sent.text)
    return sentences


def _get_sentences_stanza(text):
    en_nlp = stanza.Pipeline('en')
    doc = en_nlp(text)
    sentences = []
    for sentence in doc.sentences:
        sentences.append(sentence.text)
    return sentences




