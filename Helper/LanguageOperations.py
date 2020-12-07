
from Helper.DataOperations import flatten_list
import en_core_web_sm
from nltk.stem import PorterStemmer 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas
import mlconjug
from subprocess import call
import re
from numpy.core.defchararray import strip
import inflect
import spacy
from itertools import groupby
nlp = spacy.load('en_core_web_sm')
import nltk 
from Helper.Subject_Verb_Object_Extract import _get_all_objs


def get_synonym_list(word, stem_id):
    synonyms = [] 
    if stem_id == 1:
        for syn in wordnet.synsets(word): 
            for l in syn.lemmas(): 
                ps = PorterStemmer() 
                stem_l = ps.stem(l.name())
                synonyms.append(stem_l)
    else:
        for syn in wordnet.synsets(word): 
            for l in syn.lemmas(): 
                synonyms.append(l.name()) 
    return synonyms


def get_synonyms_sequentiality():
   syn_list = ['after', 'subsequently', 'next', 'when']
   syn_list.append(get_synonym_list('after', 0))
   syn_list.append(get_synonym_list('subsequently', 0))
   syn_list.append(get_synonym_list('next', 0))
   syn_list.append(get_synonym_list('when', 0))
   syn_list = flatten_list(syn_list)
   syn_list = set(syn_list)
   #print(syn_list)
   return syn_list

def get_synonyms_backTracking():
    syn_list = ['again', 'once_more', 'go_back', 'another']
    syn_list.append(get_synonym_list('again', 0))
    syn_list.append(get_synonym_list('once_more', 0))
    syn_list.append(get_synonym_list('go_back', 0))
    syn_list.append(get_synonym_list('another', 0))
    syn_list = flatten_list(syn_list)
    syn_list = set(syn_list)
    #print(syn_list)
    return syn_list

def get_synonyms_exclusiveChoice():
    syn_list = ['if', 'else', 'otherwise']
    syn_list.append(get_synonym_list('if', 0))
    syn_list.append(get_synonym_list('else', 0))
    syn_list.append(get_synonym_list('otherwise', 0))
    syn_list = flatten_list(syn_list)
    syn_list = set(syn_list)
    #print(syn_list)
    return syn_list

def get_synonyms_skipping():
    syn_list = ['jump','skip','leave_out']
    syn_list.append(get_synonym_list('jump', 0))
    syn_list.append(get_synonym_list('skip', 0))
    syn_list.append(get_synonym_list('leave_out', 0))
    syn_list = flatten_list(syn_list)
    syn_list = set(syn_list)
    #print(syn_list)
    return syn_list

def get_synonyms_paralellism():
    syn_list = [('parallel','meantime', 'meanwhile', 'while')]
    #syn_list.append(get_synonym_list('parallel', 0))
    #syn_list.append(get_synonym_list('meantime', 0))
    #syn_list.append(get_synonym_list('meanwhile', 0))
    #syn_list.append(get_synonym_list('while', 0))
    syn_list = flatten_list(syn_list)
    syn_list = set(syn_list)
    #print(syn_list)
    return syn_list

def get_list_of_nouns(text):
    list_of_nouns = []
    nlp = en_core_web_sm.load()
    doc_temp = nlp(str(text))

    for chunk in doc_temp.noun_chunks:
        if chunk.root.pos_ == 'NOUN':
            list_of_nouns.append(chunk.root.pos_)

    #print(str(list_of_nouns))
    return list_of_nouns


def ident_entities(text):
    nlp = en_core_web_sm.load()
    temp = nlp(text)
    entities=[(i, i.label_, i.label) for i in temp.ents]
    return entities

def removeStopWords(text):
    stopwords = ('the', 'a', 'an')
    querywords = text.split()

    resultwords  = [word for word in querywords if word.lower() not in stopwords]
    result = ' '.join(resultwords)

    #print(result)
    return result

def plural_to_singular(word):
    p = inflect.engine()
    sing_word = p.singular_noun(word)
    if sing_word == False:
        return word
    else:
        return sing_word

def get_all_nouns(text):
    nouns = []
    doc = nlp(text)
    for chunk in doc.noun_chunks:
        noun = ' '.join([item.text for item in chunk if item.pos_ in {"NOUN","PROPN","ADJ"}])
    if len(noun) > 0:
        nouns.append(noun)
    #print(nouns)
    return nouns


def _get_lefts(token):
   lefts = []
   PHRASE_IND = {'amod', 'compound'}
   for tok in token.lefts:
      if tok.dep_ in PHRASE_IND or (tok.dep_ == 'pobj' and token.head.text != 'of'):
        lefts.append(tok)
        tmp = _get_lefts(tok)
        for x in tmp:
            lefts.append(x)
   lefts.reverse()
   return lefts


def _get_rights(token,doc):
   rights = []
   PHRASE_IND = {'amod', 'compound'}
   for tok in token.rights:
      if tok.dep_ in PHRASE_IND:
         rights.append(tok)
   if token.i + 1 < len(doc):
    if doc[token.i + 1].pos_ == 'ADP':
        rights.append(doc[token.i + 1])
        for right in doc[token.i + 1].rights:
            right_lefts = []
            for left in _get_lefts(right):
                right_lefts.append(left)
            right_lefts.append(right)
        for x in right_lefts:
            rights.append(x)
   return rights

def _get_phrase_of_token(token,doc):
   x = _get_lefts(token)
   y = _get_rights(token,doc)
   x.append(token)
   phrase = x + y
   phrase = ' '.join([item.text for item in phrase])
   return phrase


def _get_phrases_of_tokens(tokens,doc):
   phrases = []
   for token in tokens:
    phrase = _get_phrase_of_token(token,doc)
    phrases.append(phrase)
   return phrases

def _get_tokens_of_phrase_of_token(token,doc):
   x = _get_lefts(token)
   y = _get_rights(token,doc)
   x.append(token)
   phrase = x + y
   return phrase


def _get_activity(token):
    activity_list = []
    if token.pos_ == "VERB":
        base_token = token.lemma_
        activity_list.append(base_token)
    for tok in token.children:
        if tok.dep_ == "pobj" or tok.dep_ == "dobj" or tok.dep_ == "compound":
            activity_list.append(tok.text)
    if token.dep_ == "pobj":
        base_token = token.text
        activity_list.append(base_token)
        activity_list.reverse()
    activity = ' '.join([item for item in activity_list])
    return activity


def _combine_svos_over_same_verb(svos, tokens):
    svos_new = []
    for key, group in groupby(svos, lambda x: x[1]):
        groupsi = []
        for thing in group:
            groupsi.append(thing[2])
        #print(groupsi)
        combine = ""
        for x in groupsi:
            if groupsi.index(x) == 0:
                combine = x
            elif groupsi.index(x) == (len(groupsi) - 1):
                for chunk in tokens.noun_chunks:
                    #print(x,chunk,tokens[chunk[0].i - 1].text)
                    if (x == chunk[1:].text or x == chunk[0:].text) and tokens[chunk[0].i - 1].text == "and":
                        combine = combine + " and " + x
                    elif (x == chunk[1:].text or x == chunk[0:].text) and tokens[chunk[0].i - 1].text == "or":
                        combine = combine + " or " + x   
            else:
                combine = combine + ", " + x
        svo = (thing[0],thing[1],combine)
        svos_new.append(svo)
    return(svos_new)

def _get_antonym(token):
    antonyms = [] 
    is_neg = False
    for left in token.lefts:
        if left.dep_ == "neg":
            is_neg = True
    if is_neg == False:
        if token.pos_ == "ADJ" or token.pos_ == "NOUN":
            for syn in wordnet.synsets(token.text): 
                for l in syn.lemmas(): 
                    if l.antonyms(): 
                        antonyms.append(l.antonyms()[0].name()) 
            antonyms = list(set(antonyms))
            #antonyms_spacy_tok = []
            #for x in antonyms:
            #    antonyms.append(nlp(x)) 
        if token.pos_ == "VERB":
            to_append = "not " + token.text
            antonyms.append(to_append)
        if len(antonyms) == 0:
            to_append = "not " + token.text
            antonyms.append(to_append)
    else:
        antonyms.append(token.text)
    return antonyms[0]

def _verbify(noun):
    lem = wordnet.lemmas(noun)
    related_forms = [lem[i].derivationally_related_forms() for i in range(len(lem))]
    verbified = related_forms[0][0].name()
    return verbified

def _get_head_verb(token):
    if token.dep_ == "ROOT":
        return token.lemma_
    elif token.head.pos_ == "VERB" and not token.head.dep_ == "prep":
        return token.head.lemma_
    else:
        return _get_head_verb(token.head)

def _get_objects(sent):
    SUBJECTS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl","attr"}
    sent_tmp = nlp(sent)
    subject = ""
    adjectives = []
    for token in sent_tmp:
        if token.dep_ in SUBJECTS and token.pos_ == "NOUN":
            subject = token.text
        if token.dep_ in {"acomp","amod"} or token.text.lower() == "no":
            adjectives.append(token)
        if token.pos_ == "AUX" and token.dep_ == "auxpass" and token.head.pos_ == "VERB":
            adjectives.append(token.head)
    adjectives= ' '.join([item.text for item in adjectives])
    object = adjectives + " " + subject
    return object
        
def _get_prep_sent(sent):
    result_pp = []
    INPUT_IND = {'from', 'with', 'on'}
    OUTPUT_IND = {'as', 'into', 'for','towards','over'}
    ROLE_IND = {'by', 'through'}
    pp = ""
    conj = "AND"
    role = ""
    tmp = nlp(sent)
    tmp_pps = []
    for token in tmp:
        if token.head.dep_ == "prep" and token.head.text.lower() not in {"of","on"}:
            pp = ' '.join([item.text for item in token.head.subtree])
            for toki in nlp(pp):
                if toki.text.lower() == "or":
                    role = "XOR"
            reference = _get_head_verb(token.head)
            tmp_pps.append([pp,reference])
    objs = []
    for tmp_pp in tmp_pps:
        pp_nlp = nlp(tmp_pp[0])
        for token in pp_nlp:
            if token.text.lower() in INPUT_IND:
                role = "input"
            if token.text.lower() in OUTPUT_IND:
                role = "output"
            if token.text.lower() in ROLE_IND:
                role = "role"
            if role and token.pos_ == "NOUN" and token.dep_ in {"conj", "pobj", "dobj", "dative", "attr", "oprd"}:
                sentpart = ' '.join([item.text for item in token.head.subtree])
                sentpart = nlp(sentpart)
                chunkis = []
                for chunk in sentpart.noun_chunks:
                    chunkis.append(chunk)
                    if chunk.root.head.text.lower() == "of":
                        objs[-1][1] = objs[-1][1] + " of " + chunk.text
                    else:
                        if chunk[0].pos_ == "DET":
                            objs.append([tmp_pp[1],chunk[1:].text,role,conj])
                        else:
                            objs.append([tmp_pp[1],chunk.text,role,conj])
                break
    for obj in objs:
        result_pp.append(obj)
    rem_pps = []
    for rem_pp in tmp_pps:
        rem_pps.append(rem_pp[0])
    return  rem_pps, result_pp


def _get_adv_role(adv_phrase):
    role = ""
    PRECIND = {"once", "after"}
    SUCCIND = {"before"}
    advnlp = nlp(adv_phrase)
    for token in advnlp:
        if token.text.lower() in PRECIND:
            role = "precessor"
            break
        if token.text.lower() in SUCCIND:
            role = "successor"
            break
    return role

def _get_advmod_sent(sentence):
    mod_role = ""
    PRECIND = {"once","after"}
    SUCCIND = {"before"}
    nlp_sent = nlp(sentence)
    for token in nlp_sent:
        if token.dep_ == "advmod" and token.text.lower() in PRECIND:
            mod_role = "precessor"
            break
        if token.dep_ == "advmod" and token.text.lower() in SUCCIND:
            mod_role = "successor"
            break
        else: 
            mod_role = "successor"
    return mod_role
            
    
    