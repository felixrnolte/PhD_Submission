import en_core_web_sm
import spacy
from Helper.DataOperations import flatten_list
import inflect
from itertools import groupby
#from Helper.LanguageOperations import _combine_svos_over_same_verb

# use spacy small model
nlp = en_core_web_sm.load()

# dependency markers for subjects
SUBJECTS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"}
# dependency markers for objects
OBJECTS = {"dobj", "dative", "attr", "oprd"}
# POS tags that will break adjoining items
BREAKER_POS = {"CCONJ", "VERB", "PUNCT"}
# words that are negations
NEGATIONS = {"no", "not", "n't", "never", "none"}

# does dependency set contain any coordinating conjunctions?
def contains_conj(depSet):
    return "and" in depSet or "or" in depSet or "nor" in depSet or \
           "but" in depSet or "yet" in depSet or "so" in depSet 


# get subs joined by conjunctions
def _get_subs_from_conjunctions(subs):
    more_subs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if contains_conj(rightDeps):
            more_subs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(more_subs) > 0:
                more_subs.extend(_get_subs_from_conjunctions(more_subs))
    return more_subs


# get objects joined by conjunctions
def _get_objs_from_conjunctions(objs):
    more_objs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)

        rightDeps = {tok.lower_ for tok in rights}
        if contains_conj(rightDeps):
            more_objs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if len(more_objs) > 0:
                more_objs.extend(_get_objs_from_conjunctions(more_objs))
    return more_objs


# find sub dependencies
def _find_subs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB" or head.pos_ == "AUX":
        subs = []
        for tok in head.lefts:
            subs = [tok for tok in head.lefts if tok.dep_ in SUBJECTS]
        if len(subs) > 0:
            verb_negated = _is_negated(tok)
            subs.extend(_get_subs_from_conjunctions(subs))
            return subs, verb_negated
        elif head.head != head:
            return _find_subs(head)
    elif head.pos_ == "NOUN":
        return [head], _is_negated(tok)
    return [], False


# is the tok set's left or right negated?
def _is_negated(tok):
    parts = list(tok.lefts) + list(tok.rights)
    for dep in parts:
        if dep.dep_ in BREAKER_POS:
            break
        if dep.lower_ in NEGATIONS:
            return True
    return False


# get all the verbs on tokens with negation marker
def _find_svs(tokens):
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB"]
    for v in verbs:
        subs, verbNegated = _get_all_subs(v)
        if len(subs) > 0:
            for sub in subs:
                svs.append((sub.orth_, "!" + v.orth_ if verbNegated else v.orth_))
    return svs


# get grammatical objects for a given set of dependencies (including passive sentences)
def _get_objs_from_prepositions(deps, is_pas):
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" or dep.pos_ == "VERB" and (dep.dep_ == "prep" or (is_pas and dep.dep_ == "agent")) and dep.lower_ not in ["to", "with", "out", "from", "in", "for"]:
            objs.extend([tok for tok in dep.rights if tok.dep_  in OBJECTS or
                         (tok.pos_ == "PRON" and tok.lower_ == "me") or
                         (is_pas and tok.dep_ == 'pobj')])
    return objs


# get objects from the dependencies using the attribute dependency
def _get_objs_from_attrs(deps, is_pas):
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(_get_objs_from_prepositions(rights, is_pas))
                    if len(objs) > 0:
                        return v, objs
    return None, None


# xcomp; open complement - verb has no suject
def _get_obj_from_xcomp(deps, is_pas):
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(_get_objs_from_prepositions(rights, is_pas))
            if len(objs) > 0:
                return v, objs
    return None, None


# get all functional subjects adjacent to the verb passed in
def _get_all_subs(v):
    verb_negated = _is_negated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    if len(subs) > 0:
        subs.extend(_get_subs_from_conjunctions(subs))
    else:
        foundSubs, verb_negated = _find_subs(v)
        subs.extend(foundSubs)
    return subs, verb_negated


# is the token a verb?  (excluding auxiliary verbs)
def _is_non_aux_verb(tok):
    return tok.pos_ == "VERB" and (tok.dep_ != "aux" and tok.dep_ != "auxpass" and tok.dep_ != "amod" and tok.dep_ != "prep" and tok.dep_ != "acl") #and tok.dep_ != "advcl"


# return the verb to the right of this verb in a CCONJ relationship if applicable
# returns a tuple, first part True|False and second part the modified verb if True
def _right_of_verb_is_conj_verb(v):
    # rights is a generator
    rights = list(v.rights)

    # VERB CCONJ VERB (e.g. he beat and hurt me)
    if len(rights) > 1 and rights[0].pos_ == 'CCONJ':
        for tok in rights[1:]:
            if _is_non_aux_verb(tok):
                return True, tok

    return False, v


# get all objects for an active/passive sentence
def _get_all_objs(v, is_pas):
    # rights is a generator
    rights = list(v.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS or (is_pas and tok.dep_ == 'pobj')]
    objs.extend(_get_objs_from_prepositions(rights, is_pas))
    #potentialNewVerb, potentialNewObjs = _get_objs_from_attrs(rights)
    #if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
    #    objs.extend(potentialNewObjs)
    #    v = potentialNewVerb

    potential_new_verb, potential_new_objs = _get_obj_from_xcomp(rights, is_pas)
    if potential_new_verb is not None and potential_new_objs is not None and len(potential_new_objs) > 0:
        objs.extend(potential_new_objs)
        v = potential_new_verb
    if len(objs) > 0:
        objs.extend(_get_objs_from_conjunctions(objs))
    if len(objs) == 0:
        tmp = nlp("None")
        objs.extend(tmp)
    return v, objs


# return true if the sentence is passive - at he moment a sentence is assumed passive if it has an auxpass verb
def _is_passive(tokens):
    for tok in tokens:
        if tok.dep_ == "auxpass" or (tok.pos_ == "AUX" and tok.dep_ not in {"xcomp","ROOT"}) or tok.text.lower() in {"get","gets"} or (tok.pos_ == "ADJ" and tok.dep_ == "ROOT"):
            return True
        if tok.pos_ == "VERB" and tok.dep_ not in {"auxpass", "amod"}:
            return False
    return False


# resolve a 'that' where/if appropriate
def _get_that_resolution(toks):
    for tok in toks:
        if 'that' in [t.orth_ for t in tok.lefts]:
            return tok.head
    return toks


# simple stemmer using lemmas
def _get_lemma(word: str):
    tokens = nlp(word)
    if len(tokens) == 1:
        return tokens[0].lemma_
    return word


# print information for displaying all kinds of things of the parse tree
def printDeps(toks):
    for tok in toks:
        print(tok.orth_, tok.dep_, tok.pos_, tok.head.orth_, [t.orth_ for t in tok.lefts], [t.orth_ for t in tok.rights])


# expand an obj / subj for an activity combination by its additions (e.g. invoice -> blue invoice)
def expand_activity_object(item, tokens, visited):
    if item.lower_ == 'that':
        item = _get_that_resolution(tokens)

    parts = []

    if hasattr(item, 'lefts'):
        for part in item.lefts:
            if part.pos_ in BREAKER_POS and part.dep_ == "vbz":
                break
            if part.dep_ == "amod":
                parts.append(part)
    parts.append(item)
    return parts


#expand an obj / subj np using its chunk
def expand(item, tokens, visited):
    #if item.lower_ == 'that':
    #    item = _get_that_resolution(tokens)

    parts = []

    if hasattr(item, 'lefts'):
        for part in item.lefts:
            if part.pos_ in BREAKER_POS and part.dep_ == "vbz" or (part.pos_ == 'prep' and part.text.lower_ != "of"):
                break
            if not (part.lower_ in NEGATIONS or part.pos_ == "DET" or part.pos_ == "PUNCT") or (part.pos_ == "VERB" and part.dep_ == "amod"):
                parts.append(part)
    parts.append(item)

    if hasattr(item, 'rights'):
        for part in item.rights:
            if part.pos_ in BREAKER_POS or part.text.lower() == "of":
                found = 0
                for tok in item.rights:
                    if found == 0:
                        if tok.dep_ == 'prep' and tok.text.lower() == 'for':
                            break
                        if tok.pos_ == 'DET' and tokens[tok.i + 1].dep_ == 'pobj':
                            parts.append(tok)
                        if tok.dep_ == 'prep':
                            parts.append(tok)
                        if tok.dep_ == 'pobj':
                            parts.append(tok)
                            found = 1     
                break                   
            if (part.lower_ not in NEGATIONS or part.pos_ == "DET") and part.pos_ != "ADP":
                parts.append(part)
    if hasattr(parts[-1], 'rights'):
        for item2 in parts[-1].rights:
            if item2.pos_ == "DET" or item2.pos_ == "NOUN":
                if item2.i not in visited:
                    visited.add(item2.i)
                parts.extend(expand(item2, tokens, visited))
            break
    return parts

# convert a list of tokens to a string
def to_str(tokens):
    return ' '.join([item.text for item in tokens])


# find verbs and their subjects / objects to create SVOs, detect passive/active sentences
def findSVOs(tokens):
    if isinstance(tokens, str):
        tokens = nlp(tokens)
    svos = []
    verbs = [tok for tok in tokens if _is_non_aux_verb(tok)]
    if not verbs:
        verbs = [tok for tok in tokens if (tok.pos_ == "AUX" and tok.dep_ == "xcomp")]
    visited = set()  # recursion detection
    for v in verbs:
        verb_phrase = ' '.join([item.text for item in v.subtree])
        verb_phrase = nlp(verb_phrase)
        is_pas = _is_passive(verb_phrase)
        subs, verbNegated = _get_all_subs(v)
        if len(subs) > 0:
            isConjVerb, conjV = _right_of_verb_is_conj_verb(v)
            if isConjVerb:
                v2, objs = _get_all_objs(conjV, is_pas)
                for sub in subs:
                    for obj in objs:
                        objNegated = _is_negated(obj)
                        if is_pas:  # reverse object / subject for passive
                            svos.append(("",
                                         "don't " + v.lemma_ if verbNegated or objNegated else v.lemma_, to_str(expand(sub, tokens, visited))))
                            svos.append(("",
                                         "don't " + v2.lemma_ if verbNegated or objNegated else v2.lemma_, to_str(expand(sub, tokens, visited))))
                        else:                          
                            svos.append((to_str(expand(sub, tokens, visited)),
                                         "don't " + v.lemma_ if verbNegated or objNegated else v.lemma_, to_str(expand(obj, tokens, visited))))
                            svos.append((to_str(expand(sub, tokens, visited)),
                                         "don't " + v2.lemma_ if verbNegated or objNegated else v2.lemma_, to_str(expand(obj, tokens, visited))))
            else:
                v, objs = _get_all_objs(v, is_pas)
                for sub in subs:
                    for obj in objs:
                        objNegated = _is_negated(obj)
                        if is_pas:  # reverse object / subject for passive
                            svos.append(("",
                                         "don't " + v.lemma_ if verbNegated or objNegated else v.lemma_, to_str(expand(sub, tokens, visited))))
                        else:         
                            svos.append((to_str(expand(sub, tokens, visited)),
                                         "don't " + v.lemma_ if verbNegated or objNegated else v.lemma_ , to_str(expand(obj, tokens, visited))))
        else:
            isConjVerb, conjV = _right_of_verb_is_conj_verb(v)
            if isConjVerb:
                v2, objs = _get_all_objs(conjV, is_pas)        
                for obj in objs:
                        objNegated = _is_negated(obj)
                        if is_pas:  # reverse object / subject for passive
                            svos.append(("",
                                         "don't " + v.lemma_ if verbNegated or objNegated else v.lemma_, to_str(expand(sub, tokens, visited))))
                            svos.append(("",
                                         "don't " + v2.lemma_ if verbNegated or objNegated else v2.lemma_, to_str(expand(sub, tokens, visited))))
                        else:                          
                            svos.append(("",
                                         "don't " + v.lemma_ if verbNegated or objNegated else v.lemma_, to_str(expand(obj, tokens, visited))))
                            svos.append(("",
                                         "don't " + v2.lemma_ if verbNegated or objNegated else v2.lemma_, to_str(expand(obj, tokens, visited))))
            else:
                v, objs = _get_all_objs(v, is_pas)
                for obj in objs:
                    objNegated = _is_negated(obj)
                    if is_pas:  # reverse object / subject for passive
                        svos.append(("",
                                        "don't " + v.lemma_ if verbNegated or objNegated else v.lemma_, to_str(expand(obj, tokens, visited))))
                    else:
                        svos.append(("",
                                        "don't " + v.lemma_ if verbNegated or objNegated else v.lemma_ , to_str(expand(obj, tokens, visited))))


    for x in svos:
        lst = list(x)
        svos[svos.index(x)] = tuple(lst)
    
    svos = list(set(svos))

    svos = _combine_svos_over_same_verb(svos, tokens)
    return svos

def _get_svos_sents(sentences):
    nlp = spacy.load('en_core_web_sm')
    svos = []
    for x in sentences:
        tmp = nlp(x)
        svos.append(findSVOs(tmp))
    return svos

def _get_svos_sent(sentence):
    nlp = spacy.load('en_core_web_sm')
    tmp = nlp(sentence)
    svos = findSVOs(tmp)
    return svos


def _combine_svos_over_same_verb(svos, tokens):
    svos_new = []
    for key, group in groupby(svos, lambda x: x[1]):
        groupsi = []
        for thing in group:
            groupsi.append(thing[2])
        combine = ""
        for x in groupsi:
            if groupsi.index(x) == 0:
                combine = x
            elif groupsi.index(x) == (len(groupsi) - 1):
                for chunk in tokens.noun_chunks:
                    if (x == chunk[1:].text or x == chunk[0:].text) and tokens[chunk[0].i - 1].text == "and":
                        combine = combine + " and " + x
                    elif (x == chunk[1:].text or x == chunk[0:].text) and tokens[chunk[0].i - 1].text == "or":
                        combine = combine + " or " + x   
            else:
                combine = combine + ", " + x
        svo = (thing[0],thing[1],combine)
        svos_new.append(svo)
    return(svos_new)