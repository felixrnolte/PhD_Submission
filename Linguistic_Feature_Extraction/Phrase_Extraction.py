import en_core_web_sm
import spacy
import pandas as pd
nlp = en_core_web_sm.load()


def _get_phrases_of_sentence(sentence):
    phrase_df = pd.DataFrame(columns =['Core', 'Adverbial','Conditional','Relative'])
    ADVINDS = {"after","before","while","once"}
    sent_tmp = sentence
    nlp_sent = nlp(sent_tmp)
    core = []
    advcl = []
    condcl = []
    relcl = []
    for token in nlp_sent:  
        if token.dep_ == "relcl":
            relcl_tmp = ' '.join([item.text for item in token.subtree])
            relcl.append(relcl_tmp)
            sent_tmp = sent_tmp.replace(relcl_tmp, "")  
        if token.dep_ == "advcl" or (token.dep_ == "prep" and token.pos_ == "ADP" and token.text.lower() in ADVINDS):
            advcl_tmp = ' '.join([item.text for item in token.subtree])
            advcl_nlp = nlp(advcl_tmp)
            advcl_poss = []
            for token in advcl_nlp:
                advcl_poss.append(token.pos_)
            if advcl_nlp[0].text.lower() == "if":
                condcl.append(advcl_tmp)
            elif "VERB" in advcl_poss:
                advcl.append(advcl_tmp)
            sent_tmp = sent_tmp.replace(advcl_tmp, "") 
    core.append(sent_tmp)
    phrase_df = phrase_df.append({'Core': core, 'Adverbial' : advcl, 'Conditional': condcl, 'Relative': relcl}, ignore_index=True)
    return phrase_df

def _get_phrases_of_sentences(sentences):
    dfs = []
    for sentence in sentences:
        tmp = _get_phrases_of_sentence(sentence)  
        dfs.append(tmp)
    result = pd.concat(dfs, ignore_index=True)
    return result