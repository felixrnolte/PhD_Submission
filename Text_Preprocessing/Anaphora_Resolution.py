from pycorenlp import StanfordCoreNLP

# You have to download the latest StanfordCoreNLP model from https://stanfordnlp.github.io/CoreNLP/index.html#download and call it with the following command, adjusting the path accordingly
# java -mx4g -cp "D:\Felix\Downloads\stanford-corenlp-4.2.0\\*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000

nlp = StanfordCoreNLP('http://localhost:9000')

def resolve(corenlp_output):
    """ Transfer the word form of the antecedent to its associated pronominal anaphor(s) """
    for coref in corenlp_output['corefs']:
        mentions = corenlp_output['corefs'][coref]
        antecedent = mentions[0]  # the antecedent is the first mention in the coreference chain
        for j in range(1, len(mentions)):
            mention = mentions[j]
            if mention['type'] == 'NOMINAL' and mention['text'] != antecedent['text']:
                antecedent = mention
            if mention['type'] == 'PRONOMINAL':
                # get the attributes of the target mention in the corresponding sentence
                target_sentence = mention['sentNum']
                target_token = mention['startIndex'] - 1
                # transfer the antecedent's word form to the appropriate token in the sentence
                corenlp_output['sentences'][target_sentence - 1]['tokens'][target_token]['word'] = antecedent['text']

def get_resolved(text):
    output_text = ''
    corenlp_output = nlp.annotate(text, properties= {'timeout': '50000','annotators':'dcoref','outputFormat':'json','ner.useSUTime':'false'})
    #print(corenlp_output)
    resolve(corenlp_output)
    possessives = ['hers', 'his', 'their', 'theirs','ours','yours']
    for sentence in corenlp_output['sentences']:
        for token in sentence['tokens']:
            output_word = token['word']
            # check lemmas as well as tags for possessive pronouns in case of tagging errors
            if token['lemma'] in possessives or token['pos'] == 'PRP$':
                output_word += "'s"  # add the possessive morpheme
            output_word += token['after']
            #print(output_word, end='')
            output_text += output_word
    #print(output_text)
    return output_text
