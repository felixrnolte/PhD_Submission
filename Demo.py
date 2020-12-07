# This is a demo file that can be executed with a set of example inputs to show the transformation between text and model step by step. The example inputs are taken from the set of models and descriptions used for the evaluation. The inputs corresponding to the entries 1,4,5,9 and 10 of that dataset.

# 0) Preparations

# You have to download the latest StanfordCoreNLP model from https://stanfordnlp.github.io/CoreNLP/index.html#download and call it with the following command, adjusting the path accordingly
# java -mx4g -cp "D:\Felix\Downloads\stanford-corenlp-4.2.0\\*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000

# a) Test Inputs

import pandas

test_01 = "From the product interest, the offer is being prepared. The offer is then either accepted or the offer is declined. From the accepted offer, the order is sent, for the completion of the delivery."

test_04 = "The received invoice is booked to the accounts payable. Then, the booked invoice is paid, by the accountant. From the sent invoice, the accountant books the invoice to accounts receivable. The outgoing booked invoice is then checked for payment by the accountant for either a payment reminder or a payment confirmation. The due salaries are booked by the accountant. Then he pays the salaries. With the paid invoice, the received payment, and the paid salaries, he then books the bank statement. Lastly, the accountant declares the monthly taxes."

test_05 = "With an express package, we get the capabilities first. With the retrieved capabilities, the warehouse operator selects a specific service. Then, he books a pickup. With the booked pickup, he downloads the shipping label. Following, he prints the label. Lastly, the label is attached to the parcel, for labeled package."

test_09 = "With a new series, the designer first checks the popularity for a popular or non-popular series. For the popular series we then negotiate a license and get either an agreement or no agreement. Then we sign the license contract for the license and new license product. After this, we design new motives. The portfolio is expanded with the new motives."

test_10 = "We send the invoice that was prepared."

test_tmp = "After Laura sends the invoice for the notice, Marc approves the invoice."

test_rndm = "Marc registers the order from the mail. Then, he checks the order that was registered. If the order is fine, Laura prepares the list of suppliers. If the order is declined, Alex sends a notice to the customer. Before the accountant ships the product, the documents get signed based on the list of suppliers."

# Set on test case as input for the demo
input = test_tmp

# 1) Text Pre-Processing
# a Anaphora Resolution
from Text_Preprocessing.Anaphora_Resolution import get_resolved

proc_text = get_resolved(input)
print("=============== Anaphora Resolution - DONE ===============")

# c) Sentence Splitter
from Text_Preprocessing.Sentence_Splitting import _get_sentences_spacy

sentences = _get_sentences_spacy(proc_text)

print("=============== Sentence Splitter - DONE ===============")

# 2) Linguistic Feature Extraction
# For each sentence now phrases are extracted, keywords of these identified and process relevant information extracted

# a) Phrase Extraction
from Linguistic_Feature_Extraction.Phrase_Extraction import _get_phrases_of_sentences

phrases = _get_phrases_of_sentences(sentences)

print(phrases)
print("=============== Phrase Extraction - DONE ===============")

# b) Information Extraction (using also keywords)
from Linguistic_Feature_Extraction.Information_Extraction import _get_ling_information

features = _get_ling_information(phrases)

#features.to_csv(r'Data\features.csv', index=False)

print(features)
print(features.iloc[0][1])
print("=============== Information Extraction- DONE ===============")

# 3) Model Element Mapping
from Model_Element_Mapping.Mapping import model_element_mapping

mapping_df = model_element_mapping(features)

print(mapping_df)
print("=============== Mapping - DONE ===============")
# 4) Model Generation

from Model_Generation.Generation import _connect_parts

model = _connect_parts(mapping_df)

print(model)
print("=============== Generation - DONE ===============")