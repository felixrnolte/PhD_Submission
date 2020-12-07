import pandas as pd


descriptions_df = pd.read_csv(r"Data\Descriptions.csv", sep='; ')
id = 1

for index, row in descriptions_df.iterrows():
    description = row[0]
    print("=============== MODEL " + str(id) + "===============")
    print(description)
    # 1) Text Pre-Processing
    # a Anaphora Resolution
    from Text_Preprocessing.Anaphora_Resolution import get_resolved

    proc_text = get_resolved(description)
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
    phrases.to_csv(r'Data\Phrases\Phrases_' + str(id) + '.csv', index = False)
    print("=============== Phrase Extraction - DONE ===============")

    # b) Information Extraction (using also keywords)
    from Linguistic_Feature_Extraction.Information_Extraction import _get_ling_information

    features = _get_ling_information(phrases)
    features.to_csv(r'Data\Features\Features_' + str(id) + '.csv', index = False)

    #features.to_csv(r'Data\features.csv', index=False)

    print(features)
    print("=============== Information Extraction- DONE ===============")

    # 3) Model Element Mapping
    from Model_Element_Mapping.Mapping import model_element_mapping

    mapping_df = model_element_mapping(features)

    mapping_df.to_csv(r'Data\Mapping\Model_' + str(id) + '.csv', index = False)
    print(mapping_df)
    print("=============== Mapping - DONE ===============")
    # 4) Model Generation

    from Model_Generation.Generation import _connect_parts

    model = _connect_parts(mapping_df)

    print(model)
    model.to_csv(r'Data\Generation\Model_' + str(id) + '.csv', index = False)
    print("=============== Generation - DONE ===============")
    id = id + 1