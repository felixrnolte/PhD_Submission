# PhD_Submission

For the execution of the Demo.py or Data_Generation.py the following preprations are required:

* Check and if necessary arrange the following paths in the Data_Generation.py:
    * Line 4 for loading the descriptions from the .csv file.
    * Line 34 for the output of the phrase extraction.
    * Line 40 for the output of the linguistic feature extraction.
    * Line 53 for the output of the model mapping.
    * Line 63 for the output of the model generation/connection.
   
* Paths are assigned with the relative paths and should not require changes in case the repository is imported as a workspace.
   
* Download the latest StanfordCoreNLP model from https://stanfordnlp.github.io/CoreNLP/index.html#download and call it with the command *, adjusting the path accordingly. 

    *The StanfordCoreNLP server is started in the context of the implementation for the submitted thesis with the exemplary command:
    java -mx4g -cp "D:\Felix\Downloads\stanford-corenlp-4.2.0\\*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000
