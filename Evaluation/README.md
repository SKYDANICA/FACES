## For the BLEU and Sentence-BERT metrics, you can open the Bert_SentenceBert file and obtain the results by running the command 
    python evalue.py
    Note that you need to modify the paths on lines 66 and 67.

## For the ROUGE and METEOR metrics, open the Rouge_Meteor file and run `python evalue.py`.
To obtain METEOR and ROUGE-L, we need to activate the environment that contains python 2.7

    conda activate py27
    python evaluate.py 
Note that you need to modify the paths on lines 40 and 41.
Tip: The path should only contain English characters.