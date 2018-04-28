Installation
------------

	1. Install NLTK, StanfordCoreNLP
	2. Download stanford-corenlp, stanford-parser, stanford-postagger, stanford-ner
	3. Set CLASSPATH & STANFORD_MODELS var in python environment (code already does this)
	4. Set JAVAHOME (You might not need this)
	5. for negtool: pip install pystruct cvxopt networkx==1.11 sklearn scipy
	6. update all paths in config.txt

Negtool
-------
	1.fix_reviews_json.py to add punctuation to each review
	2.run_negtool.py to get negscope

SA Pipeline
-----------
	1. run.py to run SAPipeline