def set_environment_paths(paths_json):
    
    root_path = paths_json["STANFORD_MODELS_PATH"]
    root = Path(root_path)

    class_paths = ["stanford-corenlp-full-2018-02-27","stanford-parser-full-2018-02-27","stanford-postagger-2018-02-27","stanford-ner-2018-02-27"]
    model_paths = ["stanford-corenlp-full-2018-02-27","stanford-parser-full-2018-02-27",os.path.join("stanford-postagger-2018-02-27","models"), os.path.join("stanford-ner-2018-02-27","classifiers")]

    os.environ['CLASSPATH'] = "".join([str(root / path)+":" for path in class_paths])[:-1]
    os.environ['STANFORD_MODELS'] = "".join([str(root / path)+":" for path in model_paths])[:-1]


def tokenize(corenlp, review, span=False):
    r_dict = corenlp._request('ssplit', review)
    tokens2 = StanfordTokenizer().tokenize(review)
    print(r_dict)
    print(tokens2)
    tokens = [token['word'] for s in r_dict['sentences'] for token in s['tokens']]

    sentences = []
    current_sentence = []
    for token in tokens:
        if(not bool(re.compile(r'[^\!\?]').search(token)) or token == "."): #only ! or ?
            current_sentence.append(token)
            sentences.append(current_sentence)
            current_sentence = []
        else:
            current_sentence.append(token)

    #return [" ".join(sentence[:-1])+sentence[-1] for sentence in sentences] #return sentences
    return sentences #return tokenized sentences

import os, json, itertools, re
from nltk.parse.stanford import StanfordParser
from SentimentModelFunctions import *
from pathlib import Path
from stanfordcorenlp import StanfordCoreNLP
from nltk.tokenize.stanford import StanfordTokenizer
paths_file = open(Path(os.getcwd()) / "config.txt", "r")
paths_json = json.load(paths_file)
paths_file.close()

set_environment_paths(paths_json)

review = "Hurray..... Lol like seriously -- what the heck!!? what are you talking about...? what do you mean..? i mean this!?>"
PARSER = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
print(StanfordTokenizer().tokenize(review))
# for things in list(PARSER.parse_sents(review)):
# 	for thing in things:
# 		print(thing)