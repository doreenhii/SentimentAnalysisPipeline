from pycorenlp import StanfordCoreNLP
import json
from nltk.tree import Tree
from SentimentModelFunctions import *

if __name__ == '__main__':
    nlp = StanfordCoreNLP('http://localhost:9000')


    #14, 37, 58, 97, 99
    text = ["Hai"]
    for t in text:
        print("Text: {}".format(t))
        output = nlp.annotate(t, properties={
            'annotators': 'tokenize,ssplit, parse',
            'outputFormat': 'json',
            'parse.model' : 'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'
        })
        print(type(output))
        print(type("CoreNLP request timed out. Your document may be too long."))
        print(json.dumps(output, indent = 4))
        for i in range(len(output['sentences'])):
            tokenized_sent = [token_json['word'] for token_json in output['sentences'][i]['tokens']] 
            print(tokenized_sent)
            parsetree = Tree.fromstring(output['sentences'][i]['parse'])
            print(parsetree)
            print(getTreeIndices(parsetree))
            print(map_valence_tree({}, parsetree))
            #print(tokenized_sent)
            #print(parsetree)

        print("\n\n")