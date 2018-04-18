import os, json, itertools,re
from nltk.parse.stanford import StanfordParser
from SentimentModelFunctions import *
import numpy as np
import pprint, nltk

def split_sentence_flat(sentence):
    contractions = ["n't", "'m", "'ll","'d","'s","'ve","'re"]

    sentence_sequence = []

    #split sentences on delimiters { " ", "--" } Any others?
    split_sentence = re.split(" |--", sentence)
    
    #split contractions
    for word in split_sentence:
        hasCont = False
        for cont in contractions:
            if(cont in word):
                sentence_sequence.append(word[:-len(cont)])
                sentence_sequence.append(cont)
                hasCont = True
                break
        if not hasCont:
            sentence_sequence.append(word)

    return sentence_sequence

#i'm not sure if you need different split functions for {flat vs parse}
def split_sentence_parse(sentence):
    contractions = ["n't", "'m", "'ll","'d","'s","'ve","'re"]

    sentence_sequence = []

    #split sentences on delimiters { " ", "--" } Any others?
    split_sentence = re.split(" |--", sentence)
    
    #split contractions
    for word in split_sentence:
        hasCont = False
        for cont in contractions:
            if(cont in word):
                sentence_sequence.append(word[:-len(cont)])
                sentence_sequence.append(cont)
                hasCont = True
                break
        if not hasCont:
            sentence_sequence.append(word)

    return sentence_sequence

class SAModel:  
    def __init__(self):
        self.sentence = ""
        self.sentence_sequence = []
        self.valence_sequence = []
        self.sentence_tree = []
        self.valence_tree = []
        self.CompleteWordIndices = []


        # config_data = open(os.path.join(os.getcwd(),"config.txt"), "r")
        # config_data_json = json.load(config_data)
        # config_data.close()
        config_data_json ={"ROOT_PATH": "C:\\Users\\King\\Desktop\\SentimentAnalysisPipeline\\"}

        root_path = config_data_json["ROOT_PATH"]
        dict_paths = {"main_val_dict" : os.path.join(*[root_path,"Data","Processed","Dictionaries","main_valence_dictionary.json"])}
        #              "wordnet_antonym" : root_path + "Data/Processed/Dictionaries/wordnet_antonym.json",
        #              "negfirst_val_dict" : root_path + "Data/Processed/Dictionaries/negfirst_valence_dictionary.json",
        #              "neg_val_dict" : root_path + "Data/Processed/Dictionaries/neg_valence_dictionary.json"}
        negtool_negscopes_path = os.path.join(*[root_path,"SAModel","output","test_sentences_neg_scopes.txt"])
        
        json_file = open(dict_paths["main_val_dict"]) #917763 terms
        self.VALENCE_DICT = json.loads(json_file.read())
        json_file.close()
        
        #json_file = open(dict_paths["wordnet_antonym"]) #3310 terms
        #self.ANTONYM_DICT = json.loads(json_file.read())

        #json_file = open(dict_paths["negfirst_val_dict"]) #2935 terms
        #self.NEGFIRST_DICT = json.loads(json_file.read())

        #json_file = open(dict_paths["neg_val_dict"]) #13584 terms
        #self.NEG_DICT = json.loads(json_file.read())

        negtool_neg_scopes = open(negtool_negscopes_path, "r")
        self.NT_neg_scopes = json.loads(negtool_neg_scopes.read())

        self.PARSER = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

        self.model = ""
        self.neg_scope_method = ""
        self.neg_res_method = ""
        self.sent_comp_method = ""

        self.window_size = 3
        self.sentence_id = 0 #for negtool purposes
        
        #constants
        self.contractions = ["n't", "'m", "'ll","'d","'s","'ve","'re"]

    def setModel(self, name = "NONE"):
        self.model = name
        self.neg_scope_method = name.split(" ")[0]
        self.neg_res_method = name.split(" ")[1]
        self.sent_comp_method = name.split(" ")[2]

    def printModel(self):
        print(" ".join([self.neg_scope_method, self.neg_res_method, self.sent_comp_method]))

    def setSentence(self, sent):
        self.sentence = sent

        if(self.model == "NONE" or self.sent_comp_method == "FLAT"):
            self.sentence_sequence = nltk.word_tokenize(self.sentence)#split_sentence_flat(self.sentence)
            self.valence_sequence = [getValence(self.VALENCE_DICT, word) for word in self.sentence_sequence]

        elif(self.sent_comp_method == "PARSETREE"):
            #if we split contractions and replace them with the full words, would that affect the parser?
            self.sentence_tree = list(self.PARSER.raw_parse(self.sentence))[0]
            self.valence_tree = map_valence_tree(self.VALENCE_DICT, self.sentence_tree)
            self.sentence_sequence = nltk.word_tokenize(self.sentence)#split_sentence_flat(self.sentence)
            self.completeTreeIndices = get_word_indices(self.sentence_tree)

            

    def detectNegScope(self):
        if(self.neg_scope_method == "PARSETREE" and self.sent_comp_method == "PARSETREE"):
            self.neg_scope = detect_neg_scope_tree(self.sentence_tree, parent_index = [])

        elif(self.neg_scope_method == "WINDOW" and self.sent_comp_method == "FLAT"):
            self.neg_scope = resolve_double_negative(detect_neg_scope_window(self.sentence_sequence, self.window_size))

        elif(self.neg_scope_method == "NEGTOOL" and self.sent_comp_method == "FLAT"):
            self.neg_scope = resolve_double_negative( list(itertools.chain(*self.NT_neg_scopes[str(self.sentence_id)]["neg_scope"])))
            
        elif(self.neg_scope_method == "WINDOW" and self.sent_comp_method == "PARSETREE"):
            self.neg_scope = resolve_double_negative(detect_neg_scope_window(self.sentence_sequence, self.window_size)) 
            self.neg_scope = SequenceToTreeIndices(self.completeTreeIndices, self.neg_scope)

        elif(self.neg_scope_method == "NEGTOOL" and self.sent_comp_method == "PARSETREE"):
            self.neg_scope = resolve_double_negative( list(itertools.chain(*self.NT_neg_scopes[str(self.sentence_id)]["neg_scope"])))
            self.neg_scope = SequenceToTreeIndices(self.completeTreeIndices, self.neg_scope)

        else:
            self.neg_scope = []
    
    def neg_res(self):
        if(self.neg_scope_method != "PARSETREE" and self.neg_res_method == "ANTONYM_LOOKUP" and self.sent_comp_method == "PARSETREE"):
                
                #get linear sequence
                sequence_indices = TreeToSequenceIndices(self.completeTreeIndices, self.neg_scope)
                
                #go through valence_tree to resolve negation
                for i in range(len(self.neg_scope)):
                    tree_index = self.neg_scope[i]
                    temp = self.valence_tree

                    for j in tree_index[:-1]:
                        temp = temp[j]
                    
                    antonym_word = self.sentence_sequence[sequence_indices[i]]
                    if(antonym_word is None):
                        negated_valence = -999
                    else:
                        negated_valence = antonym_lookup_negate(self.VALENCE_DICT, antonym_word)

                    temp[tree_index[-1]] = negated_valence

        elif(self.sent_comp_method == "FLAT"):

            if(self.neg_res_method == "ANTONYM_LOOKUP"):
                #go through valence_sequence to resolve negation
                for neg_index in self.neg_scope:
                        antonym_word = antonym_lookup(self.sentence_sequence[neg_index])
                        if(antonym_word is None):
                            negated_valence = -999
                        else:
                            negated_valence = antonym_lookup_negate(self.VALENCE_DICT, antonym_word)
                        
                        self.valence_sequence[neg_index] = negated_valence
            else:
                for index in self.neg_scope:
                    if(self.valence_sequence[index] != -999):
                        self.valence_sequence[index] = negate(self.valence_sequence[index], self.neg_res_method)
            
            #print("After negating: {}".format(self.valence_sequence)) #[DEBUG]

    def compose(self):
        if(self.model == "NONE" or self.sent_comp_method == "FLAT"):

            sentiment = flat_composition(self.valence_sequence, self.neg_scope, self.neg_res_method)
        
        elif(self.sent_comp_method == "PARSETREE"):

            sentiment = tree_composition(self.valence_tree, [], self.neg_scope, self.neg_res_method)

        return sentiment