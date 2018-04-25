import os, json, itertools, re
from nltk.parse.stanford import StanfordParser
from SentimentModelFunctions import *
from pathlib import Path
from stanfordcorenlp import StanfordCoreNLP

def tokenize(corenlp, review, span=False):
    r_dict = corenlp._request('ssplit', review)
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

def set_environment_paths(paths_json):
    
    root_path = paths_json["STANFORD_MODELS_PATH"]
    root = Path(root_path)
    java_path = paths_json["JAVA_PATH"]

    class_paths = ["stanford-corenlp-full-2018-02-27","stanford-parser-full-2018-02-27","stanford-postagger-2018-02-27","stanford-ner-2018-02-27"]
    model_paths = ["stanford-corenlp-full-2018-02-27","stanford-parser-full-2018-02-27",os.path.join("stanford-postagger-2018-02-27","models"), os.path.join("stanford-ner-2018-02-27","classifiers")]

    os.environ['CLASSPATH'] = "".join([str(root / path)+";" for path in class_paths])[:-1]
    os.environ['STANFORD_MODELS'] = "".join([str(root / path)+";" for path in model_paths])[:-1]
    os.environ['JAVAHOME'] = java_path

class SAModel:  
    def __init__(self, paths_json):
        set_environment_paths(paths_json)

        self.sentence_sequences = []
        self.valence_sequences = []
        self.sentence_trees = []
        self.valence_trees = []
        self.CompleteWordIndices = []

        self.model = ""
        self.neg_scope_method = ""
        self.neg_res_method = ""
        self.sent_comp_method = ""
        

        valence_dict_path = paths_json["VALENCE_DICT"]
        json_file = open(valence_dict_path)
        self.VALENCE_DICT = json.loads(json_file.read())
        json_file.close()

        negtool_negscopes_path = paths_json["NEGTOOL_NEGSCOPE"]
        negtool_neg_scopes_file = open(negtool_negscopes_path, "r")
        self.NT_neg_scopes = json.loads(negtool_neg_scopes_file.read())
        negtool_neg_scopes_file.close()

        #window neg scope
        self.window_size = 3

        self.review_id = 0
        self.sentence_id = 0 #for negtool purposes
        
        #constants
        self.contractions = ["n't", "'m", "'ll","'d","'s","'ve","'re"]

        #parser and tokenizer initialization
        self.PARSER = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
        self.nlp = StanfordCoreNLP(paths_json["PATH_TO_CORENLP"])
        
    def setModel(self, name = "NONE"):
        self.model = name
        self.neg_scope_method = name.split(" ")[0]
        self.neg_res_method = name.split(" ")[1]
        self.sent_comp_method = name.split(" ")[2]

    def printModel(self):
        print(" ".join([self.neg_scope_method, self.neg_res_method, self.sent_comp_method]))

    def setReview(self, review):

        self.sentence_sequences = tokenize(self.nlp, review)
        self.review_size = len(self.sentence_sequences) #number of sentences

        if(self.model == "NONE" or self.sent_comp_method == "FLAT"):
            
            self.valence_sequences = []
            for sentence in self.sentence_sequences:
                self.valence_sequences.append([getValence(self.VALENCE_DICT, word) for word in sentence])

        elif(self.sent_comp_method == "PARSETREE"):
            
            split_review = [" ".join(tokenized_sent[:-1])+tokenized_sent[-1] for tokenized_sent in self.sentence_sequences]
            self.sentence_trees = []
            self.valence_trees = []
            self.completeTreesIndices = []
            for sentence in split_review:
                self.sentence_trees.append(list(self.PARSER.raw_parse(sentence))[0])
                self.valence_trees.append(map_valence_tree(self.VALENCE_DICT, self.sentence_trees[-1]))
                self.completeTreesIndices.append(getTreeIndices(self.sentence_trees[-1]))
            

    def detectNegScope(self):
        self.neg_scopes = []

        for i in range(len(self.sentence_sequences)):

            if(self.neg_scope_method == "PARSETREE" and self.sent_comp_method == "PARSETREE"):
                self.neg_scopes.append(detect_neg_scope_tree(self.sentence_trees[i], parent_index = []))

            elif(self.neg_scope_method == "WINDOW" and self.sent_comp_method == "FLAT"):
                self.neg_scopes.append(resolve_double_negative(detect_neg_scope_window(self.sentence_sequences[i], self.window_size)))

            elif(self.neg_scope_method == "NEGTOOL" and self.sent_comp_method == "FLAT"):
                self.neg_scopes.append(resolve_double_negative( list(itertools.chain(*self.NT_neg_scopes[str(self.sentence_id+i)]["neg_scope"]))))
            
            elif(self.neg_scope_method == "WINDOW" and self.sent_comp_method == "PARSETREE"):
                negscope = resolve_double_negative(detect_neg_scope_window(self.sentence_sequences[i], self.window_size)) 
                self.neg_scopes.append(SequenceToTreeIndices(self.completeTreesIndices[i], negscope))

            elif(self.neg_scope_method == "NEGTOOL" and self.sent_comp_method == "PARSETREE"):
                negscope = resolve_double_negative( list(itertools.chain(*self.NT_neg_scopes[str(self.sentence_id+i)]["neg_scope"])))
                self.neg_scopes.append(SequenceToTreeIndices(self.completeTreesIndices[i], negscope))
                
    
    def neg_res(self): #this function is just for antonym_lookup neg res

        for i in range(len(self.sentence_sequences)):

            if(self.neg_scope_method != "PARSETREE" and self.neg_res_method == "ANTONYM_LOOKUP" and self.sent_comp_method == "PARSETREE"):
                    
                    #get linear sequence
                    sequence_indices = TreeToSequenceIndices(self.completeTreesIndices[i], self.neg_scopes[i])
                    
                    #go through valence_tree to resolve negation
                    for j in range(len(self.neg_scopes[i])):
                        tree_index = self.neg_scopes[i][j]
                        temp = self.valence_trees[i]

                        for k in tree_index[:-1]:
                            temp = temp[k]

                        antonym_word = self.sentence_sequences[i][sequence_indices[j]]
                        if(antonym_word is None):
                            negated_valence = -999
                        else:
                            negated_valence = antonym_lookup_negate(self.VALENCE_DICT, antonym_word)
                        
                        temp[tree_index[-1]] = negated_valence

            elif(self.sent_comp_method == "FLAT"):
                

                if(self.neg_res_method == "ANTONYM_LOOKUP"):

                    #go through valence_sequence to resolve negation
                    for neg_index in self.neg_scopes[i]:
                            antonym_word = antonym_lookup(self.sentence_sequences[i][neg_index])
                            if(antonym_word is None):
                                negated_valence = -999
                            else:
                                negated_valence = antonym_lookup_negate(self.VALENCE_DICT, antonym_word)
                            
                            self.valence_sequences[i][neg_index] = negated_valence
                else:
                    for index in self.neg_scopes[i]:
                        if(self.valence_sequences[i][index] != -999):
                            self.valence_sequences[i][index] = negate(self.valence_sequences[i][index], self.neg_res_method)
            else:
                """
                [ ] NEGTOOL SYM_INVERT PARSETREE
                [ ] NEGTOOL SYM_SHIFT PARSETREE
                [ ] NEGTOOL ASYM_SHIFT PARSETREE
                [ ] WINDOW SYM_INVERT PARSETREE
                [ ] WINDOW SYM_SHIFT PARSETREE
                [ ] WINDOW ASYM_SHIFT PARSETREE
                [ ] PARSETREE SYM_INVERT PARSETREE
                [ ] PARSETREE SYM_SHIFT PARSETREE
                [ ] PARSETREE ASYM_SHIFT PARSETREE
                """
                pass

    def compose(self):

        sentiment_scores = []

        for i in range(len(self.sentence_sequences)):

            if(self.model == "NONE" or self.sent_comp_method == "FLAT"):

                sentiment = flat_composition(self.valence_sequences[i])
            
            elif(self.sent_comp_method == "PARSETREE"):

                sentiment = tree_composition(self.valence_trees[i], [], self.neg_scopes[i], self.neg_res_method)

            sentiment_scores.append(sentiment)

        sentiment_scores = list(filter(lambda a: a != -999, sentiment_scores))
        
        if(len(sentiment_scores) == 0):
            avg_valence = None
        else:
            avg_valence = sum(sentiment_scores)/float(len(sentiment_scores))

        return avg_valence


##################################################################
# Store Results
##################################################################


class SAResults:
    def __init__(self, paths_json):

        self.results_json = {}

        root_path = paths_json["ROOT_PATH"]

        models_list_filepath = Path(root_path) / "models_to_run.txt"
        models_list_file = open(models_list_filepath, "r")
        models_list = models_list_file.readlines()
        models_list_file.close()

        for model in models_list:
            self.results_json[model.rstrip("\n")] = {
                    "category_results" : [0,0,0,0,0],
                    "correct_binary_results" : 0 ,
                    "incorrect_binary_results" : 0 ,
                    "mse" : 0,
                    "binary_results_dict" : 
                    {
                        "true_positive" : 0, "false_negative" : 0, "false_positive" : 0, "true_negative" : 0,
                        "true_neutral" : 0, "positified_neutral" : 0, "negatified_neutral" : 0,
                        "neutralized_positive" : 0, "neutralized_negative" : 0
                    }
                }

        #in depth results variables
        self.category_difference = []
        self.correct = [] #1 for correct, 0 for incorrect
 
    def categorize_prediction(self, model, truth, predicted_value):
        if (-1. <= predicted_value < -.6):
            result = 1
        elif (-.6 <= predicted_value < -.2):
            result = 2
        elif (-.2 <= predicted_value < .2):
            result = 3
        elif(.2 <= predicted_value < .6):
            result = 4
        else:
            result = 5

        diff = int(abs(result-truth))

        self.results_json[model]["category_results"][diff] += 1
        self.results_json[model]["mse"] += diff*diff
        self.check_truth(model, truth, result)

        #indepth results
        self.category_difference.append(diff)
        
    def binary_prediction(self, model, truth, predicted_value):
        normalized_truth = (truth - 3)/2.0 

        if(normalized_truth >= 0 and predicted_value < 0) or (normalized_truth < 0 and predicted_value >= 0): #consider 0 truth as positive
            self.results_json[model]["incorrect_binary_results"] += 1 #incorrect
            self.correct.append(0) #1 for correct, 0 for incorrect
        else:
            self.results_json[model]["correct_binary_results"] += 1 #correct
            self.correct.append(1) #1 for correct, 0 for incorrect
        


    def get_in_depth_results(self):
        results = (self.category_difference, self.correct)
        #in depth results variables
        self.category_difference = []
        self.correct = [] #1 for correct, 0 for incorrect

        return results

    def check_truth(self, model, truth, result):
        
        binary_results_dict = self.results_json[model]["binary_results_dict"]

        if (4<= truth <= 5) and (4<= result <= 5):
            binary_results_dict["true_positive"] += 1

        elif (4<= truth <= 5) and (1<= result <= 2):
            binary_results_dict["false_negative"] += 1

        elif (4<= truth <= 5) and (result ==3):
            binary_results_dict["neutralized_positive"] += 1

        elif (1<= truth <= 2) and (4<= result <= 5):
            binary_results_dict["false_positive"] += 1

        elif (1<= truth <= 2) and (1<= result <= 2):
            binary_results_dict["true_negative"] += 1

        elif (1<= truth <= 2) and (result ==3):
            binary_results_dict["neutralized_negative"] += 1

        elif (truth == 3) and (result ==3):
            binary_results_dict["true_neutral"] += 1

        elif (truth == 3) and (4<= result <= 5):
            binary_results_dict["positified_neutral"] += 1

        elif (truth == 3) and (1<= result <= 2):
            binary_results_dict["negatified_neutral"] += 1

    def write_to_file(self):
        import datetime
        outfile = open(Path("results") / ("sa_results-"+str(datetime.date.today())+".json"),"w")
        json.dump(self.results_json, outfile , indent=4)
        outfile.close()

"""
self.results_json[model]["binary_results_dict"][]
self.results_json = 
        {
            "MODEL_NAME" : 
            {
                "category_results" : [0,0,0,0,0],
                "correct_binary_results" : 0 ,
                "incorrect_binary_results" : 0 ,
                "binary_results_dict" : 
                {
                    "true_positive" : 0, "false_negative" : 0, "false_positive" : 0, "true_negative" : 0,
                    "true_neutral" : 0, "positified_neutral" : 0, "negatified_neutral" : 0,
                    "neutralized_positive" : 0, "neutralized_negative" : 0
                }
            },
            "MODEL_NAME_2" :
            {...}

        }


"""

"""
# to write to file at the end, 
# outfile = open("precision_data.txt","w")
# outfile.write('Truth stats: \t\t\t Predicted\n\t\t\t\t\tPositive\t\tNegative\t\tNeutral\n')
# outfile.write('\t\t\tPositive: '+str(binary_results_dict["true_positive"])+'\t\t'+str(binary_results_dict["false_negative"])+'\t\t'+str(binary_results_dict["neutralized_positive"])+'\n')
# outfile.write('Observed\tNegative: '+str(binary_results_dict["false_positive"])+'\t\t\t\t'+str(binary_results_dict["true_negative"])+'\t\t'+str(binary_results_dict["neutralized_negative"])+'\n')
# outfile.write('\t\t\tNeutral:  '+str(binary_results_dict["positified_neutral"])+'\t\t'+str(binary_results_dict["negatified_neutral"])+'\t\t'+str(binary_results_dict["true_neutral"])+'\n\n')
# outfile.close()

"""


