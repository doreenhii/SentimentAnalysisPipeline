import re
import time
import json
import os
from SAModel import *
from nltk.parse.stanford import StanfordParser


# config_data = open(os.path.join(os.getcwd(),"config.txt"), "r")
# config_data_json = json.load(config_data)
# config_data.close()
config_data_json ={"ROOT_PATH": "C:\\Users\\King\\Desktop\\SentimentAnalysisPipeline\\"}
root_path = config_data_json["ROOT_PATH"]


class_paths = ["stanford-corenlp-full-2018-02-27","stanford-parser-full-2018-02-27","stanford-postagger-2018-02-27","stanford-ner-2018-02-27"]
model_paths = ["stanford-corenlp-full-2018-02-27","stanford-parser-full-2018-02-27",os.path.join("stanford-postagger-2018-02-27","models"), os.path.join("stanford-ner-2018-02-27","classifiers")]

os.environ['CLASSPATH'] = "".join([os.path.join(*[root_path,"StanfordModels",path])+";" for path in class_paths])[:-1]
os.environ['STANFORD_MODELS'] = "".join([os.path.join(*[root_path,"StanfordModels",path])+";" for path in model_paths])[:-1]
os.environ['JAVAHOME'] = "C:\\Program Files\\Java\\jre-10\\bin\\java.exe;"

parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")


amazon_reviews_path = os.path.join(*[root_path,"Data","AmazonReviews","test_file_small.json"])
dataf = open(amazon_reviews_path, 'r')

reviews = []
for line in dataf.readlines():
    line = line.rstrip()
    #Question, could you have the ' " ' char in the review?
    review = line[16:].split('"')[0]
    rating = float(line[-4])
    reviews.append([review,rating])
    
start_time = time.time()
sentence_compute_times = []

debug_log_path = os.path.join(*[root_path, "SAModel","debug_log.txt"])
results_log_path = os.path.join(*[root_path, "SAModel","results_log.txt"])
debug_log = open(debug_log_path,"w")
results_log = open(results_log_path,"w")

def split_sentences(st):
    st = st.replace("...","")
    st = st.replace("..","")
    sentences = re.split(r'[.?!]\s*', st)
    sentences = [s for s in sentences if len (s) != 0]
    if sentences[-1]:
        return sentences
    else:
        return sentences[:-1]

#results_data
results_data = {}

#1. Create SAModel object
model = SAModel()

composition_methods = [ "PARSETREE", "FLAT"]
neg_detection_methods = [[ "NEGTOOL", "WINDOW", "PARSETREE"], [ "NEGTOOL", "WINDOW" ]]
negres_list = ["ANTONYM_LOOKUP", "SYM_INVERT", "SYM_SHIFT", "ASYM_SHIFT" ]

results_data["Sentences"] = {}
results_data["Sentiment"] = {}

possible_models = []
for i in range(len(composition_methods)):
    for j in range(len(neg_detection_methods[i])):
        for neg_res in negres_list:
            model_type = neg_detection_methods[i][j] + " " + neg_res + " " + composition_methods[i]
            results_data["Sentiment"][model_type] = {}
            possible_models.append(model_type)
            #print(model_type)

review_id = 5
model.sentence_id = 13
results_data["Reviews"] = {}

num_reviews = len(reviews[6:])

for review in reviews[6:]:
    print("REVIEW_ID: {}\n\n".format(review_id))
    #split review into individual sentences
    sentences = split_sentences(review[0])
    
    truth = review[1]
    normalized_truth = (review[1] - 3)/2.0 #normalized truth

    results_data["Reviews"][review_id] = { "truth":truth, "start_s_id" : model.sentence_id, "end_s_id" : model.sentence_id + len(sentences) - 1}
    
    
    #run each sentence through SA pipeline and retrieve per-sentence sentiment
    for sentence in sentences:
        print("\nSentence: {}".format(sentence))
        #record sentence_id : sentence
        results_data["Sentences"][model.sentence_id] = sentence
        #print("S_ID[{}]:{}".format(model.sentence_id,sentence))
        
        #record time
        sentence_start_time = time.time()
        
        for i in range(len(composition_methods)):
            
            #setting sentiment composition method
            model.sent_comp_method = composition_methods[i]
            
            #3. Set the sentence
            model.setSentence(sentence)

            for j in range(len(neg_detection_methods[i])):
                
                #setting negation detection method
                model.neg_scope_method = neg_detection_methods[i][j]
                
                #4. detect neg scope
                model.detectNegScope()
                
                for neg_res in negres_list:
                    
                    #setting negation resolution method
                    model.neg_res_method = neg_res
                    
                    #resolve negation
                    model.neg_res()

                    #compose valences
                    sentence_sentiment = model.compose()

                    if(sentence_sentiment == -999):
                        debug_log.write("Sentence Sentiment for {} came out to be -999. \n\tModel:{}\n".format(sentence,model_type))
                    elif(sentence_sentiment is None):
                        debug_log.write("Sentence Sentiment for {} came out to be None. \n\tModel:{}\n".format(sentence,model_type))
                    
                    model_type = neg_detection_methods[i][j] + " " + neg_res + " " + composition_methods[i]
                    results_data["Sentiment"][model_type][model.sentence_id] = sentence_sentiment
                    print("{} + {} -> {}".format(model.sentence_id, model_type, sentence_sentiment))
        
        sentence_compute_time = time.time() - sentence_start_time
        sentence_compute_times.append(sentence_compute_time)
        print("S_ID[{}]: '{}' in {} seconds".format(model.sentence_id,sentence, sentence_compute_time))
        model.sentence_id += 1
        
    review_id += 1
    
#computing model accuracies
    

    
    
    
for model_type in possible_models:
    
    binary_tally = [0,0]
    fiveStar_tally = [0,0]
    
    for R_ID in range(num_reviews):
        
        start_s_id = results_data["Reviews"][R_ID]["start_s_id"]
        end_s_id = results_data["Reviews"][R_ID]["end_s_id"]
    
        results = [ results_data["Sentiment"][model_type][i] for i in range(start_s_id, end_s_id + 1) if (results_data["Sentiment"][model_type][i] != -999) and (results_data["Sentiment"][model_type][i] is not None) ]
        result = sum(results)/float(len(results))
        results_data["Reviews"][R_ID]["Result"] = result
        
        #BINARY CLASSIFICATION
        if(normalized_truth >= 0 and result < 0) or (normalized_truth < 0 and result >= 0): #consider 0 truth as positive
            binary_tally[1] += 1
        else:
            binary_tally[0] += 1

        
        
        #FIVE STAR CLASSIFICATION
        if(result < -0.6):
            result = 1
        elif(result < -0.2):
            result = 2
        elif(result < 0.2):
            result = 3
        elif(result < 0.6):
            result = 4
        else:
            result = 5

        if(truth != result):
            fiveStar_tally[1] += 1

        else:
            fiveStar_tally[0] += 1

    results_data["Sentiment"][model_type]["binary_classification_acc"] = float(binary_tally[0])/(binary_tally[0]+binary_tally[1])
    results_data["Sentiment"][model_type]["five_star_acc"] = float(fiveStar_tally[0])/(fiveStar_tally[0]+fiveStar_tally[1])
    print("Model Type: {} [BIN_ACC : {} / FS_ACC : {}".format(model_type, results_data["Sentiment"][model_type]["binary_classification_acc"], results_data["Sentiment"][model_type]["five_star_acc"]))


results_data["AverageSentenceComputeTime"] = sum(sentence_compute_times)/float(len(sentence_compute_times))
results_data_outfile_path = os.path.join(*[root_path, "SAModel","output","results_data_throw_away.txt"])
results_data_outfile = open(results_data_outfile_path, 'w')
json.dump(results_data,results_data_outfile,indent=4)

results_data_outfile.close()
results_log.close()
debug_log.close()
