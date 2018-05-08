import os, time, json
from SAModel import *
from pathlib import Path

model_types = {
"NONE NONE FLAT" : 0,

"NEGTOOL ANTONYM_LOOKUP PARSETREE" : 1,
"NEGTOOL SYM_INVERT PARSETREE" : 2,
"NEGTOOL AFFIRM_SHIFT PARSETREE" : 3,
"NEGTOOL MEANINGSPEC_FREQ PARSETREE" : 4,
"NEGTOOL MEANINGSPEC_FREQDP PARSETREE" : 5,

"WINDOW ANTONYM_LOOKUP PARSETREE" : 6,
"WINDOW SYM_INVERT PARSETREE" : 7,
"WINDOW AFFIRM_SHIFT PARSETREE" : 8,
"WINDOW MEANINGSPEC_FREQ PARSETREE" : 9,
"WINDOW MEANINGSPEC_FREQDP PARSETREE" : 10,

"PARSETREE SYM_INVERT PARSETREE" : 11,
"PARSETREE AFFIRM_SHIFT PARSETREE" : 12,
"PARSETREE MEANINGSPEC_FREQ PARSETREE" : 13,
"PARSETREE MEANINGSPEC_FREQDP PARSETREE" : 14,

"NEGTOOL ANTONYM_LOOKUP FLAT" : 15,
"NEGTOOL SYM_INVERT FLAT" : 16,
"NEGTOOL AFFIRM_SHIFT FLAT" : 17,
"NEGTOOL MEANINGSPEC_FREQ FLAT" : 18,
"NEGTOOL MEANINGSPEC_FREQDP FLAT" : 19,

"WINDOW ANTONYM_LOOKUP FLAT" : 20,
"WINDOW SYM_INVERT FLAT" : 21,
"WINDOW AFFIRM_SHIFT FLAT": 22,
"WINDOW MEANINGSPEC_FREQ FLAT": 23,
"WINDOW MEANINGSPEC_FREQDP FLAT": 24

}

paths_file = open(Path(os.getcwd()) / "config.txt", "r")
paths_json = json.load(paths_file)
paths_file.close()
root_path = paths_json["ROOT_PATH"]


review_results = open(str(Path(root_path) / "results" / "review_results.txt"), "a+")
error_log = open(str(Path(root_path) / "results" / "error_log.txt"), "a+")



model = SAModel(paths_json)
composition_methods = [ "FLAT", "PARSETREE"]
neg_detection_methods = [[ "WINDOW", "NEGTOOL"], [ "WINDOW", "NEGTOOL", "PARSETREE"]
neg_res_methods = [ "ANTONYM_LOOKUP", "AFFIRM_SHIFT", "SYM_INVERT", "MEANINGSPEC_FREQ", "MEANINGSPEC_FREQDP"] #"ANTONYM_LOOKUP", #missing antonym lookup

model.use_negtool = True #if you're not running negtool, set this to False

# reviews
reviews_path = "/Users/alanyuen/Desktop/UCI_NLP/SentimentAnalysisPipeline/reviews/test_file_1000_nt_fixed.json"
reviews_infile = open(reviews_path, "r")

start_offset = 0
model.review_id = start_offset
model.sentence_id = 0

while(model.negtool_neg_scopes_file_current_line < model.sentence_id):
    model.negtool_neg_scopes_file.readline()
    model.negtool_neg_scopes_file_current_line += 1

pipeline_start_time = time.time()

for i in range(start_offset+1):
	review_json = reviews_infile.readline()

while model.review_id != 1000:
	review_json = json.loads(review_json)
	review = review_json["reviewText"]
	truth = review_json["overall"]
	
	start_time = time.time()
	
	model_predictions = [0 for i in range(25)]

	try:
		model.preparePipeline(review, "PARSETREE")
	except Exception as e:

		error_log.write("{}\n".format(model.review_id))
		print("ERROR {}, could not parse reivew_id {}\n".format(e, model.review_id))
		review_results.write(json.dumps( { "rid" : model.review_id, "sid" : model.sentence_id, "t" : int(truth), "p" : model_predictions } ) + "\n")

		model.getReviewSize(review)
		model.sentence_id += model.review_size
		model.review_id += 1
		model.getNegtoolNegscope()
		review_json = reviews_infile.readline()
		continue


	model.getNegtoolNegscope()
	#model.tokenizeReview(review) #for slow

	for i in range(len(composition_methods)):
		

		model.sent_comp_method = composition_methods[i]	

		# try:
		# 	model.setReview()
		# except:
		# 	error_log.write("{}\n".format(model.review_id))
		# 	continue
		
		for j in range(len(neg_detection_methods[i])):

			model.neg_scope_method = neg_detection_methods[i][j]
			model.detectNegScope()
			
			for k in range(len(neg_res_methods)):

				model.neg_res_method = neg_res_methods[k]

				model_name = " ".join([neg_detection_methods[i][j], neg_res_methods[k], composition_methods[i]])
				if(model_name == "PARSETREE ANTONYM_LOOKUP PARSETREE"):
					continue

				model.neg_res()

				review_sentiment = round(model.compose(), 3)
				model_predictions[model_types[model_name]] = review_sentiment

				model.reset_after_neg_res()

		model.model = ""

	model.model = "NONE"

	review_sentiment = round(model.compose(), 3)
	model_predictions[model_types["NONE NONE FLAT"]] = review_sentiment

	model.model = ""

	end_time = time.time()

	print("Review_{} in {} seconds: {}".format(model.review_id, (end_time-start_time), review))

	review_results.write(json.dumps( { "rid" : model.review_id, "sid" : model.sentence_id, "t" : int(truth), "p" : model_predictions } ) + "\n")
	

	model.sentence_id += model.review_size
	model.review_id += 1

	
	review_json = reviews_infile.readline()

pipeline_end_time = time.time()

print("Total time: {}".format(pipeline_end_time - pipeline_start_time))
review_results.close()
error_log.close()
model.close_files()