import os, time, json
from SAModel import *
from pathlib import Path

paths_file = open(Path(os.getcwd()) / "config.txt", "r")
paths_json = json.load(paths_file)
paths_file.close()
root_path = paths_json["ROOT_PATH"]

results_data = SAResults(paths_json)
sentence_results_file = open(str(Path(root_path) / "results" / "indepth_results.txt"), "w")
sentence_results_file.write("{}\t{}\t{}\n".format("review_id", "binary_results", "category_results"))

model = SAModel(paths_json)
composition_methods = [ "PARSETREE", "FLAT"]
neg_detection_methods = [[ "NEGTOOL", "WINDOW", "PARSETREE"], [ "NEGTOOL", "WINDOW" ]]
neg_res_methods = [ "ANTONYM_LOOKUP", "SYM_INVERT", "SYM_SHIFT", "ASYM_SHIFT" ]

#reviews
reviews_path = Path(root_path) / "reviews" / "test_file_small_fixed.json"
reviews_infile = open(reviews_path, "r")

review_json = reviews_infile.readline()

while review_json:
	review_json = json.loads(review_json)

	review = review_json["reviewText"]
	truth = review_json["overall"]

	start_time = time.time()
	

	for i in range(len(composition_methods)):
		
		model.sent_comp_method = composition_methods[i]
		model.setReview(review)
		
		for j in range(len(neg_detection_methods[i])):

			model.neg_scope_method = neg_detection_methods[i][j]
			model.detectNegScope()
			
			for k in range(len(neg_res_methods)):

				model.neg_res_method = neg_res_methods[k]

				model_name = " ".join([neg_detection_methods[i][j], neg_res_methods[k], composition_methods[i]])
				if(model_name == "PARSETREE ANTONYM_LOOKUP PARSETREE"):
					continue

				model.neg_res()
				review_sentiment = model.compose()
				
				results_data.categorize_prediction(model_name, truth, review_sentiment)
				results_data.binary_prediction(model_name, truth, review_sentiment)



				#print("\t\t{}".format(model_name))
		model.model = "NONE"
		model.setReview(review)
		review_sentiment = model.compose()
		results_data.categorize_prediction("NONE NONE FLAT", truth, review_sentiment)
		results_data.binary_prediction("NONE NONE FLAT", truth, review_sentiment)
		model.model = ""

	end_time = time.time()
	print("Review_{} in {} seconds: {}".format(model.review_id, (end_time-start_time), review))

	category_results, binary_results = results_data.get_in_depth_results()
	sentence_results_json = {"review_id":model.review_id, "binary_results":binary_results, "category_results":category_results}
	print("\t\tModels Results: Binary: {}\n\t\tCategory: {}".format(binary_results, category_results))
	sentence_results_file.write(json.dumps(sentence_results_json)+"\n")

	model.sentence_id += model.review_size
	model.review_id += 1

	review_json = reviews_infile.readline()

results_data.write_to_file()

sentence_results_file.close()





"""
	NEGTOOL ANTONYM_LOOKUP PARSETREE
	NEGTOOL SYM_INVERT PARSETREE
	NEGTOOL SYM_SHIFT PARSETREE
	NEGTOOL ASYM_SHIFT PARSETREE
	WINDOW ANTONYM_LOOKUP PARSETREE
	WINDOW SYM_INVERT PARSETREE
	WINDOW SYM_SHIFT PARSETREE
	WINDOW ASYM_SHIFT PARSETREE
	PARSETREE SYM_INVERT PARSETREE
	PARSETREE SYM_SHIFT PARSETREE
	PARSETREE ASYM_SHIFT PARSETREE
	NEGTOOL ANTONYM_LOOKUP FLAT
	NEGTOOL SYM_INVERT FLAT
	NEGTOOL SYM_SHIFT FLAT
	NEGTOOL ASYM_SHIFT FLAT
	WINDOW ANTONYM_LOOKUP FLAT
	WINDOW SYM_INVERT FLAT
	WINDOW SYM_SHIFT FLAT
	WINDOW ASYM_SHIFT FLAT
"""