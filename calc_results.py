import utils

def calcPrecision(results, models):
	"""
	Precision is the fraction of events where we correctly declared i out of all instances where the algorithm declared i.

		p = true_positive / (true_positive + false_positive + positified_neutral)
	"""
	precision_results = {}
	
	for model in models:
		true_positive = results[model]["binary_results_dict"]["true_positive"]
		false_positive = results[model]["binary_results_dict"]["false_positive"]
		positified_neutral = results[model]["binary_results_dict"]["positified_neutral"]
		precision_results[model] = true_positive / (true_positive + false_positive + positified_neutral)

	return precision_results

def calcRecall(results, models):
	"""
	recall is the fraction of events where we correctly declared i out of all of the cases where the true of state of the world is i

		r = true_positive / (true_positive + false_negative + neutralized_positive)
	"""	
	recall_results = {}
	
	for model in models:
		true_positive = results[model]["binary_results_dict"]["true_positive"]
		false_negative = results[model]["binary_results_dict"]["false_negative"]
		neutralized_positive = results[model]["binary_results_dict"]["neutralized_positive"]
		recall_results[model] = true_positive / (true_positive + false_negative + neutralized_positive)

	return recall_results

def calcF1Score(results,models):
	f1_results = {}
	p_results = calcPrecision(results, models)
	r_results = calcRecall(results, models)

	for model in models:
		f1_results[model] = 2 * (p_results[model] * r_results[model]) / (p_results[model] + r_results[model])
	
	return f1_results

def calcAccuracy(results, models):
	accuracy_results = {}

	for model in models:
		correct_classifications = results[model]["correct_binary_results"]
		incorrect_classifications = results[model]["incorrect_binary_results"]
		accuracy_results[model] = correct_classifications / (correct_classifications + incorrect_classifications)
		#print("model: {} = {}/{}".format(model,correct_classifications,incorrect_classifications))
		
	return accuracy_results


from pathlib import Path
import json

"""
####################################################################################
Accuracy, precision, f1scores of each model
####################################################################################
"""
paths_json = utils.get_paths_json()
results_path = Path(paths_json["ROOT_PATH"]) / "results" / "sa_results-2018-04-25.json"
results = json.load(open(results_path, "r"))

models_path =  Path(paths_json["ROOT_PATH"]) / "models_to_run.txt"
models = [model.rstrip("\n") for model in open(models_path).readlines()]

f1_scores = calcF1Score(results, models)
accuracy_scores = calcAccuracy(results, models)


print("\nModel: {: >15} {: >15} {: >15} \t\t {: >4}".format("Negscope","Neg Resolution","Composition", "Score"))
print("-"*100)
results = sorted(f1_scores.items(), key=lambda x: x[1]) #F1Scores
for model, score in results:
	print("Model: {: >15} {: >15} {: >15} \t\t F1:{: >4}".format(model.split(" ")[0],model.split(" ")[1],model.split(" ")[2], round(score,4)))

print("\n\n")

print("\nModel: {: >15} {: >15} {: >15} \t\t {: >4}".format("Negscope","Neg Resolution","Composition", "Score"))
print("-"*100)
results = sorted(accuracy_scores.items(), key=lambda x: x[1])
for model, score in results:
	print("Model: {: >15} {: >15} {: >15} \t\t Bin_Acc:{:>4}".format(model.split(" ")[0],model.split(" ")[1],model.split(" ")[2], round(score,4)))

#print("F1 Scores:",json.dumps(f1_scores, indent = 4))

"""
####################################################################################
Finding hard sentences (which sentences were largely missed by all the models)
####################################################################################
"""
# indepth_results_path = Path(paths_json["ROOT_PATH"]) / "results" / "indepth_results.txt"
# indepth_results = open(indepth_results_path,"r").readlines()[1:]

# indepth_results_json = {}
# for row in indepth_results:
# 	row = row.rstrip("\n")
# 	row = row.split("\t")
# 	review_id = row[0]
# 	row = json.loads("{\"binary_results\":"+row[1]+",\"category_results\":"+row[2]+"}")
# 	indepth_results_json[review_id] = row 

# bin_results_json = {}
# for review_id in indepth_results_json.keys():
# 	bin_results = indepth_results_json[review_id]["binary_results"]
# 	average_accuracy = sum(bin_results)/float(len(bin_results))
# 	bin_results_json[review_id] = average_accuracy

# sorted_review_ids = sorted(bin_results_json.items(), key=lambda x: x[1])
# #reviews
# reviews_path = Path(paths_json["ROOT_PATH"]) / "reviews" / "test_file_small_fixed.json"
# reviews_infile = open(reviews_path, "r")

# review_json = reviews_infile.readlines()


# for (r_id, avg_acc) in sorted_review_ids:
# 	print("Avg Acc: {} - \"{}\"".format(avg_acc, json.loads(review_json[int(r_id)])["reviewText"]))
# 	print("\n")

	

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