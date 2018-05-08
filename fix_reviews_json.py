#add ending punctuation to reviews in json
import json, os
from pathlib import Path

punctuations = '''!"#$%&()*+,./\ :;<=>?@[\]^_`{|}~\n'''


paths_file = open(Path(os.getcwd()) / "config.txt", "r")
paths_json = json.load(paths_file)
paths_file.close()
root_path = Path(paths_json["ROOT_PATH"])

#amazon
# infile = open(root_path / "reviews" / "test_file_small.json", "r")
# fixed_reviews_json = open(root_path / "reviews" / "test_file_small_fixed.json","w")
# just_reviews_file = open(root_path / "reviews" /  "negtool_negscope" / "test_file_small_just_reviews.txt","w")


#aclImdb
infile = open(root_path / "reviews" / "aclImdb" / "compiled_reviews.txt", "r")
fixed_reviews_json = open(root_path / "reviews" / "aclImdb" / "aclimbd_compiled_reviews_fixed.txt","w")
just_reviews_file = open(root_path / "reviews" /  "negtool_negscope" / "aclimbd_compiled_reviews_fixed_just_reviews.txt","w")


line = infile.readline()



line_json_format = json.loads(line)
line_json_format["reviewText"] = line_json_format["reviewText"].rstrip(punctuations)+" ."
fixed_reviews_json.write(json.dumps(line_json_format))
just_reviews_file.write(line_json_format["reviewText"])

line = infile.readline()

while line:

	line_json_format = json.loads(line)
	
	line_json_format["reviewText"] = line_json_format["reviewText"].rstrip(punctuations)+" ."
	fixed_reviews_json.write("\n"+json.dumps(line_json_format))
	just_reviews_file.write("\n"+line_json_format["reviewText"])

	line = infile.readline()

infile.close()
fixed_reviews_json.close()
