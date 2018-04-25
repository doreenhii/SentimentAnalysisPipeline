#add ending punctuation to reviews in json
import json

punctuations = '''!"#$%&()*+,./\ :;<=>?@[\]^_`{|}~\n'''

infile = open("C:\\Users\\King\\Desktop\\SentimentAnalysisPipeline4.22\\reviews\\test_file_small.json", "r")
fixed_reviews_json = open("C:\\Users\\King\\Desktop\\SentimentAnalysisPipeline\\reviews\\test_file_small_fixed.json","w")
just_reviews_file = open("C:\\Users\\King\\Desktop\\SentimentAnalysisPipeline\\reviews\\negtool_negscope\\test_file_small_just_reviews.txt","w")
line = infile.readline()

line_json_format = json.loads(line)
line_json_format["reviewText"] = line_json_format["reviewText"].rstrip(punctuations)+"."
fixed_reviews_json.write(json.dumps(line_json_format))
just_reviews_file.write(line_json_format["reviewText"])

line = infile.readline()

while line:

	line_json_format = json.loads(line)
	
	line_json_format["reviewText"] = line_json_format["reviewText"].rstrip(punctuations)+"."
	fixed_reviews_json.write("\n"+json.dumps(line_json_format))
	just_reviews_file.write("\n"+line_json_format["reviewText"])

	line = infile.readline()

infile.close()
fixed_reviews_json.close()
