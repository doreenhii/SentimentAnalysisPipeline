def extract_neg_scope(neg_infile_path):

	import csv,json

	neg_words = ["no","not"]

	contractions = ["n't", "'m", "'ll","'d","'s","'ve","'re"]

	#opening infile
	infile_path = neg_infile_path
	neg_infile = open(infile_path, "r")
	reader = csv.reader(neg_infile, dialect='excel', delimiter='\t')

	#json storing negation index (token index; does not include punctuation)
	sentences = {} #sentences = { s_id : { "tokenized_sent": [w1,w2,...], "negation_scope": [id1,id2,...] } } 

	new_sentence = True
	s_id = 0

	for row in reader:
		#print(row)
		#empty row == new sentence
		if(len(row) == 0):

			#after iterating through a sentence, enter in the negationscope into json
			for i in valid_scopes:
				sentences[s_id]["neg_scope"].append(neg_scopes[i])

			new_sentence = True
			s_id += 1
			continue
	  
		#open new key-value pair in dictionary
		elif(new_sentence):
			sentences[s_id] = { "tokenized_sent":[], "neg_scope":[] }

			num_neg_scopes = int((len(row) - 7)/3)#number of negation scopes determined by negtool
			valid_scopes = [] #lists which columns are negation word scopes and not negation prefix scopes
			neg_scopes = [[] for i in range(num_neg_scopes)]
			#print("# of neg scopes : {}".format(num_neg_scopes))

			word_index = 0
			new_sentence = False


		#include each word in tokenized_sent
		sentences[s_id]["tokenized_sent"] += [row[1]]

		#for each of the negation scope columns, record each word index
		for i in range(num_neg_scopes):
			#column index 8 will be the words being negated
			if(row[8+(i*3)] != '_'):
				neg_scopes[i] += [word_index]

		#if you catch a "neg" label, record w`hich column it's referring to
		if(row[6] == "neg" or row[1] in neg_words):
			#column index 7 will be the negation word
			valid_scopes.append([(col != '_') for col in row[7::3]].index(True))

		#count tokens
		word_index += 1

	neg_infile.close()

	#print(json.dumps(sentences,indent = 4))
	#save to outfile
	outfile_path = neg_infile_path.rstrip(".neg")+"_neg_scopes.txt"
	outfile = open(outfile_path, "w")
	json.dump(sentences, outfile, indent = 4)
	outfile.close()


if __name__ == '__main__':
	import os,json
	from pathlib import Path
	# import argparse
	# argparser = argparse.ArgumentParser()
	# #argparser.add_argument('-m', '--mode', help="program mode. either raw or parsed or retraining", type=str, choices=['raw','parsed'])
	# argparser.add_argument('-f', '--filename', help="input file", type=str, nargs='?')
	# argparser.add_argument('-p', '--path_to_corenlp', help="path to stanford corenlp", type=str, default = "StanfordModels/stanford-corenlp-full-2018-02-27/")
	# argparser.add_argument('-d', '--negtool_directory', help="path to negtool's models", type=str, default = "NegToolFiles/")
	# args = argparser.parse_args()

	# config_data = open(os.path.join(os.getcwd(),"config.txt"), "r")
	# config_data_json = json.load(config_data)
	# config_data.close()
	paths_file = open(Path(os.getcwd()) / "config.txt", "r")
	paths_json = json.load(paths_file)
	paths_file.close()
	root_path = paths_json["ROOT_PATH"]
	negtool_directory = paths_json["NEGTOOL_DIRECTORY"]
	path_to_corenlp = paths_json["PATH_TO_CORENLP"]
	filename = "C:\\Users\\King\\Desktop\\SentimentAnalysisPipeline4.22\\reviews\\negtool_negscope\\test_file_small_just_reviews.txt"

	import sys
	sys.path.append(negtool_directory)
	import negtool
	negtool.run(mode = "raw", filename = filename, path_to_corenlp = path_to_corenlp, negtool_directory = negtool_directory)
	neg_file_path = filename.rstrip(".txt") + ".neg"
	extract_neg_scope(neg_file_path)



