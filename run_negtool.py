def extract_neg_scope(neg_infile_path):

	import csv,json, re

	pattern = re.compile(" doesnt | dont | cant | wont | didnt | havent | shant | cannot | couldnt | neednt | wasnt | isnt | arent | oughtnt | wouldnt | hasnt | mightnt | mustnt | shouldnt | Doesnt |Dont |Cant |Wont |Didnt |Havent |Shant |Cannot |Couldnt |Neednt | Wasnt |Isnt |Arent |Oughtnt |Wouldnt |Hasnt |Mightnt |Mustnt | Shouldnt |\w+n't | no | not | none | no one | nobody | nothing | neither | nowhere | never | nor |No |Not | None |No one |Nobody |Nothing |Neither |Nowhere |Never |Nor ",re.IGNORECASE)

	infile_path = neg_infile_path
	neg_infile = open(infile_path, "r")#opening infile


	dir_path = os.path.dirname(infile_path)
	outfile_path = Path(dir_path) / "aclimdb_negtool_neg_scopes.txt"
	
	reader = csv.reader(neg_infile, dialect='excel', delimiter='\t')

	#json storing negation index (token index; does not include punctuation)
	sentence = {}

	new_sentence = True

	endbytelog_path = Path(dir_path) / 'endbytelog.txt'
	if not endbytelog_path.is_file():
		with open(endbytelog_path, 'w') as endbytelog:
			pass

	with open(endbytelog_path,'r+') as endbytelog:
		log = endbytelog.readlines()
		if(len(log) == 0):
			s_id = 0
		else:	
			s_id = int(log[-1].split("\t")[1].rstrip("\n"))


	for row in reader:
		#print(row)
		#empty row == new sentence
		if(len(row) == 0):

			#after iterating through a sentence, enter in the negationscope into json
			for i in valid_scopes:
				try:
					sentence["negscope"].append(neg_scopes[i])
				except IndexError:
					pass


			#opening outfile
			with open(outfile_path, 'a') as outfile:
				#write negscope to big negscope file
				json.dump(sentence, outfile)
				outfile.write(os.linesep)


			new_sentence = True
			s_id += 1
			continue
	  
		#open new key-value pair in dictionary
		elif(new_sentence):
			
			if(len(row) == 0):
				break

			sentence = { "sentence_id" : s_id, "negscope" : [] }

			num_neg_scopes = int((len(row) - 7)/3)#number of negation scopes determined by negtool
			valid_scopes = [] #lists which columns are negation word scopes and not negation prefix scopes
			neg_scopes = [[] for i in range(num_neg_scopes)]
			#print("# of neg scopes : {}".format(num_neg_scopes))

			word_index = 0
			new_sentence = False


		#for each of the negation scope columns, record each word index
		for i in range(num_neg_scopes):
			#column index 8 will be the words being negated
			if(row[8+(i*3)] != '_'):
				neg_scopes[i] += [word_index]

		#if you catch a "neg" label, record w`hich column it's referring to
		if(row[6] == "neg" or re.search(pattern, row[1]) != None):
			#column index 7 will be the negation word
			valid_scopes.append([(col != '_') for col in row[7::3]].index(True))

		#count tokens
		word_index += 1

	neg_infile.close()
	
	with open(Path(dir_path) / 'endbytelog.txt','a+') as endbytelog:
		with open(outfile_path, 'a+') as outfile:
			end_byte = outfile.tell()
			endbytelog.write(str(end_byte) + "\t" + str(s_id) + "\n")

	#print(json.dumps(sentences,indent = 4))
	
	


if __name__ == '__main__':
	import os,json, sys
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

	sys.path.append(negtool_directory)
	import negtool

	reviews_filename = str(Path(root_path) / "reviews" / "negtool_negscope" / "aclimbd_compiled_reviews_fixed_just_reviews.txt")

	file_for_negtool_path = str(Path(root_path) / "reviews" / "negtool_negscope" / "splitaa.txt")
	neg_file_path = file_for_negtool_path.split(".txt")[0] + ".neg"

	review_cap = 500

	with open (reviews_filename, 'r') as infile:

		count = 1

		current_top = count + review_cap - 1

		outfile = open(file_for_negtool_path, 'w') #splitaa

		for line in infile:

			
			if (count == current_top):

				outfile.write(line.split("\n")[0])
				outfile.close()
				
				negtool.run(mode = "raw", filename = file_for_negtool_path, path_to_corenlp = path_to_corenlp, negtool_directory = negtool_directory)
				
				extract_neg_scope(neg_file_path)

				current_top = count + review_cap

				outfile = open(file_for_negtool_path, 'w')

				print ('round done')

			else:

				outfile.write(line)

			count += 1

		outfile.close()


