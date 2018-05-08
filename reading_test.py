import os, time, json
from SAModel import *
from pathlib import Path

paths_file = open(Path(os.getcwd()) / "config.txt", "r")
paths_json = json.load(paths_file)
paths_file.close()
root_path = paths_json["ROOT_PATH"]


#reviews
reviews_path = Path(root_path) / "reviews" / "test_file_small_fixed.json"
reviews_infile = open(reviews_path, "r")
review_json = reviews_infile.readline()

review_id = 10


#negtool
endbytelog_path = str(Path(root_path) / "reviews" / "negtool_negscope" / "endbytelog.txt")

with open(endbytelog_path, "r") as endbytelog:
	endbytes_arr = [ (int(line.split("\t")[0]) , int(line.split("\t")[1].rstrip("\n")))  for line in endbytelog.readlines()]

print(endbytes_arr)

review_chunk_size = 5
current_chunk = int(review_id//review_chunk_size) 
current_endbyte = endbytes_arr[current_chunk][0]

negscope_path = str(Path(root_path) / "reviews" / "negtool_negscope" / "negtool_neg_scopes.txt")
with open(negscope_path, "r") as negscope_file:
	negscope_file.seek(current_endbyte)

	while review_json:
		review_json = json.loads(review_json)
		neg_scope = negscope_file.readline()
		
		review_id += 1
		if(review_id % 5 == 0):
			current_chunk +=1
			if(current_chunk < len(endbytes_arr)):
				current_endbyte = endbytes_arr[current_chunk][0]
				print("negscope.tell() :{}, logged endbyte: {}".format(negscope_file.tell(), current_endbyte))

			
		review_json = reviews_infile.readline()

reviews_infile.close()