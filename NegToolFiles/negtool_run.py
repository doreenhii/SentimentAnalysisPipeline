import os

# from get_ns_neg import big_list
# from sentiment_analyzer import f

#TODO: standardize input_file names
class Negtool:
	def __init__(self, input_file=""):
		#to be passed from sentiment_analyzer main
		self.input_file = "Corrected_amazon.txt"
		self.processed_file= (os.getcwd() + "/output_uci_dataset/" + self.input_file.strip('.txt')+ "_output.txt")

#move this to negation_scope_detection.py 
negtool_object= Negtool(input_file)
