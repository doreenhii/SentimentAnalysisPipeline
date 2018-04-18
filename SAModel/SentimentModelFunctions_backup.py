import os,sys
import csv, json
from nltk.corpus import wordnet as wn 
import numpy as np
def sign(value):
    if(value < 0):
        return -1.0
    else:
        return 1.0

def getValence(valence_dict, word):
	#removes punctuation from word
	
	punctuations = '''!"#$%&()*+,./:;<=>?@[\]^_`{|}~\n'''
	
	if (sys.version_info > (3, 0)):
		word = word.translate(punctuations)
	else:
		print("ERROR, need py3")
	if(word in valence_dict):
		return valence_dict[word]
	else:
		return -999

def map_valence_tree(valence_dict, tree):
	# valence_tree = np.array([])
	# punctuations = '''!"#$%&()*+,./:;<=>?@[\]^_`{|}~\n'''
	# for subtree in tree:
	# 	if(subtree.height() > 2):
	# 		valence_tree = np.append(valence_tree, map_valence_tree(valence_dict, subtree))
	# 	elif(subtree[0] not in punctuations):
	# 		valence_tree = np.append(valence_tree,getValence(valence_dict,subtree[0]))
	# return valence_tree
	valence_tree = []
	punctuations = '''!"#$%&()*+,./:;<=>?@[\]^_`{|}~\n'''
	for subtree in tree:
		if(subtree.height() > 2):
			valence_tree.append(map_valence_tree(valence_dict, subtree))
		elif(subtree[0] not in punctuations):
			valence_tree.append(getValence(valence_dict,subtree[0]))
	return valence_tree
def get_word_indices(subtree, parent_index = []):
	indices = []
	for i in range(len(subtree)):
		#if python 3: str, if python 2: unicode
		
		if(sys.version_info > (3, 0) and isinstance(subtree[i], str)):
			indices.append(parent_index)
		else:#(subtree[i].height() > 2):
			indices += get_word_indices(subtree[i], parent_index+[i])
	return indices

def SequenceToTreeIndices(completeTreeIndices, sequence_indices):
    return [completeTreeIndices[i] for i in sequence_indices]

def TreeToSequenceIndices(completeTreeIndices, tree_indices):
	return [completeTreeIndices.index(i) for i in tree_indices]
	
def isNegator(word):
    negators = ["cannot", "n't", "no", "never", "not", "nothing", "nobody"]
    if(word in negators):
        return True
    else:
        return False

#*------------------------------------------------------------------------------------------*#
#*                             Negation Scope Detection Methods                             *#
#*------------------------------------------------------------------------------------------*#

def detect_neg_scope_tree(subtree, parent_index = []):
    neg_scope = []
    for i in range(len(subtree)):
        isNegatorCheck1 = (subtree[i].height() < 3 and isNegator(subtree[i][0]))
        isNegatorCheck2 = (subtree[i].label() == "ADVP" and isNegator(subtree[i][0][0]))
        if(isNegatorCheck1 or isNegatorCheck2):
            for j in range(len(subtree)-(i+1)):
                neg_scope.append(parent_index+[j+i+1]+[0])
        elif(subtree[i].height() > 2):
            neg_scope += detect_neg_scope_tree(subtree[i], parent_index+[i])

    return neg_scope

def detect_neg_scope_window(sentence_sequence, window_size = 0):
    neg_scope = []
    num_scopes = 0
    last_scope_count = 0

    for i in range(len(sentence_sequence)):

        if(isNegator(sentence_sequence[i])):
            num_scopes += 1
            last_scope_count = 0

        elif(num_scopes > 0):
            for j in range(num_scopes):
                neg_scope.append(i)

            if(window_size > 0): #window_size = 0 signifies end-of-sentence scope
                last_scope_count += 1
                if(last_scope_count >= window_size):
                    num_scopes = 0
                    last_scope_count = 0

    return neg_scope

def resolve_double_negative(neg_scope):
    new_thing = []
    for coord in neg_scope:
        if(neg_scope.count(coord)%2==1 and new_thing.count(coord) == 0):
            new_thing.append(coord)
    return new_thing

def negtool():
    pass


#*------------------------------------------------------------------------------------------*#
#*                                 Negation Resolution Methods                              *#
#*------------------------------------------------------------------------------------------*#
"""
    1. sym_inversion
    2. asym_inversion
    3. sym_shift
    4. asym_shift
    5. lookup_shift
    6. multiple_reg
"""
def negate(valence, neg_res = "ASYM_SHIFT"):
	if(neg_res == "SYM_INVERT"):
		weight = 1.0
		return -weight*valence
	elif(neg_res == "ASYM_INVERT"):
		positive_weight = 1.0
		negative_weight = 0.75
		if(valence < 0):
			return -negative_weight*valence
		else:
			return -positive_weight*valence
	elif(neg_res == "SYM_SHIFT"):
		weight = 0.75
		return valence - sign(valence)*weight
	elif(neg_res == "ASYM_SHIFT"):
		positive_weight = 0.90
		negative_weight = 0.60
		if(valence < 0):
			return valence - sign(valence)*negative_weight
		else:
			return valence - sign(valence)*positive_weight
	elif(neg_res == "ANTONYM_LOOKUP"):
		return valence
	elif(neg_res == "MULTIPLE_REG"):
		pass
		
def mother(keyword_lemma):
	track_antonym= []

	if len(keyword_lemma) >25:
		return track_antonym

	# check synset (synonyms)
	keyword_synsets = check_has_synset(keyword_lemma)
	word_list_lemma = change_to_lemma(keyword_synsets)

	track_antonym = check_has_antonym(word_list_lemma)
	if track_antonym !=[]:
		return track_antonym

	#if there is no antonym for all keyword_synset in keyword_list,
	#get attribute of the keywords
	attribute_synset = check_has_attribute(keyword_lemma)

	#if there are keyword->attribute
	if attribute_synset !=[]:
		attribute_lemma = change_to_lemma(attribute_synset)
		track_antonym = check_has_antonym(attribute_lemma)

		if track_antonym != []:
			#if keyword->attribute has antonym, return that
			return track_antonym

	#if attribute_list==[] or keyword->attribute->antonym ==[]
	#check pertainym of keyword
	pertainym_list = check_has_pertainym(keyword_lemma)

	#if there are keyword->pertainym
	if pertainym_list!=[]:
		track_antonym = check_has_antonym(pertainym_list)

		if track_antonym !=[]:
			return track_antonym

		#if there is keyword->pertainym but no keyword->pertainym->antonym
		track_antonym = mother(pertainym_list)

		if track_antonym !=[]:
			return track_antonym

	#if pertainym_list ==[], check derivationally_related_forms
	derivation_list = check_has_derivation(keyword_lemma)

	if derivation_list !=[]:
		track_antonym = check_has_antonym(derivation_list)

		if track_antonym !=[]:
			return track_antonym

		#print 'derivation_list', derivation_list
		track_antonym = mother(derivation_list)
		if track_antonym !=[]:
			return track_antonym

	#if keyword->derivation_list or keyword->derivation->antonym ==[]
	#check similar to
	similar_list = check_has_similar(keyword_lemma)

	if similar_list !=[]:
		track_antonym = check_has_antonym(similar_list)

		if track_antonym !=[]:
			return track_antonym

		track_antonym = mother(similar_list)
		if track_antonym !=[]:
			return track_antonym

	#If all means exhausted and still no relation
	return track_antonym

def change_to_synset(lemma_list):
	synset_list=[]

	for lemma in lemma_list:
		synset= lemma.synset()
		synset_list.append(synset)

	# print 'synset_list', synset_list
	return synset_list

def change_to_lemma(keyword_synsets):
	#switch to lemma for antonyms
	word_list_lemma=[]
	for synset in keyword_synsets:
		temp_list_lemma= synset.lemmas()
		for temp in temp_list_lemma:
			word_list_lemma.append(temp)

	return word_list_lemma

def check_has_synset(keyword_lemma):
	# print 'check_has_synset'
	synset_list = []

	for lemma in keyword_lemma:
		temp_synset = wn.synsets(lemma.name())

		if temp_synset != []:
			for synset in temp_synset:
				synset_list.append(synset)
	return synset_list

def check_has_attribute(keyword_lemma):
	# print 'check_has_attribute'
	attribute_list=[]

	keyword_synset_list = change_to_synset(keyword_lemma)

	for keyword_synset in keyword_synset_list:
		temp_attribute_list= keyword_synset.attributes()
		if temp_attribute_list !=[]:
			for temp in temp_attribute_list:
				attribute_list.append(temp)

	return attribute_list

def check_has_antonym(word_list):
	# print 'check_has_antonym'
	antonym=[]

	for lemma in word_list:
		# if lemma.antonyms():
		# 	antonym.append(lemma.antonyms()[0].name())

		antonym_list =lemma.antonyms()

		if antonym_list != []:
			for antonym_word in antonym_list:
				antonym.append(antonym_word.name())

	return antonym

def check_has_pertainym(word_list_lemma):
	# print 'check_has_pertainym'
	pertainym_list=[]

	for lemma in word_list_lemma:
		temp_pertainym_list = lemma.pertainyms()

		if temp_pertainym_list!= []:
			for temp in temp_pertainym_list:
				pertainym_list.append(temp)

	# print 'pertainym_list',pertainym_list
	return pertainym_list

def check_has_derivation(word_list_lemma):
	# print 'check_has_derivation'

	derivation_list=[]

	for lemma in word_list_lemma:
		temp_derivation_list = lemma.derivationally_related_forms()

		if temp_derivation_list!= []:
			for temp in temp_derivation_list:
				derivation_list.append(temp)

	# print 'derivation_list',derivation_list
	return derivation_list
			
def check_has_similar(word_list):
	# print 'check_has_similar'
	similar_list= []

	for synset in word_list:
		temp_similar_list=synset.similar_tos()
		if temp_similar_list != []:
			for temp in temp_similar_list:
				similar_list.append(temp)

	# print 'similar_list', similar_list
	return similar_list

def antonym_lookup_from_file():
	table_file=open('/Users/doreenhiiyiijiehii/Desktop/Meaning_Spec/NRC-Sentiment-Emotion-Lexicons/AutomaticallyGeneratedLexicons/NRC-Emoticon-AffLexNegLex-v1.0/cleanup-corrected/kirit_data_all_neg.csv','rb')
	table_file.readline()

	for row in csv.reader(table_file):
		word = row[0]

		print (word)
		antonym= []
		keyword_synsets = wn.synsets(word)

		# print ('synsets', keyword_synsets)

		word_list_lemma = change_to_lemma(keyword_synsets)

		antonym = check_has_antonym(word_list_lemma)

		#if there is no antonym for all keyword_synset in keyword_list,
		if antonym ==[]:
			antonym = mother(word_list_lemma)

		#antonym output as a list
		print(antonym)

def antonym_lookup(word):

	antonym= []
	keyword_synsets = wn.synsets(word)

	# print ('synsets', keyword_synsets)

	word_list_lemma = change_to_lemma(keyword_synsets)

	# print 'word_list_lemma', word_list_lemma
	antonym = check_has_antonym(word_list_lemma)

	#if there is no antonym for all keyword_synset in keyword_list,
	if antonym ==[]:
		antonym = mother(word_list_lemma)

	#antonym output as a list
	for anto in antonym:
		if '_' not in anto:
			return anto
	# return antonym[0]

#*------------------------------------------------------------------------------------------*#
#*                               Sentiment Composition Methods                              *#
#*------------------------------------------------------------------------------------------*#

def tree_composition(tree, parent_index, neg_scope, neg_res):
    valence = []
    for i in range(len(tree)):
        subtree = tree[i]

        current_index = parent_index+[i]

        if(isinstance(subtree, float) or isinstance(subtree, int)):
            if(subtree == -999):
                continue

            if(current_index in neg_scope):
                subtree = negate(subtree, neg_res)
            valence.append(subtree)
        else:
            valence_from_tree = tree_composition(subtree, current_index, neg_scope, neg_res)
            if(valence_from_tree == -999):
                continue
            if(current_index in neg_scope):
                valence_from_tree = negate(valence_from_tree, neg_res)
            valence.append(valence_from_tree)
    if(len(valence) == 0):
        return -999
    else:
        return sum(valence)/float(len(valence))

def flat_composition(valence_sequence, neg_scope, neg_res):
	if(neg_res != "ANTONYM_LOOKUP"):
		for index in neg_scope:
			if(valence_sequence[index] != -999):
				valence_sequence[index] = negate(valence_sequence[index], neg_res)

	if(sys.version_info>(3, 0)):
		valence_sequence = list(filter(lambda a: a != -999, valence_sequence))
	else:
		valence_sequence = filter(lambda a: a != -999, valence_sequence)
	
	if(len(valence_sequence) == 0):
		avg_valence = None
	else:
		avg_valence = sum(valence_sequence)/len(valence_sequence)
	return avg_valence