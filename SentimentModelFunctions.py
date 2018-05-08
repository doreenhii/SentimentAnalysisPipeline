import os,sys
import csv, json
from nltk.corpus import wordnet as wn
import numpy as np 
import math

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
	valence_tree = []
	for subtree in tree:
		if(subtree.height() > 2):
			valence_tree.append(map_valence_tree(valence_dict, subtree))
		else:
			valence_tree.append(getValence(valence_dict,subtree[0]))
	return valence_tree

def getTreeIndices(subtree, parent_index = []):
	indices = []
	for i in range(len(subtree)):
		if(isinstance(subtree[i], str)):
			indices.append(parent_index)
		else:
			subtree_indices = getTreeIndices(subtree[i], parent_index+[i])
			indices += subtree_indices

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
                neg_scope.append(parent_index+[j+i+1] )
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
def negate(valence, neg_res):
	if(neg_res == "SYM_INVERT"):
		weight = 1.0
		return -weight*valence
	
	elif(neg_res == "AFFIRM_SHIFT"):
		return affirm_shift(valence)

	elif(neg_res == "ANTONYM_LOOKUP"):
		return valence
		
	elif(neg_res == "MEANINGSPEC_FREQ"):
		return meaningSpec_freq(valence)

	elif(neg_res == "MEANINGSPEC_FREQDP"):
		return meaningSpec_freqdp(valence)

def antonym_lookup_negate(valence_dict, word):
	try:
		antonym_word = antonym_lookup(word)
	except RuntimeError as re:
		antonym_word = None
		print("antonym_lookup({}) error: {}".format(word, re)) #antonym_lookup(us) error: maximum recursion depth exceeded
	
	if(antonym_word is None):
		antonym_valence = -999
	else:
		antonym_valence = getValence(valence_dict, antonym_word)

	return antonym_valence




def inferDist(valence):
	if valence < 0:
		cat = max(-1.0,math.floor(valence*10)/10)
	else:
		cat = min(1.0,math.ceil(valence*10)/10)

	# get mu and sigma of distribution

	distribution_dict = {"0.0": { "dispersion": [  0.20614098272, 0.22694636438 ], "frequency": [ 479.274819788, 4015.73513203 ], "mi": [ -1.09900650837, 0.69416713447 ] },"0.9": {"dispersion": [0.12472186402450802, 0.09911062196507409], "frequency": [22.9, 27.595108262154], "mi": [-1.26807775218268, 0.4524461318217832]}, "-1.0": {"dispersion": [0.04998100523029686, 0.02711892049235589], "frequency": [204.14285714285714, 289.54616141789336], "mi": [-1.086831152715014, 0.049598695417991334]}, "1.0": {"dispersion": [0.10890036984885758, 0.060863198898303106], "frequency": [147.33333333333334, 206.12267113434066], "mi": [-0.9231636802480843, 0.1768632676312558]}, "0.8": {"dispersion": [0.15360643607399108, 0.19340300482732256], "frequency": [159.71428571428572, 509.58784866812124], "mi": [-0.9303541096407597, 0.27695365484961487]}, "0.1": {"dispersion": [0.20454758778312587, 0.22518391704432866], "frequency": [465.9272160313278, 3388.1897711252336], "mi": [-1.0999982981715237, 0.7074418916449938]}, "0.3": {"dispersion": [0.19665532651445136, 0.22324480430568763], "frequency": [484.5969602388384, 2939.9888491791694], "mi": [-1.1039163410909294, 0.6955724444275289]}, "0.2": {"dispersion": [0.22166842325962727, 0.23573709193709425], "frequency": [400.03768904155334, 4690.652448879055], "mi": [-1.102633642615733, 0.716783444108477]}, "0.5": {"dispersion": [0.17670576582301936, 0.22131469767550652], "frequency": [506.4294871794872, 1410.555114913072], "mi": [-1.1170572688370082, 0.635648109631374]}, "0.4": {"dispersion": [0.1999171897839913, 0.23057491755201875], "frequency": [839.0542317173378, 5769.444666267686], "mi": [-1.109713555516979, 0.7341320015551404]}, "0.7": {"dispersion": [0.24880076994973527, 0.2552488974610296], "frequency": [395.00315043318454, 2196.7817479887976], "mi": [-1.1132897896385205, 0.7735700923121853]}, "0.6": {"dispersion": [0.1875678513731465, 0.20471073202040857], "frequency": [461.16073546856467, 3746.019064291584], "mi": [-1.0499511762794622, 0.6016039039300725]}, "-0.5": {"dispersion": [0.22136302583087178, 0.24114976353799328], "frequency": [224.8783783783784, 814.458175932513], "mi": [-1.0369568752815406, 0.6886765429777627]}, "-0.4": {"dispersion": [0.21872712888097973, 0.22923387560048356], "frequency": [291.389369592089, 1415.1060945294914], "mi": [-1.0762527681455216, 0.6902407499747721]}, "-0.7": {"dispersion": [0.27747355611549085, 0.2611843398381495], "frequency": [214.93642123661786, 1145.4502395849472], "mi": [-1.1348907917033266, 0.786204860333708]}, "-0.6": {"dispersion": [0.19992424122173133, 0.2232316858208454], "frequency": [491.8213640469738, 4191.259477862486], "mi": [-1.088972804213507, 0.6613249480674773]}, "-0.1": {"dispersion": [0.20773437766093059, 0.22870881171663437], "frequency": [492.6224235452018, 4643.280492939766], "mi": [-1.0980147185721394, 0.6808923772954889]}, "-0.3": {"dispersion": [0.214874647451952, 0.23298127313552053], "frequency": [401.7641728134879, 3240.3418595458165], "mi": [-1.101105479503926, 0.6919831794197995]}, "-0.2": {"dispersion": [0.20504053253078453, 0.2277381324722484], "frequency": [525.7266731016731, 6724.672228898145], "mi": [-1.1055872238838251, 0.6802925777926746]}, "-0.9": {"dispersion": [0.08326982478338668, 0.06417157968847934], "frequency": [97.0, 139.58549749566075], "mi": [-1.3345951723483673, 1.3075244337747716]}, "-0.8": {"dispersion": [0.10144763865432462, 0.13556403758020347], "frequency": [962.96, 4107.693683613714], "mi": [-0.9908078800097886, 0.2387426365729821]}}

	data = distribution_dict[str(cat)]

	random_freq_mu = data["frequency"][0]
	random_freq_sigma = data["frequency"][1]

	random_freq = np.random.normal(random_freq_mu, random_freq_sigma, 1)
	while (random_freq < 0):
		random_freq = np.random.normal(random_freq_mu, random_freq_sigma, 1)
	
	
	random_dp_mu = data["dispersion"][0]
	random_dp_sigma = data["dispersion"][1]

	random_dp = np.random.normal(random_dp_mu, random_dp_sigma, 1)
	while (random_dp < 0):
		random_dp = np.random.normal(random_dp_mu, random_dp_sigma, 1)
		
	# random_mi_mu = data["mi"][0]
	# random_mi_sigma = data["mi"][1]

	# random_mi = np.random.normal(random_mi_mu, random_mi_sigma, 1)
	
	return {"frequency" : random_freq[0], "dispersion": random_dp[0]}


def affirm_shift(Affirm):
	return -0.065916 -0.363218*Affirm

def meaningSpec_freq(Affirm):
	Freq = inferDist(Affirm)["frequency"]

	return -7.747130e-02 -3.850748e-01*Affirm + Freq*5.326080e-09


def meaningSpec_freqdp(Affirm):

	inferred = inferDist(Affirm)
	Freq = inferred["frequency"]
	DP = inferred["dispersion"]

	return -6.112665e-02 +Affirm*-3.851552e-01+Freq*7.751644e-09+DP*-2.260250e+00+Freq*DP*-1.976151e-06

#*--------------------------------------------------------------------------------------*#
#*                               Antonym Dictionary Method                              *#
#*--------------------------------------------------------------------------------------*#

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

def antonym_lookup(word):
	try:
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
	except RuntimeError as re:
		#print("antonym_lookup({}):{}".format(word, re)) #"us","me","50"
		return None
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

def flat_composition(valence_sequence):
	if(sys.version_info>(3, 0)):
		valence_sequence = list(filter(lambda a: a != -999, valence_sequence))
	else:
		valence_sequence = filter(lambda a: a != -999, valence_sequence)
	
	if(len(valence_sequence) == 0):
		avg_valence = -999
	else:
		avg_valence = sum(valence_sequence)/len(valence_sequence)
	return avg_valence
