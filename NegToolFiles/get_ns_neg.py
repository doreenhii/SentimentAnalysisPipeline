##Getting index of words in negation scope as a list

import csv, os
from negtool_run import *


# output= open("/Users/doreenyiijiehii/Desktop/NLP/My_Code/Running_Negtool/stuff_needed/output_uci_dataset/Corrected_amazon_output.txt",'r')
output = open(negtool_object.processed_file, 'r') 
reader = csv.reader(output, dialect='excel', delimiter='\t')



def list_index_handler():
    sentence_counter =0
    row_counter =0
    big_list=[]

    
    for row in reader:
        row_counter += 1

        #if it is still in the same sentence
        if (row != []):
            #look for "neg" label at col 7
            #other kinds of negation scope is ignored - they are such as prefix/words with NEGATive meaning
            if (row[6] == 'neg'):
                negated_list,negated_index= neg_detected(next(reader))
                #sentence_counter and negated_index start from 0
                big_list.append(sentence_counter)
                big_list.append(negated_index)

        #end of a sentence is marked by an additional \n
        elif (row ==[]):
            sentence_counter += 1

    #big list format
    #sentence_index, [words_indices]
            
    print("big_list=" )
    print(big_list)
    output.close()


def neg_detected(next_row):
    is_negated = True
    negated_list = []
    negated_index=[]


    max_col = len(next_row)
    current_col = 8
    while (is_negated and current_col < max_col and next_row!=[]):
        #look through all col for negated words
        if (next_row[current_col] == "_"):
              current_col += 1
        #record the index of word in negation scope
        elif (next_row[current_col] !="_" and next_row[current_col]!=""):
            negated_list.append(next_row[current_col])
            negated_index.append(int(next_row[0])-1)
            next_row =next(reader)
        else:
            is_negated = False
            break
    
    #return the list for the sentence
    return negated_list, negated_index


#Call function!
list_index_handler()
