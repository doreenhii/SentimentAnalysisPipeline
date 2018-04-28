import itertools
#from utils import get_affix_cue, count_multiword_cues, mwc_start, convert_to_IO, in_scope_token
#from utils import *
import numpy as np

def convert_cues_to_fileformat(sentences, labels, affix_cue_lexicon, filename, mode):
    """
    Write the predicted cues to file, using the CD format for cues.
    """
    infile = open(filename, "r")
    output_filename = filename.split(".")[0] + "_cues.neg"
    outfile = open(output_filename, "w")
    sent_counter = 0
    line_counter = 0
    #corenlp generates one less column in original file than conll-x format
    upper_limit = 7 if mode == "raw" else 8
    n_cues = sum(i > 0 for i in labels[sent_counter])
    n_mwc, has_mwc = count_multiword_cues(sentences[sent_counter], labels[sent_counter])
    if has_mwc:
        n_cues += n_mwc - 1
    written_cues = n_cues*[False]
    for line in infile:
        tokens = line.split()
        if len(tokens) == 0:
            sent_counter += 1
            line_counter = 0
            if sent_counter < len(labels):
                n_cues = sum(i > 0 for i in labels[sent_counter])
                n_mwc, has_mwc = count_multiword_cues(sentences[sent_counter], labels[sent_counter])
                if has_mwc:
                    n_cues += n_mwc - 1
                written_cues = n_cues*[False]
            outfile.write("\n")
        else:
            written_cue_on_line = False
            #write the columns in the original parsed file to the outfile
            for i in range(upper_limit):
                outfile.write("%s\t" %tokens[i])
            if n_cues == 0:
                outfile.write("***\n")
            else:
                for cue_i in range(n_cues):
                    if labels[sent_counter][line_counter] < 0:
                        outfile.write("_\t_\t_\t")
                    else: #cue-line
                        if written_cues[cue_i] or written_cue_on_line:
                            #if cue on curr line is already processed, skip to next cue in sentence
                            outfile.write("_\t_\t_\t")
                        else:
                            affix = get_affix_cue(tokens[1].lower(), affix_cue_lexicon)
                            if affix != None:
                                outfile.write("%s\t_\t_\t" %affix)
                                written_cues[cue_i] = True
                            else:
                                outfile.write("%s\t_\t_\t" %tokens[1])
                                prev_token = sentences[sent_counter][line_counter-1][3].lower() if line_counter > 0 else 'null'
                                if not mwc_start(tokens[1].lower(), prev_token):
                                    written_cues[cue_i] = True
                            written_cue_on_line = True
                line_counter += 1
                outfile.write("\n")
    infile.close()
    outfile.close()

def convert_scopes_to_fileformat(sentences, labels, filename, mode):
    """
    Write predicted scopes to file, using the CD format for cues and scopes
    """
    filename_base = filename.split("_cues.neg")[0]
    output_filename = filename_base + ".neg"
    infile = open(filename, "r")
    outfile = open(output_filename, "w")
    sent_counter = 0
    line_counter = 0
    scope_counter = 0
    #corenlp generates one less column in original file than conll-x format
    upper_limit = 7 if mode == "raw" else 8
    n_cues = 0
    for line in infile:
        tokens = line.split()
        if len(tokens) == 0:
            sent_counter += 1
            scope_counter += n_cues
            line_counter = 0
            n_cues = 0
            outfile.write("\n")
        elif tokens[-1] == "***":
            outfile.write(line)
        else:
            sent = sentences[sent_counter]
            cues = sent['cues']
            n_cues = len(cues)
            #write the columns in the original parsed file to the outfile
            for i in range(upper_limit):
                outfile.write("%s\t" %tokens[i])
            for cue_i in range(n_cues):
                outfile.write("%s\t" %tokens[upper_limit + 3*cue_i]) #write gold cue
                #write scope
                if in_scope_token(labels[scope_counter][line_counter], cues[cue_i][2]):
                    if cues[cue_i][2] == 'a' and sent[int(cues[cue_i][1])][3] == tokens[1]:
                        #if token matches base of affixal cue, write it as in-scope
                        outfile.write("%s\t" %(tokens[1].replace(cues[cue_i][0], "")))
                    elif tokens[upper_limit + 3*cue_i] != "_":
                        #if current token is (part of) cue, do not write it as in-scope
                        outfile.write("_\t")
                    else:
                        outfile.write("%s\t" %tokens[1])
                else:
                    outfile.write("_\t")

                outfile.write("%s\t" %tokens[upper_limit + 2 + 3*cue_i]) #write gold event
                scope_counter += 1
            
            scope_counter -= n_cues
            line_counter += 1
            outfile.write("\n")

    infile.close()
    outfile.close()


    
###################################################################
#everything from utils
import networkx as nx

def make_discrete_distance(dist):
    if dist <= 3:
        return 'A'
    elif dist <= 7:
        return 'B'
    elif dist > 7:
        return 'C'

def get_affix_cue(cue, affixal_cue_lexicon):
    for prefix in affixal_cue_lexicon['prefixes']:
        if cue.lower().startswith(prefix):
            return prefix
    for suffix in affixal_cue_lexicon['suffixes']:
        if cue.lower().endswith(suffix):
            return suffix
    for infix in affixal_cue_lexicon['infixes']:
        if infix in cue.lower() and not (cue.lower().startswith(infix) or cue.lower().endswith(infix)):
            return infix
    return None

def print_cue_lexicons(cue_lexicon, affixal_cue_lexicon):
    print ("Cues:")
    for key, value in cue_lexicon:
        print (key, value)
    print ("\nAffixal cues:")
    for cue in affixal_cue_lexicon:
        print (cue)

def make_dir_graph_for_sentence(sentence):
    graph = nx.DiGraph()
    for key, value in sentence.items():
        if isinstance(key, int):
            head_index = int(value['head']) - 1
            if head_index > -1:
                graph.add_edge(str(head_index), str(key))
    return graph

def make_bidir_graph_for_sentence(sentence):
    graph = nx.DiGraph()
    for key, value in sentence.items():
        if isinstance(key, int):
            head_index = int(value['head']) - 1
            if head_index > -1:
                graph.add_edge(str(head_index), str(key), {'dir': '/'})
                graph.add_edge(str(key), str(head_index), {'dir': '\\'})
    return graph

def get_shortest_path(graph, sentence, cue_index, curr_index):
    cue_head = int(sentence[cue_index]['head']) - 1
    if cue_head < 0 or curr_index < 0:
        return 'null'
    try:
        path_list = nx.dijkstra_path(graph, str(cue_head), str(curr_index))
        return make_discrete_distance(len(path_list) - 1)
    except nx.NetworkXNoPath:
        return 'null'

def get_dep_graph_path(graph, sentence, cue_index, curr_index):
    if cue_index < 0 or curr_index < 0:
        return 'null'
    try:
        path_list = nx.dijkstra_path(graph, str(curr_index), str(cue_index))
        prev_node = str(curr_index)
        dep_path = ""
        for node in path_list[1:]:
            direction = graph[prev_node][node]['dir']
            dep_path += direction
            if direction == '/':
                dep_path += sentence[int(node)]['deprel']
            else:
                dep_path += sentence[int(prev_node)]['deprel']
            prev_node = node
        return dep_path
    except nx.NetworkXNoPath:
        return 'null'

def get_cue_lexicon(sentence_dicts):
    """
    Extracts cue lexicon and affixal cue lexicon from the sentence dictionary structure
    """
    cue_lexicon = {}
    affixal_cue_lexicon = {'prefixes': [], 'suffixes': [], 'infixes': []}
    for sent in sentence_dicts:
        for (cue, cue_pos, cue_type) in sent['cues']:
            if cue_type == 'a':
                cue_token = sent[cue_pos][3].lower()
                if cue_token.startswith(cue.lower()):
                    if not cue.lower() in affixal_cue_lexicon['prefixes']:
                        affixal_cue_lexicon['prefixes'].append(cue.lower())
                elif cue_token.endswith(cue.lower()):
                    if not cue.lower() in affixal_cue_lexicon['suffixes']:
                        affixal_cue_lexicon['suffixes'].append(cue.lower())
                else:
                    if not cue.lower() in affixal_cue_lexicon['infixes']:
                        affixal_cue_lexicon['infixes'].append(cue.lower())
            elif cue_type == 's':
                if not cue.lower() in cue_lexicon:
                    cue_lexicon[cue.lower()] = cue_type
    return cue_lexicon, affixal_cue_lexicon

def get_character_ngrams(word, affix, m):
    n = len(word)
    return word[0:m], word[(n-m):]

def check_by_no_means(sentence, index):
    if index == 0:
        return False
    if sentence[index][3].lower() == "no" and sentence[index-1][3].lower() == "by" and sentence[index+1][3].lower() == "means":
        return True
    return False

def check_neither_nor(sentence, index):
    if sentence[index][3].lower() == "nor" and any(sentence[key][3].lower() == "neither" for key in sentence if isinstance(key,int)):
        return True
    return False

def find_neither_index(sentence):
    for key,value in sentence.items():
        if isinstance(key,int):
            if value[3].lower() == "neither":
                return key
    return -1

def find_nor_index(sentence):
    for key,value in sentence.items():
        if isinstance(key,int):
            if value[3].lower() == "nor":
                return key
    return -1

def make_complete_labelarray(sentences, labels):
    """
    Make nested label array where each label array matches the length of the sentences.
    I.e. make labels for the words that were not predicted by the cue classifier
    """
    y = []
    label_counter = 0
    for sent in sentences:
        sent_labels = []
        for key, value in sent.items():
            if isinstance(key, int):
                if 'not-pred-cue' in value:
                    sent_labels.append(-2)
                else:
                    if labels[label_counter] == -1:
                        sent_labels.append(-1)
                    else:
                        sent_labels.append(1)
                    label_counter += 1
        y.append(sent_labels)
    return y

def mwc_start(token, prev_token):
    """
    Check if the current token is part of a multiword cue
    """
    mw_lexicon = ['neither', 'by', 'rather', 'on']
    
    return any(token.lower() == w for w in mw_lexicon) or (prev_token == "by" and token == "no")

def make_splits(X, y, splits):
    """
    Split the labels from the scope prediction into nested arrays that match the sentences
    """
    i = 0
    j = 0
    X_train = []
    y_train = []
    offset = splits[j] + 1
    while j < len(splits) and offset <= len(X):
        offset = splits[j] + 1
        X_train.append(np.asarray(X[i:(i + offset)]))
        y_train.append(np.asarray(y[i:(i + offset)]))
        i += offset
        j += 1
    return np.asarray(X_train), np.asarray(y_train)

def convert_to_IO(y):
    """
    Converts beginning of scope (2) and cue (3) labels into inside (0) and outside (1) of scope
    """
    for i in range(len(y)):
        if y[i] == 2:
            y[i] = 0
        elif y[i] == 3:
            y[i] = 1
    return y

def count_multiword_cues(sentence, labels):
    mwc_counter = 0
    has_mwc = False
    for key,value in sentence.items():
        if isinstance(key,int):
            if check_by_no_means(sentence, key):
                labels[key-1] = 1
                labels[key] = 1
                labels[key+1] = 1
                mwc_counter += 1
                has_mwc = True
            if check_neither_nor(sentence, key):
                neither_i = find_neither_index(sentence)
                if not (labels[neither_i] == 1 and labels[key] == 1):
                    mwc_counter += 1
                has_mwc = True
                labels[neither_i] = 1
                labels[key] = 1

    return mwc_counter, has_mwc

def not_known_cue_word(token, cue_lexicon, affixal_cue_lexicon):
    return (not token in cue_lexicon) and get_affix_cue(token, affixal_cue_lexicon) == None

def in_scope_token(token_label, cue_type):
    return token_label == 0 or token_label == 2 or (token_label == 3 and cue_type == 'a')




            
