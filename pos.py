'''
Created on Aug 22, 2013

@author: tvandrun
'''

from __future__ import division
import operator
import os
from collections import defaultdict

import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist

def nltk_to_normalized_tag(nltk_tag):
    penntb_to_reduced = {}
    # noun-like
    for x in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'EX', 'WP', 'FW', 'UH']:
        penntb_to_reduced[x] = 'N'
    # verb-like
    for x in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD', 'TO']:
        penntb_to_reduced[x] = 'V'
    # adjective-like
    for x in ['POS', 'PRP$', 'WP$', 'JJ', 'JJR', 'JJS', 'DT', 'CD', 'PDT', 'WDT', 'LS']:
        penntb_to_reduced[x] = 'AJ'
    # adverb-like
    for x in ['RB', 'RBR', 'RBS', 'WRB']:
        penntb_to_reduced[x] = 'AV'
    # preposition-like
    for x in ['IN', 'RP']:
        penntb_to_reduced[x] = 'P'
    # conjunctions
    for x in ['CC']:
        penntb_to_reduced[x] = 'C'
    # interjections
#    for x in ['FW', 'UH']:
#        penntb_to_reduced[x] = 'I'
    # symbols
#    for x in ['SYM', '$', '#']:
#        penntb_to_reduced[x] = 'S'
    # groupings
    for x in ['\'\'', '(', ')', ',', ':', '``', '"', 'SYM', "$", '#']:
        penntb_to_reduced[x] = 'G'
    # end-of-sentence symbols
    for x in ['.', '!', '?']:
        penntb_to_reduced[x] = 'E'
    return penntb_to_reduced[nltk_tag]

def main(verbose=False, corpus='baum-train-quarter.txt'):
    reader = PlaintextCorpusReader('.', '.*\.txt')
    all_text = nltk.Text(reader.words(corpus))


    all_text_lower = [x.lower() for x in all_text]
    freq_dist = FreqDist(all_text_lower)
    
    vocab = freq_dist.keys()#[:500]
    vocab.append('***')


    reduced_tags = ['N', 'V', 'AJ', 'AV', 'G', 'E', 'P', 'C']
    
    all_tagged = nltk.pos_tag(all_text)
    
    word_tag_tally = {y: {x: 1 for x in vocab} for y in reduced_tags}
    
    tag_trans_tally = {y: {x: 1 for x in reduced_tags} for y in reduced_tags}
    
    
    totals = {}
    for key in reduced_tags:
        totals[key] = 1
    
    
    previous_tag = 'E' # "ending" will be the dummy initial tag
    for (word, tag) in all_tagged :
        word = word.lower()
    
        # '-NONE-' occurs when a token doesn't match one of the standard tags. 
        # Most commonly occurs when two punctuation symbols appear beside each other 
        # with no space in between (eg ," or .")
        if(tag == '-NONE-'):
            if(word[0] == '.' or word[0] == '!' or word[0] == '?'):
                tag = 'E'
            else:
                tag = 'G'
        else:
            tag = nltk_to_normalized_tag(tag)
    
        if(word in vocab):  
            word_tag_tally[tag][word] += 1
    
        tag_trans_tally[previous_tag][tag] += 1
        totals[tag] += 1
    
        previous_tag = tag
    
    K = 1
    trans_probs = {y: defaultdict(lambda: K / (totals[y] + K * len(word_tag_tally[y].values()))) for y in reduced_tags}
    for tg1 in reduced_tags:
        for tg2 in reduced_tags:
            trans_probs[tg1][tg2] = tag_trans_tally[tg1][tg2] / totals[tg1]
    
    if verbose: print trans_probs 
    
    emit_probs = {y: defaultdict(lambda: K / (totals[y] + K * len(word_tag_tally[y].values()))) for y in reduced_tags}
    for tg in reduced_tags:
        for wd in vocab:
            emit_probs[tg][wd] = word_tag_tally[tg][wd] / totals[tg]
    
    if verbose: print emit_probs

    return [trans_probs, emit_probs, reduced_tags, 'E']


def pos_tagging(sequence, trans_probs, emit_probs, tags, start):
    O = sequence
    A = trans_probs
    B = emit_probs
    V = [{}]
    P = {}
    for tag in tags:
        V[0][tag] = A[start][tag] * B[tag][O[0]]
        P[tag] = [tag]
    for t in range(1, len(O)):
        V.append({})
        new_path = {}
        for tag in tags:
            p, best_tag = max([(V[t - 1][tag_] * A[tag_][tag] * B[tag][O[t]], tag_) for tag_ in tags], key=operator.itemgetter(0))
            V[t][tag] = p
            new_path[tag] = P[best_tag] + [tag]
        P = new_path
    p, best_tag = max([(V[t][tag], tag) for tag in tags], key=operator.itemgetter(0))
    return P[best_tag]

if __name__ == '__main__':
    main(os.getenv('POS_VERBOSE', False))
