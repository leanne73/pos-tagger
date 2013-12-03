'''
Created on Aug 22, 2013

@author: tvandrun
'''

from __future__ import division
import os
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist

def main(verbose=False):
	# 1. Load a (training) corpus.
	# In the code below, the corpus will be
	# referred to by variable all_text
	reader = PlaintextCorpusReader('.', '.*\.txt')
	all_text = nltk.Text(reader.words('baum-train-quarter.txt'))
	
	
	# make the training text lowercase
	all_text_lower = [x.lower() for x in all_text]
	freq_dist = FreqDist(all_text_lower)
	
	# make a reduced vocabulary (here, 500 types)
	vocab = freq_dist.keys()[:500]
	vocab.append('***')
	
	
	
	
	# 2. Make a reduced form of the PennTB tagset
	penntb_to_reduced = {}
	# noun-like
	for x in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'EX', 'WP']:
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
	for x in ['FW', 'UH']:
	    penntb_to_reduced[x] = 'I'
	# symbols
	for x in ['SYM', '$', '#']:
	    penntb_to_reduced[x] = 'S'
	# groupings
	for x in ['\'\'', '(', ')', ',', ':', '``', '"']:
	    penntb_to_reduced[x] = 'G'
	# end-of-sentence symbols
	for x in ['.', '!', '?']:
	    penntb_to_reduced[x] = 'E'
	
	reduced_tags = ['N', 'V', 'AJ', 'AV', 'I', 'S', 'G', 'E', 'P', 'C']
	
	# 3. tag the corpus
	all_tagged = nltk.pos_tag(all_text)
	
	# 4. make the probability matrices
	
	# a tally from types to tags; a tally from tags to next tags
	# LaPlace smoothing---add 1 to each
	
	# emission probs
	word_tag_tally = {y:{x:1 for x in reduced_tags} for y in vocab}
	
	#transition probs
	tag_trans_tally = {y:{x:1 for x in reduced_tags} for y in reduced_tags}
	
	
	#keeps track of the total individual occurences of each tag
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
	        tag = penntb_to_reduced[tag]
	
	    if(word in vocab):  
	        word_tag_tally[word][tag] += 1
	
	    tag_trans_tally[previous_tag][tag] += 1
	    totals[tag] += 1
	
	    previous_tag = tag
	
	# fill this out:
	#	For each tag tg1 compute the probabilities for transitioning to
	#	each tag (say, tg2). Using relative frequency estimation,
	#	that would mean dividing the number of times tg2 follows tg1 by
	#	the absolute number of times t1 occurs. (But, what if tg1 never occurs..?)
	#	Recommendation: think in terms of "for each tg2, how many times had
	#	we transitioned from tg1?"
	
	# now, make the actual transition probability matrices 
	trans_probs = {y:{x:0 for x in reduced_tags} for y in reduced_tags}
	for tg1 in reduced_tags:
	    for tg2 in reduced_tags:
	        trans_probs[tg1][tg2] = tag_trans_tally[tg1][tg2] / totals[tg1]
	
	if verbose: print trans_probs 
	
	# Fill this out:
	#	For each tag tg1 compute the probabilities for emitting each word v.
	#	Recommendation: think in terms of "for each word v, how many times
	#	did tg1 emit v?"
	emit_probs = {y:{x:0 for x in vocab} for y in reduced_tags}
	for tg in reduced_tags:
	    for wd in vocab:
	        emit_probs[tg][wd] = word_tag_tally[wd][tg] / totals[tg]
	
	if verbose: print emit_probs

	# 6. try it out: run the algorithm on the test data
	test_sample = []
	test_tagged = pos_tagging(test_sample, trans_probs, emit_probs, vocab, reduced_tags, 'E')            


# 5. implement Viterbi. 
# Write a function that takes a sequence of tokens,
# a matrix of transition probs, a matrix of emit probs,
# a vocabulary, a set of tags, and the starting tag

def pos_tagging(sequence, trans_probs, emit_probs, vocab, tags, start) :

    #TODO

    tagged = []

        
    return tagged

if __name__ == '__main__':
	main(os.getenv('POS_VERBOSE', False))
