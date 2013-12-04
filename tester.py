from __future__ import division
import nltk
from nltk.corpus import PlaintextCorpusReader
import pos

for corpus in ['baum-train-quarter.txt', 'baum-train-half.txt', 'baum-train.txt']:
	correct_tags = []
	with open('baum-test-hand.txt') as hand:
		for line in hand:
			token, tag = line.strip().split()
			correct_tags.append(tag)
	
	trans_probs, emit_probs, tags, start = pos.main(corpus=corpus)
	with open('baum-tiny-test.txt') as corpus:
		words = " ".join(corpus)
		generated_tags = pos.pos_tagging(words.strip().split(), trans_probs, emit_probs, tags, start)
	
	reader = PlaintextCorpusReader('.', '.*\.txt')
	with open('baum-tiny-test.txt') as baum:
		test_text = nltk.Text(" ".join(baum).strip().split())
	
	nltk_tags = nltk.pos_tag(test_text)
	nltk_tags = [pos.nltk_to_normalized_tag(tag) for (word, tag) in nltk_tags]
	
	
	#crank magic comparing correct_ and generated_tags and the NLTK tags
	
	def percent_correct(tags):
		correct = 0
		for x in range(len(tags)):
			if tags[x] == correct_tags[x]:
				correct += 1
		return (correct / len(tags)) * 100
	
	print "%s:"%corpus
	print "\tPercent Correct of Our Tags:"
	print percent_correct(generated_tags)
	print "\tPercent Correct of nltk Tags:"
	print percent_correct(nltk_tags)
