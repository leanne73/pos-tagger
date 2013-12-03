correct_tags = []
with open('baum-test-hand.txt') as hand:
	for line in hand:
		token, tag = line.split()
		correct_tags.append((token, tag))

generated_tags = [] 
with open('baum-test-machine.txt') as machine:
	for line in machine:
		token, tag = line.split()
		generated_tags.append((token, tag))

#crank NLTK magic here

#crank magic comparing correct_ and generated_tags and the NLTK tags
