lowers="abcdefghijklmnopqrstuvwxyz"
def pangram(sentence_value):
	check=True
	sentence=sentence_value.lower()

	alpha=[]
	for a in range(len(lowers)):
		alpha.append(0)

	for s in sentence:
		for l in range(len(lowers)):
			if s==lowers[l]:
				alpha[l]+=1
				break

	for a in range(len(lowers)):
		if alpha[a]==0:
			check=False
			break
	print(check)




'''
>>> pangram('The quick brown fox jumps over the lazy dog.')
True
>>> pangram('The quick brown fox jumped over the lazy dog.')
False

>>> window('The quick brown fox jumps over the lazy dog.')
'quick brown fox jumps over the lazy dog'
>>> window('The quick brown fox jumped over the lazy dog.')
>>> window("I sang, and thought I sang very well; but he just looked up into my face with a very quizzical expression, and said, 'How long have you been singing, Mademoiselle?' ")
'g very well; but he just looked up into my face with a very quizzical ex'
>>> window("'We are all from Xanth,' Cube said quickly. 'Just visiting Phaze. We just want to find the dragon.'")
"from Xanth,' Cube said quickly. 'Just visiting Phaze. W"
>>> window("Further, fractal geometries are replicated on a human level in the production of certain 'types' of subjectivity: for example, aging kid quiz show whiz Donnie Smith (William H. Macy) and up and coming kid quiz show whiz Stanley Spector (Jeremy Blackman) are connected (or, perhaps, being cloned) in ways they couldn't possibly imagine.")
'bjectivity: for example, aging kid quiz show'
'''