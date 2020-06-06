vowels_1st=["a", "e", "i", "o", "u", "y"]
vowels=vowels_1st
for v in range(len(vowels_1st)):
		vowels.append(vowels_1st[v].upper())
#print(vowels)
def chemifyWord(word):
	word_key=word
	check = True
	cnt=1
	while check:
		for v in vowels:
			last = word_key[-cnt]
			if last in v:
				word_key=word_key[:-cnt]
		if last not in v:
			check=False
		#print(word_key)
	print(word_key+"ium")


'''
>>> chemifyWord('California')
'Californium'
>>> chemifyWord('BERKELEY')
'BERKELium'
>>> chemifyWord('of')
'ofium'
>>> chemifyWord('Belgium')
'Belgium'

'''
