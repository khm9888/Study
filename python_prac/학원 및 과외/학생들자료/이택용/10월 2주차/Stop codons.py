

def isStopCodon(key):
	StopCodons=['TAG', 'TGA', 'TAA']
	key_search=key.upper()
	if key_search in StopCodons:
		print(True)
	else:
		print(False)

#isStopCodon('TAA')
#isStopCodon('tag')
#isStopCodon('ATC')
'''
>>> isStopCodon('TAA')
True
>>> isStopCodon('tag')
True
>>> isStopCodon('ATC')
False
'''

def reverseComplement(word):
	key = word[::-1]
	complement=""
	#print(key)
	for x in key:
		if x=="A"or x=="a":
			complement+="T"

		elif x=="T"or x=="t":
			complement+="A"

		elif x=="G"or x=="g":
			complement+="C"

		elif x=="C" or x=="c":
			complement+="G"
	print(complement)

#A 랑 T 가 짝이고 G 랑 C 가 짝
'''
>>> reverseComplement('AAGTC')
'GACTT'
>>> reverseComplement('agcttcgt')
'ACGAAGCT'
>>> reverseComplement('AGTCTTACGCTTA')
'TAAGCGTAAGACT'
'''
#reverseComplement('AAGTC')
#reverseComplement('agcttcgt')
#reverseComplement('AGTCTTACGCTTA')
