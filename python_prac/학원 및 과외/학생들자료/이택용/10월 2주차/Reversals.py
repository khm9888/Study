lowers="abcdefghijklmnopqrstuvwxyz"
uppers=lowers.upper()


#print(lowers.index("b"))
def abecedarian(a_v,b_v):
	b=b_v.lower()
	a=a_v.lower()
	a_count=0
	b_count=0
	b_count_before=-1
	check=True
	#print(a)
	for i in range(len(a)):
		#value=a[i]
		#print(b.find(value))
		b_count=b.find(a[i])
		#print(b_count)
		#print(b_count)
		#if a[i] == "-":
		#	pass
		if b_count_before>b_count and a[i] !="-":
			check = False
			break
		b_count_before=b_count
	print(check)

#('Aegilops', 'abcdefghijklmnopqrstuvwxyz')
#abecedarian('billowy', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
#abecedarian('spoon-feed', 'abcdefghijklmnopqrstuvwxyz')
#abecedarian('spoon-feed', 'zyxwvutsrqponmlkjihgfedcba')
'''
>>> abecedarian('Aegilops', 'abcdefghijklmnopqrstuvwxyz')
True
>>> abecedarian('billowy', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
True
>>> abecedarian('spoon-feed', 'abcdefghijklmnopqrstuvwxyz')
False
>>> abecedarian('spoon-feed', 'zyxwvutsrqponmlkjihgfedcba')
True
'''

def reversal(word):
	check=[]
	word_r=word[::-1].lower()
	for w in range(len(word)):
		if word[w] in uppers:
			check.append(1)
			#word_r=word_r[:w]+word_r[w].lower()+word_r[w+1:]
		elif word[w] in lowers:
			check.append(0)
			#print(word_r[w])
			#word_r=word_r[:w]+word_r[w].upper()+word_r[w+1:]
	for w in range(len(check)):
		if check[w] ==1:
			word_r=word_r[:w]+word_r[w].upper()+word_r[w+1:]
	print(word_r)

def doubleReversal(keyword):

	list_words=keyword.split()
	#print(list_words)
	word_print=""
	for i in range(len(list_words)):
		check=[]
		word=list_words[i]
		word_r=word[::-1].lower()
		#print(word_r)
		for w in range(len(word)):
			if word[w] in uppers:
				check.append(1)
		#word_r=word_r[:w]+word_r[w].lower()+word_r[w+1:]
			elif word[w] in lowers:
				check.append(0)
		#print(word_r[w])
		#word_r=word_r[:w]+word_r[w].upper()+word_r[w+1:]
		for w in range(len(check)):
			if check[w] ==1:
				word_r=word_r[:w]+word_r[w].upper()+word_r[w+1:]
		if i !=len(list_words):
			word_print+=word_r+" "
		else: 
			word_print+=word_r
	print(word_print)



'''
>>> abecedarian('Aegilops', 'abcdefghijklmnopqrstuvwxyz')
True
>>> abecedarian('billowy', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
True
>>> abecedarian('spoon-feed', 'abcdefghijklmnopqrstuvwxyz')
False
>>> abecedarian('spoon-feed', 'zyxwvutsrqponmlkjihgfedcba')
True

>>> reversal('Marshall')
'Llahsram'
>>> reversal('BeAn')
'NaEb'
>>> reversal('Aegilops')
'Spoligea'

>>> doubleReversal('Marshall Bean')
'Naeb Llahsram'
>>> doubleReversal('Barak Obama')
'Amabo Karab'
>>> doubleReversal('Yitzhak Rabin')
'Nibar Kahztiy'
>>> doubleReversal('Jar Jar Binks')
'Sknib Raj Raj'
>>> doubleReversal('Klat Rehctub')
'Butcher Talk'
'''
