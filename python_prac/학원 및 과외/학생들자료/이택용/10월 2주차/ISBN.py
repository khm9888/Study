value1=input()
str_value=""
#조건 1. 10자리여야한다.
#1~9번째 자리수의 합 을 11로 나눈 나머지의 값은 10번째 값과 같아야한다.
#문자열이여야한다.

def isISBN(value):
	final=0
	total=0
	mod=1
	check=False
	if len(value)==10: #조건 1
		for v in range(len(value)):
			if v==9:
				final=int(value[v])
				print("final는 %d"%final)
			if v==8:
				total+=int(value[v])
				print(total)
				mod = total%11
				print("mod는 %d"%mod)
				
			else:
				total+=int(value[v])
				print(total)
	if final==mod:
		check=True
	#print(type(value))
	if check and type(value)==str:
		print(True)
	else:
		print(False)

isISBN(value1)
