'''
x=set()

n_list=[] #set 리스트

n=3

sample=["a","b","c"]

for i in range(len(sample)): #3
	sample=["a","b","c"]
	for j in range(len(sample)):
		if i!=j:
			x.add(sample[j])
			x.add(sample[i])
			n_list.append(x)
			x.clear()		
print(n_list)
print(n_list[0])

'''
'''
print("안녕하세요")

def func(a,b,c):
	y=a+b+c
	print(y)

def func2(a,b,c):
	y=a+b+c
	return y

func(1,1,1)
print(func2(2,2,2))

a=[1,2]

x="안녕하세요"

'''