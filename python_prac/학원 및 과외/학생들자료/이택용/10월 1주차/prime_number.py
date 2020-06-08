x=input()
a=x[0]#.
b=x[1]#8
column=0#c
lock=0

while lock==0:
	row=input()
	d=len(row)
	column+=1
	#print(line)
	for i in range(d):
		if a != row[i] and b != row[i]:
			print("character \'%s\' only occur at row %d and column %d"%(row[i],column,i+1))
			lock=1

