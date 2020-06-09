#hard 2


f = open("D://Study/book/김예림씨/HARD2_recursiveSumming.txt","r")

s = 0
lines = f.readlines()
for line in lines:
    values = list(map(int,line.split("-")))
    s+=sum(values)
f.close()
print(s)
