#4

f = open("D:\\Study/book/김예림씨/MEDIUM4_rearrangeSentences.txt","r")
order = []
line_order = []
lines = f.readlines()
for line in lines:
    order.append(line[9])
    line_order.append(line)
f.close()

for i in range(len(order)):
    order[i]=ord(order[i])



for i in range(len(order)-1):
    for j in range(i,len(order)):
        if order[i]>order[j]:
            order[i],order[j]=order[j],order[i]
            line_order[i],line_order[j]=line_order[j],line_order[i]
        
# print(order)
# print(line_order)

f2 = open("D://Study/book/김예림씨/MEDIUM4_rearrangeSentences.txt","w")
for line in line_order:
    f2.write(line)
f2.close()
