#hard 4\
    
text = input("Type the input string:").split(" ")

lines_list=[]
i=1

while True:
    t = str(i)
    t+=" "
    for j in range(i):
        try:
            t+=(text.pop(0)+" ")
        except:
            if j == i-1:
                t+="-"
            else:
                t+="- "
    t+="\n"
    lines_list.append(t)
    if not text:
        break 
    i+=1
    
f2 = open("D://Study/book/김예림씨/wordsOnLine.txt","w")
for line in lines_list:
    f2.write(line)
f2.close()