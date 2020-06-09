#5

f = open("D://Study/book/김예림씨/MEDIUM5_replaceWord.txt","r")

name = input("input your name : ")
lines_list=[]
lines = f.readlines()
for line in lines:
    cnt=0
    for i,j in enumerate(line):
        if line[i:i+len("-*-*-")]=="-*-*-":
            line=line[:i]+name+line[i+len("-*-*-"):]
    # line+="\n"a
    lines_list.append(line)
f.close()
f2 = open("D://Study/book/김예림씨/MEDIUM5_replaceWord.txt","w")
for line in lines_list:
    f2.write(line)
f2.close()