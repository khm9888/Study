
#328, 행맨게임

#맞으면 맞춘 글자가 보이고, 틀리면 시도횟수만 하나 증가한다.(맞춰도 시도횟수 카운트는 올라간다.)
#행맨의 단어는 words.txt 파일에서 불러온다
#랜덤으로 단어 중에 하나를 가져오고, 각 단어에 대해서 앞에서 부터 한 글자씩 비교한다.
#그래서 한 번이라고 글자가 맞으면 그글자는 글자가 보이게끔 하고, 아직 맞추지 못한 글자는 그대로 출력한다.
#모든 글자가 밝혀지면 지금까지 마쳤다는 신호와 지끔까지 시도횟수를 보여준다.

import random

#파일생성.

outfile=open("d:/words.txt","w")
outfile.write("apple \n")
outfile.write("wonderful \n")
outfile.write("approve \n")
outfile.write("tablet \n")
outfile.close()

infile = open("d:/words.txt","r")
lines = infile.readlines() #리스트 형태로 들어감

word = random.choice(lines).strip()
print(word)
count=0
guess=""
question="_"*len(word)
print("%s %d글자입니다."%(question,len(question)))
while guess!=word:
    guess=input("한글자씩 맞춰보세요!(알파벳) : ")
    for w in range(len(word)):
        if word[w]==guess:
            temp_l=question[:w]
            temp_r=question[w+1:]
            question=temp_l+guess+temp_r
    if guess==word:
        question=word
    print(question)
    count +=1
    print("시도횟수 %d"%count)

print("도전완료! 정답은 %s" %question)
print("총 시도횟수 %d!!"%count)

