#타이핑 게임 만들기.

#타이핑 단어 개수 입력
#타이핑 단어 입력
#각 단어가 랜덤으로 1개씩 보여질 때 맞춰서 타이핑하는 게임
#각 단어를 타이핑할 때마다 타이핑 속도, 정확도 등이 출력됨.

import random
import time
import hgtk
import os

WORD_LIST = []
#단어리스트 생성
#몇 개를 입력 받으시겠습니까?
word_count=int(input("몇 개를 입력 받으시겠습니까?"))
for i in range(1,word_count+1):
	x=input("%s번째 단어를(문장을) 입력하세요"%i)
	WORD_LIST.append(x)
#단어 후보 등록
random.shuffle(WORD_LIST)
list_len = len(WORD_LIST)#5개
current_count = 0#초기값

while current_count < list_len:#단어 개수만큼 진행 - 종료값
    #os.system("cls")
    q = WORD_LIST[current_count]
    current_count += 1
    
    start_time = time.time()
    user_input = input(q + '\n')
    end_time = time.time() - start_time

    src = hgtk.text.decompose(q).replace("ᴥ", "")
    print(src)
    tar = hgtk.text.decompose(user_input).replace("ᴥ", "")

    correct = 0
    for i, c in enumerate(src, start=0):
        try:
            if tar[i] == c:
                correct += 1
        except:
            pass
    enumerate(
    src_len = len(src)
    c = correct / src_len * 100 # 정확도
    e = (src_len - correct) / src_len * 100 # 오타율
    speed = float(correct / end_time) * 60
    
    print("속도: {:0.2f} 정확도: {:0.2f} % 오타율: {:0.2f} %".format(speed, c, e))
    #os.system("pause")
