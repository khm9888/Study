
#import pygame.midi
# 7개의 코드생성
#import pretty_midi
import random
from pyknon.genmidi import Midi
from pyknon.music import NoteSeq, Note,Rest

chord = {}

chord["C"]=("g","e","c")
chord["Dm"]=("a","f","d")
chord["Em"]=("b","g","e")
chord["F"]=("c","a","f")
chord["G"]=("d","b","g")
chord["Am"]=("e","c","a")
chord["Bdim"]=("f","d","b")

chord["CM7"]=("b", "g", "e", "c")
chord["Dm7"]=("c", "a", "f", "d")
chord["Em7"]=("d", "b", "g", "e")
chord["FM7"]=("e", "c", "a", "f")
chord["G7"]=("f", "d", "b", "g")
chord["Am7"]=("g", "e", "c", "a")
chord["Bm7(b5)"]=("a", "f", "d", "b")

chord["Csus4"]=("g","f","c")

melody_list=["c","d","e","f","g","a","b"] #음계

#print(chord.keys())
chord_key=list(chord.keys())#코드 값만 리스트의 형태로 변환
#print(list(chord.keys()))
chord_value=list(chord.values())#멜로디 값만 리스트의 형태로 변환, type(), print()
#print(list(chord.values()))

#각 코드에 따른 멜로디를 값(values)들로 매칭시킴
order=[]

'''
new_beat = pretty_midi.PrettyMIDI('test.mid') #1
for instrument in new_beat.instruments: #2
    for note in instrument.notes:
    	cnt=0
    	#print(note.velocity, note.pitch, note.start, note.end)
    	while cnt<8:
    		n=""
    		value=note.pitch
    		if value==74:
    			n="c"
    		elif value==76:
    			n="d"
    		elif value==78:
    			n="e"
    		elif value==80:
    			n="f"
    		elif value==82:
    			n="g"
    		elif value==84:
    			n="a"
    		elif value==86:
    			n="b"
    		order.append(n)
    		cnt+=1
'''

'''
    Note_C = 74#도
    Note_D = 76#레
    Note_E = 78#미
    Note_F = 80#파
    Note_G = 82#솔
    Note_A = 84#라
    Note_B = 86#시
    Note_C_h = 88#도
    '''
#print(order)
for o in range(8):
	#inmelody=input("%d번째 멜로디를 입력해주세요" %(o+1))
	inmelody=random.choice(melody_list)
	order.append(inmelody)

melody= {}
#print(order)


for x in chord_key:
	for y in chord[x]:
		if y not in melody.keys():
			melody[y]=[] #멜로드 딕셔너리에 y값이 없다면 value로 리스트 생성
		melody[y].append(x)#x값(코드)을 y(멜로디)에 추가
#print(melody)

#각 멜로디에 따른 코드 값을 매칭시킴
#print(chord)
before=""#이전의 코드
after=""#지금의 코드
chord_list=[]
for i in range(8):#8마디
	for j in range(1):#박자 개념 추가시 j-for문 적용_현재 미적용 상태
		chord_one=[]
		if i==0: #1번째 또는 8번째는 C코드 적용
			select_key="C"

		elif i==7 and order[i]=="f":#마지막번째, 
			select_key=random.choice(melody[order[i]])
		elif i==7:
			select_key="C"
		else:
			#select_key=random.choice(chord_key)
			#print("order[i]는 %s" %order[i])
			#print("melody[%s] %s" %(order[i],melody[order[i]]))
			#print(melody[order[i]])
			select_key=random.choice(melody[order[i]])
		if i==6:
			while select_key=="C":
				select_key=random.choice(melody[order[i]])
		after=select_key
		for m in range(len(list(chord[select_key]))):
			chord_one.append(Note(chord[select_key][m]))
		chord_list.append(chord_one)
		while before==after:
			select_key=random.choice(melody[order[i]])
			after=select_key
		select_value=chord[after]

		print(chord_list)

		#print(order[i]+" : "+after+" - "+str(select_value))#1번째, C코드 출력 
		#print(select_value)
		#afterprint(select_value)

		before=select_key #1번째, C
midi = Midi(1, tempo=80)
midi.seq_chords(map(NoteSeq, chord_list))
midi.write("test.mid")


