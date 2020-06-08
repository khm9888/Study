import pysynth as ps
from pyknon.genmidi import Midi
from pyknon.music import NoteSeq, Note, Rest
from MarkovMusic import MusicMatrix

song = [['c',4],['c',8],['d',8]]

#s.make_wav(song,fn="examples/test.wav")#wav 파일을 만드는 함수.

matrix = MusicMatrix()

for i in song:
	matrix.add(i)

stare_note = ['c',4]

random_song=[]
for i in range(0,100):
	stare_note = matrix.next_note(stare_note)
	random_song.append(stare_note)

#ps.make_wav(random_song,fn="examples/random_song.wav")


#midi.write(midi_path="midi/random_rowboat.mid",notes=random_song)
#midi.seq_notes(NoteSeq([Note(x%12, 4, 1/16) for x in seq]))
notes1 = NoteSeq("D4 F#8 A Bb4")
midi = Midi(1, tempo=90)
midi.seq_notes(notes1, track=0)
midi.write("demo.mid")
