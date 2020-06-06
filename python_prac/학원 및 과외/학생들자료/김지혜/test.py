from pyknon.genmidi import Midi
from pyknon.music import NoteSeq

notes1 = NoteSeq("Cm4 F#4 A Bb4")
midi = Midi(1, tempo=90)
midi.seq_notes(notes1, track=0)
midi.write("demo.mid")