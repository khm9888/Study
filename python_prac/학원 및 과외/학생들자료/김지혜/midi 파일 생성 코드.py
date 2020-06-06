from pyknon.genmidi import Midi
from pyknon.music import NoteSeq

#notes1 = NoteSeq([["C4"], ["F#4"], ["A"], ["Bb4"]])
chord1 = NoteSeq([["C4"], ["F#4"], ["A"], ["Bb4"]])
#harmony1= harmonize(["C4", "F#4", "A"])
print(notes1)
midi = Midi(1, tempo=90)
midi.seq_notes(notes1, track=0)
midi.seq_chords(chord1)
print(midi)
#midi.seq_notes(harmony1, track=0)
midi.write("demo.mid")