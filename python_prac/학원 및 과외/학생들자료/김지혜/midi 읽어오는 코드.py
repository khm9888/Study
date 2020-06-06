import pretty_midi
new_beat = pretty_midi.PrettyMIDI('test.mid') #1
for instrument in new_beat.instruments: #2
    for note in instrument.notes:
    	cnt=0
    	print(note.velocity, note.pitch, note.start, note.end)
    	while cnt<8:
    		value=Note(note.pitch)
        	cnt+=1
        #notes1 = NoteSeq("D4 F#8 A Bb4")