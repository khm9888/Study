import pygame
import pygame.midi
from time import sleep
 
def midiExample():
    # Things to consider when using pygame.midi:
    #
    # 1) Initialize the midi module with a to pygame.midi.init().
    # 2) Create a midi.Output instance for the desired output device port.
    # 3) Select instruments with set_instrument() method calls.
    # 4) Play notes with note_on() and note_off() method calls.
    # 5) Call pygame.midi.Quit() when finished. Though the midi module tries
    #    to ensure that midi is properly shut down, it is best to do it
    #    explicitly. A try/finally statement is the safest way to do this.
    #
    GRAND_PIANO = 0
    CHURCH_ORGAN = 19
    Note_C = 74#도
    Note_D = 76#레
    Note_E = 78#미
    Note_F = 80#파
    Note_G = 82#솔
    Note_A = 84#라
    Note_B = 86#시
    Note_C_h = 88#도
    
    Note = [Note_C, Note_D, Note_E, Note_F, Note_G, Note_A, Note_B, Note_C_h]
    
 
    
    instrument = GRAND_PIANO
    
    pygame.init()
    pygame.midi.init()
 
    port = pygame.midi.get_default_output_id()
    print ("using output_id :%s:" % port)
    midi_out = pygame.midi.Output(port, 0)
    
    try:
        midi_out.set_instrument(instrument)
        
        #Play notes
        for n in Note :
            midi_out.note_on(n, 127) # 74 is middle C, 127 is "how loud" - max is 127
            sleep(.5)
            midi_out.note_off(n, 127)
            sleep(.5)
 
    finally:
        del midi_out
        pygame.midi.quit()
        
midiExample()

