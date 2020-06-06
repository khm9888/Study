import pysynth as ps
from pykono.genmidi import Midi
from pykono.music import NoteSeq, Note, Rest
from MarkoMusic import MusicMatrix

song = [['c4',4 ],['c4',4 ], ['c4',4 ],['d4',8 ]]

ps.make_wav(song, fn='examples/test.make_wav')
