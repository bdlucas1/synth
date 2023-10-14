import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
import notation
import builtins

notation.parser.add_argument("file")
notation.parse_args()

voices = defaultdict(notation.S)

at = 0
time_n = None
time_d = None
transpose_str = None

def process(root):

    main = P()

    for measure in root.findall(".//measure"):
    
        # time
        n = measure.find("attributes/time/beats")
        d = measure.find("attributes/time/beat-type")
        if n is not None and d is not None:
            main.append(notation.time(int(n.text), int(d.text)))
    
        # transpose
        t = measure.find("attributes/transpose/chromatic")
        if t is not None:
            main.append(notation.transpose(int(t.text)))
    
        # repeat
        # xxx ignored for now
        rpt = measure.find("barline/repeat[@direction='forward']")
    
        # divisions
        d = measure.find("attributes/divisions")
        if d is not None: divisions = int(d.text)
    
        for item in measure:
    
            if item.tag == "note":
    
                # voice
                voice = item.find("voice")
                if voice is not None: voice = int(voice.text)
        
                # pitch
                pitch = item.find("pitch/step").text.lower() + item.find("pitch/octave").text
                pitch = getattr(builtins, pitch).pitch # provided by notations
        
                # duration
                dur_divisions = int(item.find("duration").text)
                dur = notation.T(dur_divisions, divisions = divisions)
    
                # atom
                atom = notation.Atom(pitch = pitch, dur = dur)
    
                # chord?
                is_chord = item.find("chord") is not None
                
                def emit(voice, atom):
                    if is_chord:
                        last = voice[-1]
                        if isinstance(last, notation.P):
                            last.append(atom)
                        else:
                            chord = P(last, atom)
                            voice[-1] = chord
                    else:
                        voice.append(atom)

                # emit
                if voice is not None:
                    emit(voices[voice], atom)
                else:
                    emit(voices[1], atom)
                    for voice in voices.keys():
                        if voice != 1:
                            emit(voices[voice], notation.Atom(pitch = "rest", dur = dur))
    
    for voice in voices.values():
        main.append(voice)
    notation.Items.main = main
    return main
    
if __name__ == "__main__":
    root = ET.parse(notation.args.file).getroot()
    main = process(root)
    print(main.to_str(0))


