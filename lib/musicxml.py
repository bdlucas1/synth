import sys
import xml.etree.ElementTree as ET
import notation
import builtins

at = 0
time_n = None
time_d = None
transpose_str = None

main = S()
segment = P()

def end_segment():
    global segment
    if len(segment) > 0:
        main.append(segment)
        segment = P()

def read(fn):

    global segment

    root = ET.parse(fn).getroot()

    for measure in root.findall(".//measure"):
    
        for item in measure:
    
            if item.tag == "attributes":

                # xxx does attributes always require this?
                end_segment()

                # time
                n = item.find("time/beats")
                d = item.find("time/beat-type")
                if n is not None and d is not None:
                    main.append(notation.time(int(n.text), int(d.text)))
    
                # transpose
                t = item.find("transpose/chromatic")
                if t is not None:
                    main.append(notation.transpose(int(t.text)))
                
                # divisions
                d = item.find("divisions")
                if d is not None: divisions = int(d.text)

            elif item.tag == "direction":
            
                beat_unit = item.find("direction-type/metronome/beat-unit")
                per_minute = item.find("direction-type/metronome/per-minute")
                if beat_unit is not None and per_minute is not None:
                    beat_units = {"sixteenth":16, "eighth":8, "quarter":4, "half":2, "whole":1}
                    beat_unit = beat_units[beat_unit.text]
                    main.append(notation.tempo(beat_unit, int(per_minute.text)))

            elif item.tag == "barline":

                if item.find("repeat[@direction='forward']") is not None:
                    end_segment()
                if item.find("repeat[@direction='backward']") is not None:
                    segment = segment * 2
                    end_segment()

            elif item.tag == "note":
    
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
                    while len(segment) < voice:
                        segment.append(S())
                    voice = segment[voice-1]
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
                    emit(voice, atom)
                else:
                    emit(1, atom)
                    for voice in range(2, len(segment)+1):
                        # xxx could just extend last rest
                        emit(voice, notation.Atom(pitch = "rest", dur = dur))
    

    end_segment()
    notation.Items.main = main
    return main
    
if __name__ == "__main__":
    notation.parser.add_argument("file")
    notation.parse_args()
    read(notation.args.file)


