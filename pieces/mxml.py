import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

#voices = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: []))
voices = defaultdict(lambda: defaultdict(lambda: []))

at = 0
time_n = None
time_d = None
transpose_str = None

root = ET.parse(sys.argv[1]).getroot()

for measure in root.findall(".//measure"):

    # time
    n = measure.find("attributes/time/beats")
    d = measure.find("attributes/time/beat-type")
    if n is not None and d is not None:
        time_n = int(n.text)
        time_d = int(d.text)

    # transpose
    t = measure.find("attributes/transpose/chromatic")
    if t is not None: transpose_str = f"transpose({t.text})"

    # repeat
    rpt = measure.find("barline/repeat[@direction='forward']")
    #if rpt is not None: print("S(")

    d = measure.find("attributes/divisions")
    if d is not None: divisions = int(d.text)

    for item in measure:

        if item.tag == "note":

            # voice
            voice = item.find("voice")
            voice = int(voice.text) if voice is not None else 1
    
            # pitch
            pitch_step = item.find("pitch/step").text.lower()
            pitch_octave = item.find("pitch/octave").text
            pitch = pitch_step + pitch_octave
    
            # duration
            duration = int(item.find("duration").text)

            # chord?
            if item.find("chord") is not None: at -= duration

            # emit
            voices[voice][at].append((pitch, duration))
    
            # move on
            at += duration

        elif item.tag == "backup":

            # move back
            at -= int(item.find("duration").text)

#
#
#

indent = 0

def p(s):
    print(" " * indent + s)

def enter(s):
    global indent
    p(s + "(")
    indent += 2

def leave(s):
    global indent
    indent -= 2
    p(")")

def note_string(note):
    return note[0] + "/" + duration_string(note[1])

def duration_string(duration):
    duration_units = duration / (4 * divisions)
    dt = tuple()
    d = 1
    while duration_units > 0:
        if 1/d <= duration_units:
            dt = dt + (d,)
            duration_units -= 1/d
        d *= 2
    return str(dt[0]) if len(dt)==1 else f"({','.join(str(d) for d in dt)})"
    
enter("P")
p(f"time({time_n},{time_d}), ")
if transpose_str is not None: p(transpose_str + ", ")
for voice in voices.values():
    enter("S")
    running_at = 0
    measure_str = None
    for at, notes in voice.items():
        if running_at != at:
            p(note_string(("r", at-running_at)))
            running_at = at
        measure_divisions = divisions * time_n / time_d * 4
        if running_at % measure_divisions == 0:
            if measure_str is not None:
                p(measure_str)
            measure_str = "I"
        if len(notes) == 1:
            measure_str += ", " + note_string(notes[0])
        else:
            measure_str += ", P(" + ",".join(note_string(note) for note in notes) + ")"
        running_at += notes[0][1]
    if measure_str is not None:
        p(measure_str)
    leave("S")
leave("P")


