import sys
import xml.etree.ElementTree as ET
import notation
import builtins
from fractions import Fraction

at = 0
time_n = None
time_d = None
transpose_str = None

class MXML:

    def __init__(self):
        self.main = P()

    def read(self, fn):

        root = ET.parse(fn).getroot()

        voices = set(voice.text for voice in root.findall(".//voice"))
        for voice in voices:
            self.scan(root, voice)
            self.main.append(self.voice_segment)

        notation.Items.main = self.main
        return self.main

    def active_segment(self):
        if self.ending_segment is not None:
            return self.ending_segment
        elif self.repeat_segment is not None:
            return self.repeat_segment
        else:
            return self.voice_segment        

    # implict repeat back to beginning
    def repeat_to_beginning(self):
        self.repeat_segment = self.voice_segment * 2
        self.voice_segment = S(self.repeat_segment)
        self.last_repeat_segment = self.repeat_segment

    def scan(self, root, for_voice):

        self.voice_segment = S()
        self.repeat_segment = None
        self.last_repeat_segment = None
        self.ending_segment = None

        for measure in root.findall(".//measure"):

            measure_divisions = 0

            for item in measure:

                if item.tag == "attributes":

                    # xxx does attributes always require this?
                    #self.end_segment()

                    # divisions
                    d = item.find("divisions")
                    if d is not None: divisions = int(d.text)

                    # time
                    num, den = item.find("time/beats"), item.find("time/beat-type")
                    if num is not None and den is not None:
                        num, den = int(num.text), int(den.text)
                        self.active_segment().append(notation.time(num, den))
                        self.divisions_per_measure = divisions * 4 * num / den

                    # transpose
                    t = item.find("transpose/chromatic")
                    if t is not None:
                        self.active_segment().append(notation.transpose(int(t.text)))

                elif item.tag == "direction":

                    beat_unit = item.find("direction-type/metronome/beat-unit")
                    per_minute = item.find("direction-type/metronome/per-minute")
                    if beat_unit is not None and per_minute is not None:
                        beat_units = {"sixteenth":16, "eighth":8, "quarter":4, "half":2, "whole":1}
                        beat_unit = beat_units[beat_unit.text]
                        if item.find("direction-type/metronome/beat-unit-dot") is not None:
                            beat_unit = beat_unit * Fraction(2, 3)
                        tempo = notation.tempo(beat_unit, int(per_minute.text))
                        self.active_segment().append(tempo)

                elif item.tag == "barline":

                    if item.find("repeat[@direction='forward']") is not None:
                        self.repeat_segment = S() * 2
                        self.voice_segment.append(self.repeat_segment)
                        self.last_repeat_segment = self.repeat_segment # for endings
                    if item.find("repeat[@direction='backward']") is not None:
                        if self.repeat_segment is None:
                            self.repeat_to_beginning()
                        self.repeat_segment = None

                    if (ending := item.find("ending[@type='start']")) is not None:
                        if self.last_repeat_segment is None:
                            self.repeat_to_beginning()
                        E = builtins.__dict__["E" + ending.attrib["number"]]
                        self.ending_segment = E()
                        self.last_repeat_segment.append(self.ending_segment)
                    if (ending := item.find("ending[@type='stop']")) is not None:
                        self.ending_segment = None

                elif item.tag == "backup":
                        
                    measure_divisions -= int(item.find("duration").text)

                elif item.tag == "note":

                    # voice
                    voice = item.find("voice")
                    voice = voice.text if voice is not None else None

                    # pitch
                    pitch = item.find("pitch/step").text.lower() + item.find("pitch/octave").text
                    alter = item.find("pitch/alter")
                    alter = int(alter.text) if alter is not None else 0
                    pitch = getattr(builtins, pitch).pitch + alter# provided by notations

                    # duration
                    dur_divisions = int(item.find("duration").text)
                    dur_units = notation.to_units(dur_divisions, divisions = divisions)

                    if item.find("chord") is None:
                        measure_divisions += dur_divisions

                    if voice is None or voice == for_voice:

                        # atom
                        if voice is None and for_voice != "1":
                            pitch = "rest"
                        atom = notation.Atom(pitch = pitch, dur_units = dur_units)

                        # chord?
                        if item.find("chord") is not None:
                            last = self.active_segment()[-1]
                            if isinstance(last, notation.P):
                                last.append(atom)
                            else:
                                chord = P(last, atom)
                                self.active_segment()[-1] = chord
                        else:
                            self.active_segment().append(atom)

                    self.measure_segment = self.active_segment()

            # bar check at end of each measure, but not incomplete measures, except the first one
            measure_number = measure.attrib["number"]
            if measure_divisions % self.divisions_per_measure == 0 or measure_number == "1":
                i = I.copy()
                i.measure = measure_number
                self.measure_segment.append(i)

if __name__ == "__main__":
    notation.parser.add_argument("file")
    notation.parse_args()
    MXML().read(notation.args.file)


