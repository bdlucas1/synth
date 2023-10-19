import engine
import builtins
import numpy as np
import inspect
import math
import random
import sys
import time as sys_time
import atexit
import argparse
import os
import fractions

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--play", action="store_true")
parser.add_argument("-l", "--loop", action="store_true")
parser.add_argument("--print", action="store_true")
parser.add_argument("-s", "--save", action="store_true")
parser.add_argument("-d", "--dbg", action="store_true")
parser.add_argument("-b", "--bars")

def parse_args():
    global args
    args = parser.parse_args()

# process --dbg arg
def dprint(*s):
    if args.dbg:
        print(*s)

#
# a unit is a fraction of a whole note, used for durations, time in bars, etc.
# we delegate instead of inheriting so that e.g. Units+Units would returns Units, not Fraction
# this could be considered a design flaw in Fraction: it should return instance of subclass
#
def class_init(cls):
    cls.class_init()
    return cls

@class_init
class Units:

    unwrap = lambda x: x.f if isinstance(x, Units) else x
    wrap = lambda x: Units(x) if isinstance(x, fractions.Fraction) else x

    @classmethod
    def delegate(cls, m):
        m = "__" + m + "__"
        def delegated(self, *args):
            args = map(cls.unwrap, args)
            return cls.wrap(getattr(self.f, m)(*args))
        setattr(cls, m, delegated)

    @classmethod
    def class_init(cls):
        # extend this list as needed
        for m in "add sub mul truediv radd rtruediv int float mod round gt le eq hash".split():
            cls.delegate(m)

    def __init__(self, *args):
        args = map(Units.unwrap, args)
        self.f = fractions.Fraction(*args).limit_denominator(64*9*25*49)

    # this makes [Units] print as "[n/d]" instead of "[Fraction(n,d)]"
    def __repr__(self):
        return str(self.f)


def to_units(dur: int|tuple|Units, divisions = None) -> Units:
    if divisions is not None:
        # musicxml measures time in units of divisions of a quarter note
        return Units(dur, 4 * divisions)
    elif isinstance(dur, tuple):
        # sum of reciprocals: (2,) is half note, (4,) quarter note, (2,4) dotted half, ...
        return sum(Units(1, d) for d in dur)
    elif isinstance(dur, Units):
        return dur
    elif isinstance(dur, (int,float)):
        return Units(dur)
    else:
        raise Exception(f"bad dur {dur} {type(dur)}")

def to_secs(units: Units, tempo: tuple):
    return float(units * Units(*tempo) * 60)

def to_bars(units: Units, time: tuple):
    return units / Units(*time)

def to_tuple(units: Units):
    d = Units(1)
    result = []
    while units > 0:
        if 1 / d <= units:
            result.append(int(d))
            units -= 1 / d
        d *= 2 # xxx doesn't do triplets
        #d += 1 # but this gives (3,24) instead of (4,8) :(
    return tuple(result)

def to_tuple_str(units: Units):
        t = to_tuple(units)
        if len(t) == 1:
            return str(t[0])
        else:
            return "(" + ",".join(str(tt) for tt in t) + ")"

def normalize_contour(contour):
    if isinstance(contour, list):
        for i, c in enumerate(contour):
            if isinstance(c, tuple):
                 contour[i] = tuple([to_units(c[0]), *c[1:]])

class Atom:

    def __init__(self, **parms):
        self.exclude = {"exclude"}
        self.__dict__.update(parms)
        self.breaking = False

    def to_str(self, level = -1):

        def slash(attr):
            return "%" if attr in self.exclude else "/"

        if hasattr(self, "dur_units"):
            dur = slash("dur_units") + to_tuple_str(self.dur_units)
        else:
            dur = ""

        # xxx probably needs to take key into account to get correct enharmonic
        if hasattr(self, "pitch"):
            if self.pitch == "rest": pitch = "r"
            elif self.pitch == "tie": pitch = "t"
            else: pitch = pitch2str[self.pitch]
            result = pitch + dur
        elif hasattr(self, "relpitch"):
            result = relpitch2str[self.relpitch] + dur
        elif hasattr(self, "time"):
            self.breaking = True
            result = "time(" + str(self.time[0]) + "," + str(self.time[1]) + ")"
        elif hasattr(self, "tempo"):
            self.breaking = True
            num = to_tuple_str(Units(1, self.tempo[0]))
            den = str(self.tempo[1])
            result = "tempo(" + num + "," + den + ")"
        elif hasattr(self, "transpose"):
            self.breaking = True
            result = "transpose(" + str(self.transpose) + ")"
        elif hasattr(self, "bar"):
            self.breaking = True
            result = "~I"
        elif hasattr(self, "instrument"):
            self.breaking = True
            result = self.instrument
        else:
            self.breaking = True
            result = str(self.__dict__)

        # modifiers
        if hasattr(self, "pcs"):
            for pc in self.pcs:
                result += "@" + str(pc).replace(" ", "")
        if hasattr(self, "vcs"):
            for vc in self.vcs:
                result += ">" + str(vc).replace(" ", "")
        if hasattr(self, "ring"):
            result += slash("ring") + "ring"
        if hasattr(self, "vol"):
            result += slash("vol") + "v" + str(self.vol)

        return result

    class Dbg:
        def __init__(self, frame):
            info = inspect.getframeinfo(frame)
            self.fn = info.filename
            self.line = info.lineno
            self.col = info.positions.col_offset + 1 if hasattr(info, "positions") else None

    # has to be a copy of the above code else frame is wrong
    def __invert__(self):
        copy = self.copy()
        frame = inspect.currentframe().f_back
        copy.dbg = Atom.Dbg(frame)
        return copy

    def loc(self):
        if hasattr(self, "dbg"):
            loc = f"{self.dbg.fn}:{self.dbg.line}"
            if self.dbg.col is not None:
                loc += f":{self.dbg.col}"
            if hasattr(self, "time"):
                num, den = self.time
                bars, beats = math.floor(self.t_bars), round((self.t_bars % 1) * num)
                loc += f": bar {bars+1} beat {beats+1}:"
            return loc
        else:
            return ""

    #
    #
    #

    def item_contour(self, contour):

        # compute item_contour
        item_contour = []
        want_dur = self.dur_units
        if isinstance(contour, (int,float)):
            contour = [(self.dur_units, contour)]
        while want_dur and len(contour):
            if isinstance(contour[0], (int,float)):
                contour = contour.copy()
                contour[0] = [self.dur_units, contour[0]]
            have_dur = contour[0][0] # assumes already normalized so this is units (i.e. Units)
            dv = contour[0][1:]
            if want_dur == have_dur:
                have_dur = want_dur
            if want_dur >= have_dur:
                item_contour.append((have_dur, *dv))
                contour = contour[1:]
                want_dur -= have_dur
            else: # want_dur < have_dur
                item_contour.append((want_dur, *dv))
                contour = [(have_dur-want_dur, *dv)] + contour[1:]
                want_dur = 0

        # discontinuity check
        last_dv_end = None
        for segment in item_contour:
            if len(segment) == 2:
                _, dv_start = segment
                dv_end = dv_start
            elif len(segment) == 3:
                _, dv_start, dv_end = segment
            if last_dv_end != None and last_dv_end != dv_start:
                print(self.loc(), "warning: volume contour discontinuity")
            last_dv_end = dv_end

        # compute dbg
        fmt = lambda dv: ":".join(str(v) for v in dv)
        dbg = ",".join(fmt(v[1:]) for v in item_contour)

        return contour, item_contour, dbg


    def compute_contour(self, t2i, item_contour):

        # optimization: single segment of same length as self is just returned as a number
        if len(item_contour)==1 and len(item_contour[0])==2 and item_contour[0][0]==self.dur_units:
            return item_contour[0][1]

        # compute segments
        segments = []
        for segment in item_contour:
            if len(segment) == 2:
                dur, dv_start = segment
                dv_end = dv_start
            elif len(segment) == 3:
                dur, dv_start, dv_end = segment
            dur_secs = to_secs(dur, self.tempo)
            n = t2i(dur_secs)
            segments.append(np.interp(range(n), [0,n], [dv_start, dv_end]))

        # concatenate segments
        return np.concatenate(segments)


    def jitter(self, var):
        var = "jitter_" + var
        if hasattr(self, var):
            jitter = (random.random() - 0.5) * self.__dict__[var]
            return jitter
        else:
            return 0

    def __truediv__(self, other):
        return self.copy(other)

    def __mod__(self, other):
        copy = self.copy()
        if isinstance(other, (float, int)):
            copy.exclude.add("dur_units")
        else:
            copy.exclude.update(set(other.__dict__.keys()))
        return copy / other

    def copy(self, other = None):

        # copy self
        result = Atom()
        result.__dict__.update(self.__dict__)
        if hasattr(self, "vcs"):
            result.vcs = self.vcs.copy()
        if hasattr(self, "pcs"):
            result.pcs = self.pcs.copy()

        # merge other
        if other:

            # special cases for /n durs
            if isinstance(other, (float, int)):
                other = Atom(dur_units = to_units((other,)))
            elif isinstance(other, tuple):
                other = Atom(dur_units = to_units(other))

            # merge attributes
            result.__dict__.update(other.__dict__)

            # merge excludes
            result.exclude = self.exclude.union(other.exclude)

            # merge vcs
            if hasattr(other, "vcs"):
                if hasattr(self, "vcs"):
                    result.vcs = self.vcs + other.vcs
                else:
                    result.vcs = other.vcs

            # merge pcs
            if hasattr(other, "pcs"):
                if hasattr(self, "pcs"):
                    result.pcs = self.pcs + other.pcs
                else:
                    result.pcs = other.pcs

        return result

    def __mul__(self, other):
        return S(*[self]) * other

    def __or__(self, other):
        return S(self.copy(), Atom(bar = True), other)

    def __pos__(self):
        if not hasattr(self, "relpitch"):
            raise Exception("not a relative pitch")
        copy = self.copy()
        copy.relpitch += 12
        return copy

    def __neg__(self):
        if not hasattr(self, "relpitch"):
            raise Exception("not a relative pitch")
        copy = self.copy()
        copy.relpitch -= 12
        return copy

    def __gt__(self, other):
        normalize_contour(other)
        result = self / Atom(vcs = [other])
        return result

    def __matmul__(self, other):
        normalize_contour(other)
        result = self / Atom(pcs = [other])
        return result


#
# pass 1: Python evaluation. Proceeds bottom up, so we can't do any
# left-to-right processing like assigning times or propagating
# defaults, so we just construct a tree structure.
#
# pass 2: traversal. Proceeds left-to-right, flattening tree
# structure, assigning times and propagating defaults, emitting
# notes.
#
# pass 3: rendering. Having assigned times we now know how long the
# piece is, so we can allocate a buf and overlay all the notes.
#

# base for P, S, and R
class Items:

    main = None

    def process_last():
        if Items.main:
            if args.print:
                print(Items.main.to_str(0))
            if args.save:
                Items.main.write()
            if args.play or args.loop:
                while True:
                    Items.main.play()
                    if not args.loop:
                        break

    def __init__(self, *items):

        self.items = list(items)
        self.clip = None
        self.instance = None
        self.breaking = True

        # auto-play top level
        if Items.main == None:
            atexit.register(Items.process_last)
        Items.main = self

    def append(self, item):
        self.items.append(item)

    def __getitem__(self, i):
        return self.items[i]

    def __setitem__(self, i, v):
        self.items[i] = v

    def __len__(self):
        return len(self.items)

    def copy(self, instance = None):
        result = self.__class__(*self.items)
        result.instance = instance
        return result

    def to_str(self, level = -1):

        if level == 0:
            result = "import piece\n"
        else:
            result = ""

        if isinstance(self, P) and len(self) <= 4 and all(isinstance(i, Atom) for i in self.items):
            result += self.__class__.__name__ + "("
            result += ", ".join(i.to_str() for i in self.items)
            result += ")"
            self.breaking = False
        else:
            result += self.__class__.__name__ + "("
            breaking = True
            for item in self.items:
                if breaking:
                    result += "\n" + "  " * (level + 1)
                else:
                    result += " "
                item_str = item.to_str(level + 1)
                result += item_str
                result += ","
                breaking = item.breaking
                if hasattr(item, "measure"):
                    result += " # " + str(item.measure)
                    breaking = True
            result += "\n" + "  " * level + ")"

        if hasattr(self, "repeat"):
            result += " * " + str(self.repeat)

        return result

    # xxx not sure about this notation
    def __or__(self, other):
        return S(self, Atom(bar = True), other)

    # repeat other times
    def __mul__(self, other):
        #assert isinstance(other, int)
        result = self.copy()
        result.repeat = other
        return result

    # traverse tree structure flattening into result
    # fully instantiate atoms with dur_secs, pitch, instrument,
    # using defaults to maintain and provide defaults for items
    def traverse(self, defaults, t_secs, t_bars, result, indent=""):

        # handle repeats as if enclosed in an S containing multiple instances of self
        if hasattr(self, "repeat"):
            repeated = S(*(self.copy(instance = i) for i in range(1, self.repeat+1)))
            repeated.traverse(defaults, t_secs, t_bars, result, indent)
            self.dur_secs = repeated.dur_secs
            self.dur_bars = repeated.dur_bars
            return

        self.dur_secs = 0
        self.dur_bars = 0

        # save defaults if we're an R
        if isinstance(self, R):
            restore = defaults.__dict__.copy()

        for item in self.items:

            if isinstance(item, Atom):

                # copy b/c we update below
                item = item.copy()

                # start bar count
                if hasattr(item, "bar"):
                    if item.bar and t_bars == None:
                        t_bars = 0
                    item.exclude.add("bar")

                # record timing
                item.t_secs = t_secs
                item.t_bars = t_bars
                if hasattr(item, "dbg") and not hasattr(item.dbg, "t_bars"):
                    item.dbg.t_bars = t_bars

                # bar check
                if hasattr(item, "bar") and item.bar:
                    dprint(f"--- bar {float(t_bars):.2f} t {t_secs:.2f}")
                    if (t_bars % 1 != 0):
                        print(item.loc(), "error: bar check fails")
                        os._exit(-1)

                # if we're a note item
                note_attrs = ("pitch", "relpitch", "dur_units")
                if any(hasattr(item, attr) for attr in note_attrs):

                    # merge vcs - xxx similar code in item.copy - factor it out?
                    # xxx can we use item.copy here??
                    if hasattr(item, "vcs") and hasattr(defaults, "vcs"):
                        item.vcs = defaults.vcs + item.vcs
                    if hasattr(item, "pcs") and hasattr(defaults, "pcs"):
                        item.pcs = defaults.pcs + item.pcs

                    # apply defaults
                    item.__dict__ = defaults.__dict__ | item.__dict__

                    # time jitter
                    item.t_secs += item.jitter("t_secs")                    

                    # compute durs
                    item.dur_secs = to_secs(item.dur_units, item.tempo)
                    item.dur_bars = to_bars(item.dur_units, item.time)

                    # compute pitch
                    pitch_dbg = []
                    if hasattr(item, "relpitch"):
                        oct = round((item.pitch - item.relpitch%12) / 12) * 12
                        pitch_dbg.append(str(oct))
                        item.pitch = oct + item.relpitch
                        pitch_dbg.append(str(item.relpitch))
                        del item.relpitch
                    else:
                        pitch_dbg.append(str(item.pitch))

                    # compute item_vcs
                    vol_dbg = [str(item.vol)]
                    if len(item.vcs):
                        item.item_vcs = []
                        item.exclude.add("item_vcs")
                        for i, vc in enumerate(item.vcs):
                            if callable(vc):
                                vc = vc(item)
                            vc, item_vc, vc_dbg = item.item_contour(vc)
                            item.item_vcs.append(item_vc)
                            if len(vc):
                                item.vcs[i] = vc
                            else:
                                del item.vcs[i] # xxx when did we copy this?
                            vol_dbg.append(vc_dbg)
                    vol_dbg = "+".join(vol_dbg)

                    # compute item_pcs
                    if len(item.pcs):
                        item.item_pcs = []
                        item.exclude.add("item_pcs")
                        for i, pc in enumerate(item.pcs):
                            if callable(pc):
                                pc = pc(item)
                            pc, item_pc, pc_dbg = item.item_contour(pc)
                            item.item_pcs.append(item_pc)
                            if len(pc):
                                item.pcs[i] = pc
                            else:
                                del item.pcs[i] # xxx when did we copy this?
                            pitch_dbg.append(pc_dbg)
                    pitch_dbg = "+".join(pitch_dbg)
                    
                    # distinguish between numeric pitch, and string instruction
                    if isinstance(item.pitch, str):

                        # pause and hold are extra-temporal (delay the beat)
                        # pause leaves silence, hold extends previous note
                        if item.pitch in ("pause", "hold"):
                            item.dur_bars = 0

                        # tie and hold extend the previous note
                        # hold is extra-temporal
                        if item.pitch in ("tie", "hold"):
                            for atom in reversed(result):
                                if hasattr(atom, "pitch"):
                                    atom.dur_secs += item.dur_secs
                                    atom.dur_bars += item.dur_bars
                                    break
                            if isinstance(self, P):
                                self.dur_secs = max(self.dur_secs, result[-1].dur_secs)
                                self.dur_bars = max(self.dur_bars, result[-1].dur_bars)

                        # only save numeric pitches so that relpitch works
                        del item.pitch

                    else:

                        # emit instantiated atom with a numeric pitch
                        result.append(item)

                    # debug note item
                    dprint(
                        f"{indent}note {item.instrument:10s} "
                        f"pitch:{pitch_dbg:10s} "
                        f"vol:{vol_dbg:10s} "
                        f"units:{float(item.dur_units):5.3f} secs:{item.dur_secs:5.2f} "
                        f"t:{item.t_secs:5.2f} "
                    )
            
                else:
    
                    # we're a meta item
                    d = item.__dict__.copy()
                    for a in ("exclude", "t_secs", "dur_secs", "dur_units", "dbg"):
                        if a in d:
                            del d[a]
                    dprint(f"{indent}meta", *[f"{n}:{v}" for n, v in d.items()])
                    item.dur_secs = 0
                    item.dur_bars = 0

                    # append marker for bar items
                    if hasattr(item, "bar") and item.bar:
                        result.append(item)

                # save defaults, excluding attributes marked exclude
                for attr in item.__dict__:
                    if not attr in item.exclude:
                        if (attr=="bar"): dprint("propagating bar")
                        defaults.__dict__[attr] = item.__dict__[attr]

            elif isinstance(item, (P, S)):

                # recursively traverse a P or S
                # item.ending are either None or a number 1, 2, ...
                if item.ending == None or item.ending == self.instance:
                    item.traverse(defaults, t_secs, t_bars, result, indent+"")
                else:
                    item.dur_secs = 0
                    item.dur_bars = 0

            else:
                raise Exception(f"bad item {item}")

            # assign ourselves a time depending on whether we're P or S
            if isinstance(self, S):
                self.dur_secs += item.dur_secs
                self.dur_bars += item.dur_bars
                t_secs += item.dur_secs
                if t_bars != None: t_bars += item.dur_bars
            elif isinstance(self, P):
                self.dur_secs = max(self.dur_secs, item.dur_secs)
                self.dur_bars = max(self.dur_bars, item.dur_bars)

        # restore defaults if we're an R
        if isinstance(self, R):
            defaults.__dict__.update(restore)

    def render(self):
    
        # already done?
        if self.clip:
            return self.clip

        # timing
        start_time = sys_time.time()

        # half at start, half at end
        pad = 1
    
        # traverse our args as a seqence, accumulating atoms
        # and assigning them start times
        defaults = Atom(
            tempo = (4, 120),
            time = (4, 4),
            transpose = 0,
            instrument = "sin",
            vol = 50,
            vcs = [],
            pcs = [],
            pitch = c4.pitch, # middle c
            dur_units = Units(1, 4),
        )
        atoms = []
        self.traverse(defaults.copy(), pad/2, None, atoms)

        # xxx assumes 441000
        t2i = engine.Clip().t2i

        # compute clips
        for atom in atoms:

            dprint(f"rendering {atom.t_secs:.2f}s")
            sys.stdout.flush()

            # bypass if marker, e.g. bar atom
            if not hasattr(atom, "pitch"):
                continue

            # compute freq
            pitch = atom.pitch + atom.transpose
            if hasattr(atom, "item_pcs"):
                for item_pc in atom.item_pcs:
                    pitch += atom.compute_contour(t2i, item_pc)
            freq = engine.p2f(pitch - a4.pitch) * 440

            # compute vol
            vol = atom.vol
            if hasattr(atom, "item_vcs"):
                for item_vc in atom.item_vcs:
                    vol += atom.compute_contour(t2i, item_vc)

            # compute clip
            instrument = engine.instruments[atom.instrument]
            ring = hasattr(atom, "ring") and atom.ring 
            dur_secs = atom.dur_secs if not ring else None
            atom.clip = instrument.get_clip(freq, vol, dur_secs)

        # compute end
        end = max(atom.t_secs + atom.clip.dur for atom in atoms if hasattr(atom, "clip")) + pad/2
    
        # overlay all clips
        self.clip = engine.Clip().zeros(dur=end)
        self.syncpoints = {}
        for atom in atoms:
            if hasattr(atom, "clip"):
                i = self.clip.t2i(atom.t_secs)
                self.clip.buf[i : i+len(atom.clip.buf)] += atom.clip.buf
            if hasattr(atom, "bar") and atom.bar == True:
                self.syncpoints[round(atom.t_bars)] = atom.t_secs # xxx check for conflict
            
        # apply --bars
        if args.bars:
            pruned = engine.Clip() # xxx sample rate
            for spec in args.bars.split(","):
                spec = spec.split("-")
                start = int(spec[0])
                end = start if len(spec) == 1 else int(spec[1])
                start = self.syncpoints[start-1]
                end = self.syncpoints[end]
                pruned |= self.clip.sliced(start, end)
            self.clip = pruned
            self.syncpoints = None # no longer valid

        print(f"rendering time {sys_time.time()-start_time:.2f}s")
        print(f"clip duration {self.clip.dur:.3f}s")
        return self.clip

    def play(self):
        self.render()
        self.clip.play()
        return self.clip

    def write(self, f=None):

        # render if not already done
        clip = self.render()

        # write clip
        if f == None:
            f = sys.argv[0].replace(".py", ".mp3")
        print("writing", f)
        clip.write(f)

        # write syncpoints
        if self.syncpoints:
            f = f.replace(".mp3", ".sync")
            print("writing", f)
            syncpoints = [list(item) for item in sorted(self.syncpoints.items())]
            open(f, "w").write(str(syncpoints))

        return clip


# a P is set of items played in parallel
class P(Items):
    ending = None

# an S is set of items played in sequence
class S(Items):
    ending = None

# an R is a sequence that saves its defaults at the beginning
# and restores it at the end
class R(S): pass

#
# environment
#

def std_endings():
    for e in range(1, 10):
        name = "E" + str(e)
        class _(S):
            ending = e
        _.__name__ = name
        _.__qualname__ = name
        builtins.__dict__[name] = _

# make instruments from lib available
def std_instruments():
    b = builtins.__dict__
    for instrument in engine.instruments:
        b[instrument] = Atom(instrument = instrument)

#
# absolute pitches af0 through gs7, and relative pitches a through g
#
pitch2str = {"hold": "h", "pause": "p"}
relpitch2str = {}
def std_tuning():
    b = builtins.__dict__
    for sfx, off in [("f", -1), ("s", 1), ("", 0)]:
        for x, p in zip("cdefgab", [0,2,4,5,7,9,11]):
            b[x+sfx] = Atom(relpitch = p + off)
            relpitch2str[p+off] = x + sfx
            relpitch2str[p+off+12] = "+" + x + sfx
            relpitch2str[p+off-12] = "-" + x + sfx
            for i in range(8):
                pitch = i*12 + p + off
                name = x+sfx+str(i)
                b[name] = Atom(pitch = pitch)
                pitch2str[pitch] = name

def std_vol():
    b = builtins.__dict__
    for i in range(100):
        b["v"+str(i)] = Atom(vol = i)

# first arg is subdivision (e.g. 4 for quarter), second arg is bpm (e.g. 60 for 60 bpm)
def tempo(num, den):
    if isinstance(num, tuple):
        num = 1 / to_units(num)
    return Atom(tempo = (num, den))

def time(num, den):
    return Atom(time = (num, den))

def transpose(transpose):
    return Atom(transpose = transpose)

def jitter(t_secs = None):
    random.seed(0) # for predictable testing
    atom = Atom(jitter_t_secs = t_secs)
    return atom

#
#
#

class Dbg:
    sub = None

def std_defs():
    builtins.P = P
    builtins.S = S
    builtins.R = R
    builtins.I = Atom(bar = True)
    builtins.tempo = tempo
    builtins.time = time
    builtins.transpose = transpose
    builtins.jitter = jitter
    builtins.r = Atom(pitch = "rest")
    builtins.t = Atom(pitch = "tie")
    builtins.p = Atom(pitch = "pause")
    builtins.h = Atom(pitch = "hold")
    builtins._ = Atom()
    builtins.ring = Atom(ring = True)
    builtins.stop = Atom(ring = False)
    std_endings()
    std_tuning()
    std_vol()
    std_instruments()

std_defs()
