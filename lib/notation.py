import engine
import builtins
import numpy as np
import inspect
import math
import random
import sys

class Atom:

    def __init__(self, **parms):
        self.exclude = {"exclude"}
        self.__dict__.update(parms)

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
            bars, beats = self.bars2beats(self.t_bars - self.dbg.t_bars)
            loc = f"{self.dbg.fn}:{self.dbg.line}"
            if self.dbg.col is not None:
                loc += f":{self.dbg.col}"
            loc += ": bar {bars+1} beat {beats+1}:"
            return loc
        else:
            return ""

    #
    # 3 measures of time:
    #
    # secs - real time as played
    #
    # units - fractions of a whole note. Relationship between units and
    # secs is determined by item.tempo, and may be altered by pauses and holds
    #
    # bars - fractions of a bar. Relationship between units and bars is
    # determined by item.time
    #
    # beats - subdivision of a bar, determined by numerator of time signature
    #

    def units2secs(self, units):
        num, den = self.tempo
        return units * num / den * 60
        
    def units2bars(self, units):
        num, den = self.time
        return units * den / num

    def bars2beats(self, bars):
        num, den = self.time
        return math.floor(bars), round((bars % 1) * num)

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
            have_dur = contour[0][0]
            dv = contour[0][1:]
            if abs(want_dur - have_dur) < 1e-6:
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

        # compute segments
        segments = []
        for segment in item_contour:
            if len(segment) == 2:
                dur_units, dv_start = segment
                dv_end = dv_start
            elif len(segment) == 3:
                dur_units, dv_start, dv_end = segment
            dur_secs = self.units2secs(dur_units)
            n = t2i(dur_secs)
            segments.append(np.interp(range(n), [0,n], [dv_start, dv_end]))

        # extend if needed, e.g. for ringing notes
        # xxx for now only for volume b/c requires clip to have already been computed to know len
        if hasattr(self, "clip"):
            shortfall = len(self.clip) - sum(len(segment) for segment in segments)
            if shortfall > 0:
                segments.append(np.full(shortfall, dv_end))

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

            # special cases for /n durations
            if isinstance(other, (float, int)):
                other = Atom(dur_units = 1 / other)
            elif isinstance(other, tuple):
                other = Atom(dur_units = sum(1/o for o in other))

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
        return S(*[self]*other)

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
        result = self / Atom(vcs = [other])
        return result

    def __matmul__(self, other):
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

    def __init__(self, *items):
        self.items = items

    def __or__(self, other):
        return S(self, Atom(bar = True), other)

    # traverse tree structure flattening into result
    # fully instantiate atoms with dur_secs, pitch, instrument,
    # using defaults to maintain and provide defaults for items
    def traverse(self, defaults, t_secs, t_bars, result, indent=""):

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
                    print(f"--- bar {t_bars:.2f} t {t_secs:.2f}")
                    if (abs(t_bars%1) > 1e-6):
                        print(item.loc(), "error: bar check fails")
                        exit(-1)

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

                    # compute durations
                    item.dur_secs = item.units2secs(item.dur_units)
                    item.dur_bars = item.units2bars(item.dur_units)

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
                    vol_dbg = [str(item.volume)]
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
                            result[-1].dur_secs += item.dur_secs
                            result[-1].dur_bars += item.dur_bars
                            if isinstance(self, P):
                                self.dur_secs = max(self.dur_secs, result[-1].dur_secs)
                                self.dur_bars = max(self.dur_bars, result[-1].dur_bars)

                        # only save numeric pitches so that relpitch works
                        del item.pitch

                    else:
                        # emit instantiated atom with a numeric pitch
                        result.append(item)

                    # debug note item
                    print(
                        f"{indent}note {item.instrument:10s} "
                        f"pitch:{pitch_dbg:10s} "
                        f"vol:{vol_dbg:10s} "
                        f"units:{item.dur_units:5.3f} secs:{item.dur_secs:5.2f} "
                        f"t:{item.t_secs:5.2f} "
                    )
            
                else:
    
                    # we're a meta item
                    d = item.__dict__.copy()
                    for a in ("exclude", "t_secs", "dur_secs", "t_units", "dur_units", "dbg"):
                        if a in d:
                            del d[a]
                    print(f"{indent}meta", *[f"{n}:{v}" for n, v in d.items()])
                    item.dur_secs = 0
                    item.dur_bars = 0

                # save defaults, excluding attributes marked exclude
                for attr in item.__dict__:
                    if not attr in item.exclude:
                        if (attr=="bar"): print("propagating bar")
                        defaults.__dict__[attr] = item.__dict__[attr]

            elif isinstance(item, (P, S)):

                # recursively traverse a P or S
                item.traverse(defaults, t_secs, t_bars, result, indent+"")

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
    
        # half at start, half at end
        pad = 1
    
        # traverse our args as a seqence, accumulating atoms
        # and assigning them start times
        defaults = Atom(
            tempo = (4,120),
            time = (4,4),
            transpose = 0,
            instrument = "sin",
            volume = 50,
            vcs = [],
            pcs = [],
            pitch = c4.pitch, # middle c
            dur_units = 1/4
        )
        atoms = []
        self.traverse(defaults.copy(), pad/2, None, atoms)

        # xxx assumes 441000
        t2i = engine.Clip().t2i

        # compute clips
        for atom in atoms:

            print(f"rendering {atom.t_secs:.2f}s")
            sys.stdout.flush()

            # compute freq
            pitch = atom.pitch + atom.transpose
            if hasattr(atom, "item_pcs"):
                for item_pc in atom.item_pcs:
                    pitch += atom.compute_contour(t2i, item_pc)
            freq = engine.p2f(pitch - a4.pitch) * 440

            # compute clip
            instrument = engine.synth_lib(atom.instrument)
            ring = hasattr(atom, "ring") and atom.ring 
            dur_secs = atom.dur_secs if not ring else None
            atom.clip = instrument(freq, dur_secs)

            # compute and apply volume
            volume = atom.volume
            if hasattr(atom, "item_vcs"):
                for item_vc in atom.item_vcs:
                    volume += atom.compute_contour(t2i, item_vc)
            m = 100 ** (volume / 100) # v0->1, v50->10, v100->100
            atom.clip.buf *= m

        # compute end
        end = max(atom.t_secs + atom.clip.duration for atom in atoms) + pad/2
    
        # overlay all clips
        clip = engine.Clip().zeros(duration=end)
        for atom in atoms:
            i = clip.t2i(atom.t_secs)
            clip.buf[i : i+len(atom.clip.buf)] += atom.clip.buf
            
        return clip

    def play(self):
        clip = self.render()
        clip.play()
        return clip

    def write(self, f=None):
        if f == None:
            #f = sys.argv[0].replace(".py", ".ogg")
            f = sys.argv[0].replace(".py", ".mp3")
        clip = self.render()
        print("writing", f)
        clip.write(f)
        return clip

    # repeat other times
    def __mul__(self, other):
        #assert isinstance(other, int)
        return S(*[self]*other)

# a P is set of items played in parallel
class P(Items): pass

# an S is set of items played in sequence
class S(Items): pass

# an R is a sequence that saves its defaults at the beginning
# and restores it at the end
class R(S): pass


#
# environment
#

# make instruments from lib available
def std_instruments(lib):
    b = builtins.__dict__
    for instrument in lib.instruments:
        b[instrument] = Atom(instrument = instrument, lib = lib)

#
# absolute pitches af0 through gs7, and relative pitches a through g
#
def std_tuning():
    b = builtins.__dict__
    for x, p in zip("cdefgab", [0,2,4,5,7,9,11]):
        b[x] = Atom(relpitch = p)
        b[x+"s"] = Atom(relpitch = p + 1)
        b[x+"f"] = Atom(relpitch = p - 1)
        for i in range(8):
            b[x+str(i)] = Atom(pitch = i*12 + p)
            b[x+"s"+str(i)] = Atom(pitch =i*12 + p + 1)
            b[x+"f"+str(i)] = Atom(pitch = i*12 + p - 1)

def std_volume():
    b = builtins.__dict__
    for i in range(100):
        b["v"+str(i)] = Atom(volume = i)

# first arg is subdivision (e.g. 4 for quarter), second arg is bpm (e.g. 60 for 60 bpm)
def tempo(num, den):
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
    std_tuning()
    std_volume()
    std_instruments(engine.synth_lib)

std_defs()
