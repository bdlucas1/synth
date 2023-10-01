import numpy as np
import simpleaudio as sa
import soundfile as sf
import pydub
import sys
import matplotlib.pyplot as plt
import math
import atexit
import simpleaudio as sa  # pip3 install simpleaudio
import time
import collections
import os

#
# utility
#

# xxx subsume into Dbg?
def subplots(*n, fig=None, fn="/tmp/fig.png", interactive = False, figsize=None):
    if interactive:
        plt.ion()
    fig, axs = plt.subplots(*n)
    fig.set_tight_layout(True)
    if figsize != None:
        fig.set_size_inches(*figsize)
    def exit():
        if interactive:
            plt.show()
        else:
            plt.savefig(fn)
    atexit.register(exit)
    return axs

class Dbg:

    def __init__(self):
        self.all_axs = np.transpose(subplots(6, 3, figsize=(9,8)))
        self.used = [0, 0, 0]

    def axs(self, n, title):
        for col in range(len(self.used)):
            if self.used[col] + n <= len(self.all_axs[col]):
                if isinstance(title, str):
                    title = [title] * n
                axs = self.all_axs[col][self.used[col] : self.used[col]+n]
                for ax, title in zip(axs, title):
                    ax.text(.98, .95, title,
                            horizontalalignment="right", verticalalignment="top",
                            transform=ax.transAxes)
                self.used[col] += n
                return axs if n > 1 else axs[0]

dbg = Dbg()

def play(buf, sample_rate):
    buf = buf * (2**15 - 1) / np.max(np.abs(buf))
    buf = buf.astype(np.int16)
    playing = sa.play_buffer(buf, 1, 2, sample_rate)
    playing.wait_done()
    time.sleep(1) # avoid other exit activity until clip is really done

def eq_ratio(k):
    return 2**(k/12)

def plot_eq(ax, lo, hi, y=1):
    for k in range(100):
        f = lo * eq_ratio(k)
        if f > hi:
            break
        ax.plot([f, f], [0, y], color="#F0F0F0")

def p2f(p, tet=12):
    return 2 ** (p / tet)

def f2p(f, tet=12):
    return math.log2(f) * tet


#
# Clip
#

class Clip:

    def __init__(self, dbg=None):
        self.dbg = dbg
        self.zeros()

    def zeros(self, dur=0, sample_rate=44100):
        self.sample_rate = sample_rate
        n = self.t2i(dur)
        self._buf = np.zeros(n)
        return self

    def read(self, f, **kwargs):
        self._buf, self.sample_rate = sf.read(f, **kwargs)
        # always even b/c spectrum is half size of buf
        if (len(self)) % 2 == 1:
            self._buf = self._buf[:-1]
        return self

    def write(self, f, fp=False, **kwargs):
        if fp:
            sf.write(f, self.buf, self.sample_rate, subtype="FLOAT")
        elif False: #f.endswith(".flac"):
            # ok for flac
            # mp3 differs for testing purposes, but maybe is ok - switch over?
            buf = self.buf * (2**15 - 1) / np.max(np.abs(self.buf))
            buf = buf.astype(np.int16)
            sf.write(f, buf, self.sample_rate)
        else:
            buf = self.buf * (2**15 - 1) / np.max(np.abs(self.buf))
            buf = buf.astype(np.int16)
            fmt = f.split(".")[-1]
            seg = pydub.AudioSegment(
                buf.tobytes(),
                frame_rate = self.sample_rate,
                sample_width = 2,
                channels = 1
            )
            seg.export(f, format = fmt, **kwargs)
        return self

    # xxx don't zero? caller instead does Clip().like(other).zeros()?
    def like(self, other):
        self.sample_rate = other.sample_rate
        self._buf = np.zeros(len(other.buf))
        self._spectrum = np.zeros(len(other.spectrum), dtype=np.cfloat)
        return self

    def from_buf(self, buf):
        self.buf = buf
        return self

    def copy(self, other):
        self.sample_rate = other.sample_rate
        self.buf = np.copy(other.buf)
        return self

    def play(self):
        print("playing", len(self.buf), self.buf)
        play(self.buf, self.sample_rate)
        return self

    def padded(self, start, end):
        result = Clip().like(self).zeros(start) | self | Clip().like(self).zeros(end) 
        return result

    def trimmed(self, start, end):
        result = Clip().like(self)
        result.buf = self.buf[self.t2i(start) : -self.t2i(end)]
        return result

    @property
    def fs(self):
        return np.linspace(0, self.sample_rate / 2, len(self.spectrum))

    @property
    def ts(self):
        return np.linspace(0, self.dur, len(self))

    @property
    def dur(self):
        return len(self.buf) / self.sample_rate

    def __len__(self):
        return len(self.buf)

    def _ft(self):
        #print("_ft")
        self._spectrum = np.fft.rfft(self.buf)

    def _ift(self):
        #print("_ift")
        self.buf = np.fft.irfft(self.spectrum)

    @property
    def spectrum(self):
        #print("get spectrum")
        if not hasattr(self, "_spectrum"):
            self._ft()
        return self._spectrum

    @spectrum.setter
    def spectrum(self, spectrum):
        #print("set spectrum")
        self._spectrum = spectrum
        del self._buf

    @property
    def buf(self):
        #print("get buf")
        if not hasattr(self, "_buf"):
            if hasattr(self, "_spectrum"):
                self._ift()
            else:
                raise "either buf nor spectrum are set"
        return self._buf

    @buf.setter
    def buf(self, buf):
        #print("set buf")
        self._buf = buf
        if hasattr(self, "_spectrum"): del self._spectrum
        if hasattr(self, "_fs"): del self._fs

    def i2f(self, i):
        return i * self.sample_rate / 2 / len(self.spectrum)

    def f2i(self, f):
        return round(f / (self.sample_rate/2) * len(self.spectrum))

    def t2i(self, t):
        # do everything at even boundaries so buffer size is always even b/c spectrum is half buf
        return round(t * self.sample_rate / 2) * 2

    def i2t(self, i):
        return i / self.sample_rate

    def normalized(self):
        clip = Clip().copy(self)
        clip.buf /= max(abs(clip.buf))
        return clip

    # softness is more-or-less total width of cutoff
    def filter(self, lo = -math.inf, hi = math.inf, softness = 1e-100):
        softness /= 4
        hi = np.array([(1-math.erf((x-hi)/softness))/2 for x in self.fs])
        lo = np.array([(1+math.erf((x-lo)/softness))/2 for x in self.fs])
        filter = Clip().like(self)
        filter.spectrum = lo * hi
        return filter

    def filtered(self, filter):
        clip = Clip().like(self)
        clip.spectrum = self.spectrum * filter.spectrum
        return clip

    def copy_spectrum(self, fself, fother, other, width=100):
        a = other.f2i(fother-width/2)
        b = other.f2i(fother+width/2)
        c = self.f2i(fself-width/2)
        d = c + (b - a) # ensure same width
        self.spectrum[c:d] = other.spectrum[a:b]

    def warp_spectrum(self, other, f, ratio, eq=False, width=100):
        for i in range(1, 30):
            fother = f*i
            fself = fother * ratio
            if eq:
                kf = 12 * math.log2(fself / f)
                k = round(kf)
                if True: #abs(k-kf) < 0.2:
                    fself = f * (2**(k/12))
            self.copy_spectrum(fself, fother, other, width)

    # xxx enveloped (i.e. make new clip)
    def apply_envelope(self, envelope):
        self.buf *= envelope

    def interp_envelope(self, ts, ms):
        return np.interp([i/self.sample_rate for i in range(len(self.buf))], ts, ms)

    def get_envelope(self, cutoff=50, softness=10, normalize=True):

        # envelope is a time series with a sample_rate
        envelope = Clip().like(self)

        npad = self.t2i(0.2)
        pad = np.zeros(npad)

        # xxx used padded and trimmed
        # xxx magnitude is diff from buf???
        power = Clip().like(self)
        power.buf = abs(np.concatenate([pad, self.buf, pad])) ** 2
        filter = power.filter(hi=cutoff, softness=softness)
        power = power.filtered(filter)
        envelope.buf = abs(power.buf) ** 0.5 * math.sqrt(2)
        envelope.buf = envelope.buf[npad : -npad]
        if normalize:
            envelope.buf /= max(envelope.buf)
        
        #if dbg.dbg:
        #    dbg.axs_envelope[0].plot(self.ts, abs(self.buf))
        #    filter.plot_spectrum(dbg.axs_envelope[1])
        #    dbg.axs_envelope[2].plot(self.ts, envelope.buf)

        return envelope

    def get_fundamental(self):
        fs, _ = self.get_harmonics(nharmonics=1)
        return fs[0]

    def get_harmonics(self, pow=2, fundamental=None, nharmonics=20):

        # make copy for finding harmonics
        work = Clip().copy(self)
        n = len(work.buf)

        # apply guassian window signal
        # equivalent to convolving spectrum to smooth it
        window = [math.exp(- (((i - n/2) / (n/5)) ** 2)) for i in range(n)]
        work.buf *= window
        #ax1.set_xlim(0, n)
        #ax1.plot(m)

        spectrum = abs(work.spectrum)
        spectrum /= max(spectrum)

        # approximate frequency of fundamental
        # used to generate buckets for computing accurate frequency of harmonics
        if not fundamental:
            for i in range(1, len(spectrum)-1):
                if spectrum[i]>0.1 and spectrum[i]>spectrum[i-1] and spectrum[i]>spectrum[i+1]:
                    fundamental = self.i2f(i)
                    print(f"approximate fundamental {fundamental:.1f}")
                    break
        
        # divide spectrum into buckets based on multiples of fundamental
        # compute energy in each bucket, and weighted average frequency of bucket
        harmonics = np.zeros(nharmonics)
        fs = np.zeros(nharmonics)
        for i in range(1, len(spectrum)-1):
            f = self.i2f(i)
            k = round(f / fundamental)
            if 0 < k and k <= len(harmonics):
                energy = spectrum[i] ** pow
                harmonics[k-1] += energy
                fs[k-1] += energy * f

        # divide by sum of weights
        fs /= harmonics
                    
        # compute normalized amplitude from energy
        harmonics = harmonics ** (1/pow)
        harmonics /= max(harmonics)

        if self.dbg:

            def plot(spectrum, ax1, ax2, ax3):

                spectrum = abs(spectrum)
                spectrum = spectrum / max(spectrum)

                ax1.set_xlim(0, 4000)
                ax1.plot(self.fs, spectrum)
                ax1.scatter(fs, harmonics, edgecolors="red", facecolors="none")

                def closeup(ax, f):
                    ax.set_xlim(0.9*f, 1.1*f)
                    ax.plot(self.fs, spectrum)
                    ax.scatter(fs, harmonics, edgecolors="red", facecolors="none")

                closeup(ax2, fs[1])
                closeup(ax3, fs[3])

            plot(self.spectrum, *self.dbg.axs_harmonics[0:3])
            plot(work.spectrum, *self.dbg.axs_harmonics[3:6])

        return fs, harmonics


    def __add__(self, other):
        result = Clip().like(self)
        result.buf = self.buf + other.buf
        print(self.buf)
        print(other.buf)
        print(result.buf)
        return result

    def concatenate(clips):
        result = Clip().like(clips[0])
        result.buf = np.concatenate([clip.buf for clip in clips])
        return result

    def __or__(self, other):
        result = Clip()
        result.sample_rate = self.sample_rate
        #print(self.buf.dtype, other.buf.dtype, self.buf.shape, other.buf.shape)
        result.buf = np.concatenate([self.buf, other.buf])
        return result

    def plot_buf(self, ax):
        ax.plot(self.ts, abs(self.buf))

    def plot_spectrum(self, ax, mul=1, **kwargs):
        m = max(abs(self.spectrum)) * mul
        ax.set_xlim(0, 4000)
        ax.plot(self.fs, abs(self.spectrum) / m, **kwargs)

#
# Synth is a clip factory
#

class Envelope:
    def __init__(self, ts, vs):
        self.ts = ts
        self.vs = vs

class BaseSynth:

    def __init__(self, name, dbg=False):
        self.dbg = False # xxxx
        self.clips = {}
        register(name, self)

    # compute clip of given freqency (may be frequency contour) and dur
    def compute_clip(self, freq, dur):

        # xxx sample rate?
        clip = Clip()

        # compute harmonics from self.harmonics either directly or by resampling
        clip_dur, harmonics = self.get_harmonics(freq, dur, clip)

        # compute theta, whose derivative is instantaneous frequency
        n = clip.t2i(clip_dur)
        if isinstance(freq, np.ndarray):
            # variable frequency - "integrate" instantaneous frequency using cumsum
            if len(freq) < n:
                freq = np.concatenate([freq, np.full(n-len(freq), freq[-1])])
            omega = np.full(n, clip_dur/n) * freq
            theta = np.cumsum(omega)
        else:
            # optimized version for scalar frequency
            theta = np.linspace(0, clip_dur, n, False) * freq

        # combine harmonics
        thetas = [theta * (k+1) for k in range(len(harmonics))]
        #if ph: # pseudo-harmonics - needs fixing up after added support for pitch contours
        #    thetas = [theta * p2f(round(f2p(theta/thetas[0])))*thetas[0] for theta in thetas]
        clip.buf = sum(h * np.sin(2*np.pi*theta) for theta, h in zip(thetas, harmonics))

        # clip if non-elastic
        # this does nothing if dur > clip.dur.
        # xxx is this ok? should be if we add notes as events instead of concatenating them
        if not self.elastic:
            if dur < clip.dur:
                clip = clip.trimmed(0, clip.dur - dur)
            ease_out = clip.interp_envelope([0, dur-self.ease_out, dur], [1, 1, 0]) ** 2
            clip.apply_envelope(ease_out)

        # debugging plots
        if self.dbg:
            ax = dbg.axs(1, f"{self.name} {freq:.1f} Hz, {dur:.1f} s")
            clip.plot_buf(ax)
            clip.get_envelope(normalize=False).plot_buf(ax)

        return clip


    # get clip of given freq, vol, dur, using memo
    def get_clip(self, freq, vol, dur, ph=False):

        # used for "ring" for non-elastic instruments (e.g. guitar)
        if dur is None: dur = self.base_dur

        # consult memo
        num = (int,float)
        if isinstance(freq, num) and isinstance(vol, num) and (freq, vol, dur) in self.clips:
            return self.clips[(freq, vol, dur)]
        elif isinstance(freq, num) and (freq, dur) in self.clips:
            without_vol = self.clips[(freq, dur)]
        else:
            without_vol = self.compute_clip(freq, dur)

        # compute and apply vol
        mul = 100 ** (vol / 100) # v0->1, v50->10, v100->100
        if isinstance(mul, np.ndarray) and len(mul) < len(without_vol):
            mul = np.concatenate([mul, np.full(len(without_vol) - len(mul), mul[-1])])
        with_vol = Clip().copy(without_vol)
        with_vol.buf *= mul

        # remember
        #if isinstance(freq, num) and isinstance(vol, num): # makeing very little diff - working?
        #    self.clips[(freq, vol, dur)] = with_vol
        if isinstance(freq, num): # sor-op35-no1 is about 3x faster with this
            self.clips[(freq, dur)] = without_vol

        # here's our answer
        return with_vol


class HarmonicSynth(BaseSynth):

    def __init__(self, name, sample_name=None, harmonics=None, elastic=True, ease_out=0.1):

        super().__init__(name)

        self.name = name
        self.sample_name = sample_name
        self.elastic = elastic
        self.ease_out = ease_out

        # harmonics supplied, remember them
        if harmonics is not None:
            self.harmonics = []
            for h in harmonics:
                self.harmonics.append(Envelope([0,0.1,0.9,1], [0,h,h,0]))
            self.base_dur = 1
            self.elastic = True
            
    def save(self, name):
        instrument_cache = os.path.join(os.path.dirname(__file__), ".instruments")
        path = os.path.join(instrument_cache, f"{name}.npy")
        print("saving", path)
        if not os.path.exists(instrument_cache):
            os.mkdir(instrument_cache)
        np.save(path, np.array([self.__dict__]))

    def load(self, name):
        instrument_cache = os.path.join(os.path.dirname(__file__), ".instruments")
        path = os.path.join(instrument_cache, f"{name}.npy")
        if os.path.exists(path):
            print("loading", path)
            self.__dict__ = np.load(path, allow_pickle = True)[0]
            return True
        else:
            return False

    # xxx do the from_ things here and in Clip with subclasses??
    def realize(self):
        
        # already realized?
        if hasattr(self, "harmonics"):
            return

        # restorable?
        if self.load(self.name):
            return

        # read sample
        sample_dn = os.path.join(os.path.dirname(__file__), "..", "samples")
        sample_fn = os.path.join(sample_dn, f"{self.sample_name}.flac")
        sample = Clip().read(sample_fn)
        sample.buf /= max(abs(sample.buf))

        # base info
        # xxx allow fundamental to be supplied on instantiation?
        self.base_dur = sample.dur
        self.base_fundamental = sample.get_fundamental()
        print(f"computed fundamental {self.base_fundamental:.1f}")

        # show base sample if desired
        if self.dbg:
            ax = dbg.axs(1, f"{name} base sample")
            sample.plot_buf(ax)
            envelope = sample.get_envelope(normalize=False)
            envelope.plot_buf(ax)
            mx = max(envelope.buf)

        # padding avoids boundary artifacts (ringing?)
        padding = 0.2
        sample = sample.padded(padding, padding)

        self.harmonics = []
        for h in range(1, 20):
    
            # get frequencies in this band
            band = Clip().copy(sample)
            lo = (h-0.5) * self.base_fundamental
            hi = (h+0.5) * self.base_fundamental
            print(f"band {lo:.1f} {hi:.1f}")
            filter = band.filter(lo, hi)
            band = band.filtered(filter)
    
            # get envelope for this band
            # xxx explore these parameters more
            softness = 5
            cutoff = self.base_fundamental - softness
            envelope = band.get_envelope(normalize=False, cutoff=cutoff, softness=softness)
            envelope = envelope.trimmed(padding, padding)
            self.harmonics.append(Envelope(envelope.ts, envelope.buf))

            if self.dbg and h < 6:
                ax = dbg.axs(1, f"{fundamental*h:.1f} Hz band")
                ax.set_ylim(0, mx)
                band.trimmed(padding, padding).plot_buf(ax)
                envelope.plot_buf(ax)

        self.save(self.name)
        return self

    def get_harmonics(self, freq, dur, clip):
        
        # get harmonics from sample if not already supplied
        if not hasattr(self, "harmonics") and hasattr(self, "sample_name"):
            self.realize()

        # compute harmonics from self.harmonics either directly or by resampling
        if self.elastic:
            # resample band envelopes to desired dur
            d_in = self.base_dur
            d_out = dur
            # f maps output timestamps back to corresponding input timestamp
            # f(0) = 0         map output at start to input at start
            # f(d_out) = d_in  map output at end to input at end
            # f'(0) = 1        preserve envelope near beginning
            # f'(d_out) = 1    preserve envelope near end
            a = 2 * (d_out - d_in) / d_out**3
            b = 3 * (d_in - d_out) / d_out**2
            f = lambda x: a * x**3 + b * x**2 + x
            ts_out = np.linspace(0, d_out, clip.t2i(d_out))  # range of desired output timestamps
            ts_in = np.array([f(t) for t in ts_out])         # corresponding input timestamps via f
            harmonics = [np.interp(ts_in, env.ts, env.vs) for env in self.harmonics]
            clip_dur = dur
        else:
            # just copy self.harmonics
            harmonics = [env.vs for env in self.harmonics]
            clip_dur = self.base_dur

        return clip_dur, harmonics

class MultiFreqSynth(BaseSynth):

    def __init__(self, name, synths):
        super().__init__(name)
        self.name = name
        self.synths = synths

    def get_harmonics(self, freq, dur, clip):

        # xxx realizes all synths even if we only use one
        # xxx make this auto?
        for synth in self.synths:
            synth.realize()

        # parameters
        self.elastic = self.synths[0].elastic
        self.ease_out = sum(synth.ease_out for synth in self.synths) / len(self.synths)
        self.base_dur = max(synth.base_dur for synth in self.synths)
        self.harmonics = {} # map from (freq,dur) to harmonics

        # portamento - use avg freq
        if not isinstance(freq, (int, float)):
            freq = sum(freq) / len(freq)

        # already saw this one
        if (freq, dur) in self.harmonics:
            return self.harmonics[freq]

        # lo?
        if freq < self.synths[0].base_fundamental:
            result = self.synths[0].get_harmonics(freq, dur, clip)

        # hi?
        elif freq > self.synths[-1].base_fundamental:
            result = self.synths[-1].get_harmonics(freq, dur, clip)

        # interp
        else:
            for s1, s2 in zip(self.synths[:-1], self.synths[1:]):
                if freq >= s1.base_fundamental and freq <= s2.base_fundamental:
                    m2 = (freq - s1.base_fundamental) / (s2.base_fundamental - s1.base_fundamental)
                    h1_dur, h1 = s1.get_harmonics(freq, dur, clip)
                    h2_dur, h2 = s2.get_harmonics(freq, dur, clip)
                    clip_dur = max(h1_dur, h2_dur)
                    harmonics = []
                    for h1, h2 in zip(h1, h2):
                        if len(h1) < len(h2):
                            h1 = np.pad(h1, (0, len(h2) - len(h1)))
                        elif len(h2) < len(h1):
                            h2 = np.pad(h2, (0, len(h1) - len(h2)))
                        harmonics.append((1-m2)*h1 + m2*h2)
                    result = clip_dur, harmonics
                    break
            
        self.harmonics[(freq, dur)] = result
        return result

class MultiVolSynth(BaseSynth):

    def __init__(self, name, synths):
        super().__init__(name)
        self.name = name
        self.synths = synths

    def get_clip(self, freq, vol, dur, ph=False):
        
        scalar_vol = vol if isinstance(vol, (int,float)) else sum(vol) / len(vol)

        # lo?
        if scalar_vol <= self.synths[0][0]:
            result = self.synths[0][1].get_clip(freq, vol, dur)

        # hi?
        elif scalar_vol >= self.synths[-1][0]:
            result = self.synths[-1][1].get_clip(freq, vol, dur)

        # interp
        else:
            for s1, s2 in zip(self.synths[:-1], self.synths[1:]):
                if scalar_vol >= s1[0] and scalar_vol <= s2[0]:
                    clip1 = s1[1].get_clip(freq, vol, dur)
                    clip2 = s2[1].get_clip(freq, vol, dur)
                    if clip1.dur < clip2.dur:
                        clip1 = clip1.padded(0, clip2.dur - clip1.dur)
                    elif clip1.dur > clip2.dur:
                        clip2 = clip2.padded(0, clip1.dur - clip2.dur)
                    m2 = (vol - s1[0]) / (s2[0] - s1[0])
                    result = Clip().like(clip1)
                    result.buf = clip1.buf * (1-m2) + clip2.buf * m2

        return result

#
# instrument registry
#

instruments = {}

def register(name, instrument):
    instruments[name] = instrument
    globals()[name] = instrument

#
# define and register instruments
#

HarmonicSynth("sin", harmonics=[1])

#
# guitar
#

HarmonicSynth("guitar_a2_f", ease_out=0.1, elastic=False)
HarmonicSynth("guitar_a3_f", ease_out=0.1, elastic=False)
HarmonicSynth("guitar_a2_p", ease_out=0.1, elastic=False)
HarmonicSynth("guitar_a3_p", ease_out=0.1, elastic=False)

MultiFreqSynth("multi_guitar_f", synths=[guitar_a2_f, guitar_a3_f])
MultiFreqSynth("multi_guitar_p", synths=[guitar_a2_p, guitar_a3_p])

MultiVolSynth("multi_vol_guitar", synths=[(50,multi_guitar_p), (70,multi_guitar_f)])

#register("guitar", multi_guitar_p)
register("guitar", multi_vol_guitar)

#
# clarinet
#

HarmonicSynth("clarinet_a3_f", sample_name="clarinet_a3_f", elastic=True)
HarmonicSynth("clarinet_a5_f", sample_name="clarinet_a5_f", elastic=True)
HarmonicSynth("clarinet_a3_p", sample_name="clarinet_a3_p", elastic=True)
HarmonicSynth("clarinet_a5_p", sample_name="clarinet_a5_p", elastic=True)

MultiFreqSynth("multi_clarinet_f", synths=[clarinet_a3_f, clarinet_a5_f])
MultiFreqSynth("multi_clarinet_p", synths=[clarinet_a3_p, clarinet_a5_p])
        
MultiVolSynth("multi_vol_clarinet", synths=[(30,multi_clarinet_p), (70,multi_clarinet_f)])

#register("clarinet", clarinet_a3_f)
register("clarinet", multi_vol_clarinet)

#
# saxophone
#

HarmonicSynth("saxophone", sample_name="saxophone_a3", elastic=True)
