import numpy as np
import simpleaudio as sa
import soundfile as sf
import pydub
import sys
import matplotlib.pyplot as plt
import math
import atexit
import simpleaudio as sa  # pip3 install simpleaudio
import zipfile
import time
import collections
import os
#import pwlf # piecewise linear fit - exremely slow

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

    def zeros(self, duration=0, sample_rate=44100):
        self.sample_rate = sample_rate
        n = self.t2i(duration)
        self._buf = np.zeros(n)
        return self

    def read(self, f, **kwargs):
        self._buf, self.sample_rate = sf.read(f, **kwargs)
        # always even b/c spectrum is half size of buf
        if (len(self)) % 2 == 1:
            self._buf = self._buf[:-1]
        return self

    def write(self, f):
        buf = self.buf * (2**15 - 1) / np.max(np.abs(self.buf))
        buf = buf.astype(np.int16)

        fmt = f.split(".")[-1]
        if fmt == "ogg":
            # xxx this is just for testing compatibility with existing ref ogg files
            # maybe switch those to mp3?
            sf.write(f, buf, self.sample_rate)
        else:
            seg = pydub.AudioSegment(
                buf.tobytes(),
                frame_rate = self.sample_rate,
                sample_width = 2,
                channels = 1
            )
            seg.export(f, format = fmt) #, bitrate="80k")

        return self

    def lib(self, group, fn):
        zipfn = f"/Users/bdlucas1/Downloads/all-samples/{group}.zip"
        f = zipfile.ZipFile(zipfn, 'r').open(f"{fn}.mp3")
        return self.read(f)

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
        return np.linspace(0, self.duration, len(self))

    @property
    def duration(self):
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

        npad = self.t2i(0.1)
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

class Synth:

    def __init__(self, dbg=False):
        self.dbg = dbg

    # harmonics is list of 
    def from_harmonics(self, name, harmonics):
        self.harmonics = []
        for h in harmonics:
            self.harmonics.append(Envelope([0,0.1,0.9,1], [0,h,h,0]))
        self.base_duration = 1
        self.elastic = True
        return self

    # xxx do the from_ things here and in Clip with subclasses??
    def from_clip(self, name, clip, fundamental=None, elastic=True, ease_out=0.1):
        
        self.name = name
        self.elastic = elastic
        self.ease_out = ease_out

        # base info
        self.base_clip = clip
        self.base_duration = self.base_clip.duration
        if fundamental == None:
            fundamental = clip.get_fundamental()
            print(f"computed fundamental {fundamental:.1f}")
        self.base_fundamental = fundamental

        # show base clip if desired
        if self.dbg:
            ax = dbg.axs(1, f"{name} base clip")
            self.base_clip.plot_buf(ax)
            envelope = self.base_clip.get_envelope(normalize=False)
            envelope.plot_buf(ax)
            mx = max(envelope.buf)

        # padding avoids boundary artifacts (ringing?)
        padding = 0.1
        clip = clip.padded(padding, padding)

        self.harmonics = []
        for h in range(1, 20):
    
            # get frequencies in this band
            band = Clip().copy(clip)
            lo = (h-0.5) * fundamental
            hi = (h+0.5) * fundamental
            print(f"band {lo:.1f} {hi:.1f}")
            filter = band.filter(lo, hi)
            band = band.filtered(filter)
    
            # get envelope for this band
            # xxx explore these parameters more
            softness = 5
            cutoff = fundamental - softness
            envelope = band.get_envelope(normalize=False, cutoff=cutoff, softness=softness)
            envelope = envelope.trimmed(padding, padding)
            self.harmonics.append(Envelope(envelope.ts, envelope.buf))

            if self.dbg and h < 6:
                ax = dbg.axs(1, f"{fundamental*h:.1f} Hz band")
                ax.set_ylim(0, mx)
                band.trimmed(padding, padding).plot_buf(ax)
                envelope.plot_buf(ax)

        return self

    def __call__(self, freq=None, duration=None, ph=False):

        if freq is None: freq = self.base_fundamental
        if duration is None: duration = self.base_duration

        # xxx sample rate?
        clip = Clip()

        # compute harmonics from self.harmonics either directly or by resampling
        if self.elastic:
            # resample band envelopes to desired duration
            d_in = self.base_duration
            d_out = duration
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
            clip_dur = duration
        else:
            # just copy self.harmonics
            harmonics = [env.vs for env in self.harmonics]
            clip_dur = self.base_duration

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
        thetas = [theta * (k+1) for k in range(len(self.harmonics))]
        #if ph: # pseudo-harmonics - needs fixing up after added support for pitch contours
        #    thetas = [theta * p2f(round(f2p(theta/thetas[0])))*thetas[0] for theta in thetas]
        clip.buf = sum(h * np.sin(2*np.pi*theta) for theta, h in zip(thetas, harmonics))

        # clip if non-elastic
        # this does nothing if duration > clip.duration.
        # xxx is this ok? should be if we add notes as events instead of concatenating them
        if not self.elastic:
            if duration < clip.duration:
                clip = clip.trimmed(0, clip.duration - duration)
            ease_out = clip.interp_envelope([0, duration-self.ease_out, duration], [1, 1, 0]) ** 2
            clip.apply_envelope(ease_out)

        # debugging plots
        if self.dbg:
            ax = dbg.axs(1, f"{self.name} {freq:.1f} Hz, {duration:.1f} s")
            clip.plot_buf(ax)
            clip.get_envelope(normalize=False).plot_buf(ax)

        return clip



#
# samples
#

def kwargs(**kwargs): return kwargs

class Lib:

    def __init__(self, name):
        self.dbg = False
        self.loaded = {}
        self.name = name
            
    def __str__(self):
        return self.name

    def __call__(self, name):
        if not name in self.loaded:
            self.loaded[name] = self.load(name, **self.instruments[name])
        return self.loaded[name]


class ClipLib(Lib):

    instruments = {

        "guitar_a3": kwargs(
            dn = "guitar",
            fn = "guitar_A3_very-long_forte_normal",
            trim_start = 1.925,
            ease_start = 0,
            trim_end = 1,
            ease_end = 1
        ),

        "clarinet_a3": kwargs(
            dn = "clarinet",
            fn = "clarinet_A3_1_forte_normal",
            trim_start = 0,
            ease_start = 0,
            trim_end = 0,
            ease_end = 0.1,
        ),

        "saxophone_a3": kwargs(
            dn = "saxophone",
            fn = "saxophone_A3_15_forte_normal",
            trim_start = 0.035,
            ease_start = 0,
            trim_end = 0.05,
            ease_end = 0.05
        ),
    }            

    def load(self, name, dn, fn, trim_start= 0, trim_end= 0, ease_start= 0, ease_end= 0, lo=100):
            
        # get clip from library
        original_clip = Clip(self.dbg).lib(dn, fn)

        # add padding
        padding = 0.1
        clip = original_clip.padded(padding, padding)
        trim_start += padding
        trim_end += padding

        # high-pass filter
        hpfilter = clip.filter(lo=lo)
        clip = clip.filtered(hpfilter)

        # trim
        clip = clip.trimmed(trim_start, trim_end)

        # ease in/out
        # xxx make exponent a parameter?
        end = clip.duration
        ease = clip.interp_envelope([0, ease_start, end-ease_end, end], [0, 1, 1, 0]) ** 2
        clip.apply_envelope(ease)

        if self.dbg:
            ax_spectrum, ax_hpfilter, ax_clip_spectrum = dbg.axs(3, "load " + name)
            original_clip.plot_spectrum(ax_spectrum)
            hpfilter.plot_spectrum(ax_hpfilter)
            clip.plot_spectrum(ax_clip_spectrum)

            ax_envelope, ax_start, ax_end = dbg.axs(3, ["envelope", "start", "end"])

            envelope = clip.get_envelope(normalize=False)
            clip.plot_buf(ax_envelope)
            envelope.plot_buf(ax_envelope)

            ax_start.set_xlim(0, 0.1)
            clip.plot_buf(ax_start)
            envelope.plot_buf(ax_start)

            ax_end.set_xlim(clip.duration - 0.1, clip.duration)
            clip.plot_buf(ax_end)
            envelope.plot_buf(ax_end)

            #ax_absbuf, ax_envfilter, ax_envelope = dbg.axs(3)
            #clip.plot_buf(ax_absbuf)
            #filter.plot_spectrum(ax_envfilter)
            #envelope.plot_buf(ax_envelope)

        return clip

clip_lib = ClipLib("clip_lib")

class SynthLib(Lib):

    instruments = {

        "sin": kwargs(
            harmonics = [1]
        ),

        "guitar": kwargs(
            base_lib = clip_lib,
            base_name = "guitar_a3",
            elastic = False
        ),

        "clarinet": kwargs(
            base_lib = clip_lib,
            base_name = "clarinet_a3",
            elastic = True
        ),

        "saxophone": kwargs(
            base_lib = clip_lib,
            base_name = "saxophone_a3",
            elastic = True
        ),
    }

    def load(self, name, base_lib=None, base_name=None, harmonics=None, **kwargs):
        if base_lib and base_name:
            instrument_cache = ".instruments"
            path = os.path.join(instrument_cache, f"{name}.npy")
            if os.path.exists(path):
                print("loading", path)
                synth = np.load(path, allow_pickle = True)[0]                
            else:
                clip = base_lib(base_name)
                synth = Synth(dbg=self.dbg).from_clip(name, clip, **kwargs)
                print("saving", path)
                if not os.path.exists(instrument_cache):
                    os.mkdir(instrument_cache)
                np.save(path, np.array([synth]))
            return synth
        elif harmonics:
            return Synth(dbg=self.dbg).from_harmonics(name, harmonics)

synth_lib = SynthLib("synth_lib")

