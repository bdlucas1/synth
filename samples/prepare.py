import engine
import zipfile
import os

def prepare(name, dn, fn, trim_start, ease_start, trim_end, ease_end, lo=100, save=True, dbg=True):
        
    print("preparing", name)

    # get clip from library
    zipfn = f"/Users/bdlucas1/Downloads/all-samples/{dn}.zip"
    f = zipfile.ZipFile(zipfn, 'r').open(f"{fn}.mp3")
    original_clip = engine.Clip(True).read(f)

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
    end = clip.dur
    ease = clip.interp_envelope([0, ease_start, end-ease_end, end], [0, 1, 1, 0]) ** 2
    clip.apply_envelope(ease)

    if dbg:
        ax_spectrum, ax_hpfilter, ax_clip_spectrum = engine.dbg.axs(3, f"load {name}")
        original_clip.plot_spectrum(ax_spectrum)
        hpfilter.plot_spectrum(ax_hpfilter)
        clip.plot_spectrum(ax_clip_spectrum)

        ax_envelope, ax_start, ax_end = engine.dbg.axs(3, ["envelope", "start", "end"])

        envelope = clip.get_envelope(normalize=False)
        clip.plot_buf(ax_envelope)
        envelope.plot_buf(ax_envelope)

        ax_start.set_xlim(0, 0.1)
        clip.plot_buf(ax_start)
        envelope.plot_buf(ax_start)

        ax_end.set_xlim(clip.dur - 0.1, clip.dur)
        clip.plot_buf(ax_end)
        envelope.plot_buf(ax_end)

        #ax_absbuf, ax_envfilter, ax_envelope = dbg.axs(3)
        #clip.plot_buf(ax_absbuf)
        #filter.plot_spectrum(ax_envfilter)
        #envelope.plot_buf(ax_envelope)

    if save:
        fn = os.path.join(os.path.dirname(__file__), f"{name}.flac")
        clip.write(fn)

    if dbg:
        clip.play()

    return clip


samples = {
    # dn, fn, trim_start, ease_start, trim_end, ease_end
    "guitar_a2_f": ["guitar", "guitar_A2_very-long_forte_normal", 0.4, 0, 1.5, 0],
    "guitar_a3_f": ["guitar", "guitar_A3_very-long_forte_normal", 1.925, 0, 1, 1],
    "guitar_a2_p": ["guitar", "guitar_A2_very-long_piano_normal", 0.195, 0, 0.5, 0],
    "guitar_a3_p": ["guitar", "guitar_A3_very-long_piano_normal", 0.196, 0, 0, 0],
    "clarinet_a3_f": ["clarinet", "clarinet_A3_1_forte_normal", 0, 0, 0, 0.1],
    "clarinet_a3_p": ["clarinet", "clarinet_A3_1_piano_normal", 0, 0, 0, 0.1],
    "clarinet_a5_f": ["clarinet", "clarinet_A5_1_forte_normal", 0, 0, 0, 0.1],
    "clarinet_a5_p": ["clarinet", "clarinet_A5_1_piano_normal", 0, 0, 0, 0.1],
    "saxophone_a3": ["saxophone", "saxophone_A3_15_forte_normal", 0.035, 0, 0.05, 0.05],
}

def prepare_all():
    for name, args in samples.items():
        prepare(name, *args, dbg=False)

def prepare_one(name):
    prepare(name, *samples[name], dbg=True)

if __name__ == "__main__":
    prepare_all()
