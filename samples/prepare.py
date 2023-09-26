import engine
import zipfile
import os

def prepare(name, dn, fn, trim_start= 0, ease_start= 0, trim_end= 0, ease_end= 0, lo=100):
        
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
    end = clip.duration
    ease = clip.interp_envelope([0, ease_start, end-ease_end, end], [0, 1, 1, 0]) ** 2
    clip.apply_envelope(ease)

    if True:
        ax_spectrum, ax_hpfilter, ax_clip_spectrum = engine.dbg.axs(3, "load " + name)
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

        ax_end.set_xlim(clip.duration - 0.1, clip.duration)
        clip.plot_buf(ax_end)
        envelope.plot_buf(ax_end)

        #ax_absbuf, ax_envfilter, ax_envelope = dbg.axs(3)
        #clip.plot_buf(ax_absbuf)
        #filter.plot_spectrum(ax_envfilter)
        #envelope.plot_buf(ax_envelope)

    #clip.play()
    fn = os.path.join(os.path.dirname(__file__), f"{name}.wav")
    clip.write(fn, fp=True)


samples = {
    # dn, fn, trim_start, ease_start, trim_end, ease_end
    "guitar_a3": ["guitar", "guitar_A3_very-long_forte_normal", 1.925, 0, 1, 1],
    "clarinet_a3": ["clarinet", "clarinet_A3_1_forte_normal", 0, 0, 0, 0.1],
    "saxophone_a3": ["saxophone", "saxophone_A3_15_forte_normal", 0.035, 0, 0.05, 0.05],
}

def prepare_all():
    for sample, args in samples.items():
        prepare(sample, *args)

prepare_all()
