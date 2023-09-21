import notation

# rhythmic (accent) patterns apply to succeeding notes
R121 = I > [(1/4,8), (2/4,8)] # quarter note vol +8, then half note vol +8
R4 = I > [(4/4,8)]            # whole note vol +8

# define a portamento of s semitones for the last x of a note of length t
def port(t, s=-2, x=1/16):
    return [(t-x,0),(x,0,s)]

P(
    guitar,
    tempo(4,120),
    time(4,4),
    transpose(-12),

    S(
        R121, a4/4, d/2@port(1/2), b/4, R4, c/1,
        R121, a4/4, d/2          , b/4, R4, c/1,
    )

).write().play()

