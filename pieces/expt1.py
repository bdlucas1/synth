import piece

# rhythmic (accent) patterns apply to succeeding notes from start of bar (I)
I121 = I > [(1/4,8), (2/4,8)] # quarter note vol +8, then half note vol +8
I4 = I > [(4/4,8)]            # whole note vol +8

# portamento of -2 semitones for the last 1/16 of a 1/2 note
port = [(1/2-1/16, 0), (1/16, 0, -2)]

P(
    guitar,
    tempo(4,120),
    time(4,4),
    transpose(-12),

    S(
        I121, a4/4, d/2@port, b/4, I4, c/1,
        I121, a4/4, d/2     , b/4, I4, c/1,
    )

).write().play()

