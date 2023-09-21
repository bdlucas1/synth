import notation

# rhythmic (accent) patterns apply to succeeding notes of measure
I1 = I > [(1/4,8), (1/4,0), (1/4,5), (1/4,0)]
I112 = I > [(1/4,8), (1/4,0), (2/4,5)]
I121 = I > [(1/4,8), (2/4,8)]
I22 = I > [(2/4,8), (2/4,5)]
I4 = I > [(4/4,8)]
I31 = I > [(3/4,8)]
I211 = I > [(2/4,8)]

# define a portamento of s semitones for the last x of a note of length t
def port(t, s=-2, x=1/16):
    return [(t-x,0),(x,0,s)]


P(
    guitar,
    tempo(4,120),
    time(4,4),
    transpose(-12),

    S(
        ~I1,c5/4,g,c,e,    ~I1,g,e,c,g,      ~I1,a,+f,d,c,             ~I112,b,d,-g/2,
        ~I1,c5/4,g,c,e,    ~I1,g,e,c,g,      ~I121,a,d/2@port(1/2),b/4,  ~I4,c/1,

        ~I22,d5/2,e,       ~I1,d/4,-g,g,g,   ~I22,+d/2,e,       ~I1,d/4,-g,g,g,
        ~I22,b4/2,c,       ~I22,d,e,         ~I1,b/4,b,c,a,     ~I1,g,g,g,g,

        ~I22,g5/2,e,       ~I22,f,d,         ~I22,e,c,          ~I1,b/4,g,g,g,
        ~I22,g5/2,e,       ~I22,f,d,         ~I22,e,c,          ~I22,b/2,r,

        ~I1,c5/4,g,c,e,    ~I1,g,e,c,g,      ~I1,a,+f,d,c,      ~I1,b,g,a,b,
        ~I1,c5/4,g,c,e,    ~I1,e,-a,d,f,     ~I1,e,c,d,b,       I,c/(2,4)%ring
    ),

    S(
        ~I4,c4/1,          ~I31,t/(2,4),e/4, ~I4,f/1,           ~I211,g/2,r/4,f,
        ~I4,e4/1,          ~I31,t/(2,4),e/4, ~I22,f/2, g/2,     ~I1,-c/4,+g,e,c,

        ~I1,b3/4,+g,-c,+g, ~I4,-b/1,         ~I1,b3/4,+g,-c,+g, ~I4,-b/1,
        ~I1,g3/4,+g,-a,+g, ~I1,-b,+g,-c,+g,  ~I22,d/2,d,        ~I4,-g/1,

        ~I1,e4/4,g,-c,+g,  ~I1,d,g,-b,+g,    ~I1,-c,+g,e,g,     ~I4,d/1,
        ~I1,e4/4,g,-c,+g,  ~I1,d,g,-b,+g,    ~I1,-c,+g,e,g,     ~I1,g,+g,f,d,

        ~I4,c4/1,          ~I4,e,            ~I4,f,             ~I22,g/2,f,
        ~I4,e4/1,          ~I4,f,            ~I22,g/2,-g/2,      I,+c/4>-10,P(e>-8,g>-8),-c%ring
    )

).write().play()

