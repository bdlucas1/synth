import piece

# define stress (accent) patterns that apply to succeeding notes of measure
# p primary, n none, s secondary
Ipns = I > [10, 0, 5]
Ipp = I > [10, 10]
Ips = I > [10, 5]
Ip = I > [10]

# portamento of -2 semitones for the last 1/16 of a 1/2 note
#port = [(1/2-1/16,0), (1/16,0,-2)]

# more general version - can be attached to note of any length
port = lambda note: [(7/8*note.dur_units,0), (1/8*note.dur_units,0,-2)]

P(
    guitar,
    tempo(4,120),
    time(4,4),
    transpose(-12),

    S(
        ~Ipns,c5/4,g,c,e,    ~Ipns,g,e,c,g,     ~Ipns,a,+f@port,d,c, ~Ipns,b,d,-g/2,
        ~Ipns,c5/4,g,c,e,    ~Ipns,g,e,c,g,     ~Ipp,a,d/2@port,b/4, ~Ip,c/1,

        ~Ips,d5/2,e,         ~Ipns,d/4,-g,g,g,  ~Ips,+d/2,e,         ~Ipns,d/4,-g,g,g,
        ~Ips,b4/2,c,         ~Ips,d,e,          ~Ipns,b/4,b,c,a,     ~Ipns,g,g,g,g,

        ~Ips,g5/2,e,         ~Ips,f,d,          ~Ips,e,c,            ~Ipns,b/4,g,g,g,
        ~Ips,g5/2,e,         ~Ips,f,d,          ~Ips,e,c,            ~Ips,b/2,r,

        ~Ipns,c5/4,g,c,e,    ~Ipns,g,e,c,g,     ~Ipns,a,+f,d,c,      ~Ipns,b,g,a,b,
        ~Ipns,c5/4,g,c,e,    ~Ipns,e,-a,d,f,    ~Ipns,e,c@port,d,b,  I,c/(2,4)%ring
    ),

    S(
        v45,
        ~I,c4/1,            ~Ip,t/(2,4),e/4,    ~Ip,f/1,             ~Ip,g/2,r/4,f,
        ~Ip,e4/1,            ~Ip,t/(2,4),e/4,   ~Ips,f/2, g/2,       ~I,-c/4%ring,+g,e,c,

        ~Ipns,b3/4,+g,-c,+g, ~Ip,-b/1,          ~Ipns,b3/4,+g,-c,+g, ~Ip,-b/1,
        ~Ipns,g3/4,+g,-a,+g, ~Ipns,-b,+g,-c,+g, ~Ips,d/2,d,          ~I,-g/1,

        ~Ipns,e4/4,g,-c,+g,  ~Ipns,d,g,-b,+g,   ~Ipns,-c,+g,e,g,     ~Ip,d/1,
        ~Ipns,e4/4,g,-c,+g,  ~Ipns,d,g,-b,+g,   ~Ipns,-c,+g,e,g,     ~Ipns,g,g4,f,d,

        ~I,c4/1,            ~Ip,e,              ~Ip,f,               ~Ips,g/2,f,
        ~I,e4/1,            ~Ip,f,              ~Ips,g/2,-g/2,       I,+c/4>-10,P(e>-8,g>-8),-c%ring,r,
        ~I,
    )
)    
