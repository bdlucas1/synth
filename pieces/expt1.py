import notation

# rhythmic patterns
I1 = I > [(1/4,8), (1/4,0), (1/4,5), (1/4,0)]
I112 = I > [(1/4,8), (1/4,0), (2/4,5)]
I121 = I > [(1/4,8), (2/4,8)]
I22 = I > [(2/4,8), (2/4,5)]
I4 = I > [(4/4,8)]
I31 = I > [(3/4,8)]
I211 = I > [(2/4,8)]

x = 1/16
#fall = lambda t: PC((t-x,0),(x,0,-2))
fall = lambda t: _ @ [(t-x,0),(x,0,-2)]

P(
    guitar,
    tempo(4,120),
    time(4,4),
    transpose(-12),

    S(
        ~I121,a4/4,d/2/fall(1/2),b/4,  ~I4,c/1,
        ~I121,a4/4,d/2          ,b/4,  ~I4,c/1,

    ) * 1,

).write().play()

