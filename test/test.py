import notation
import sys
import soundfile as sf
import engine
import os

#
#
#

class Test:

    def test1():
        return S(tempo(4,120), sin, c4/4, e, g, P(-c, e, g, c)*2, -c/2, +c)

    def test2():
        return S(
            tempo(4,120)/sin,
            S(
                a4/4 * 2,
                P(a4/8, cs5) * 3,
                a4/8
            ) * 2,
            a4/2
        )

    # volume
    def test3():
        return S(c/4, v9, c, d, v5, c, v9, b, v5, c)

    # rests
    def test4():
        return P(
            S(c/4,  r/2,  +c/4, r, -c/2),
            S(c3/4, c, c, c,    c, c/2)
        )

    # dotted rhythms
    def test5():
        return P(
            guitar,
            tempo(1,60),
            S(a/(2,4,8), b/8, c/1/ring),
            S(e/1/ring, -g/1/ring)
        )

    # bar check
    def test6():
        return S(time(3,4) | c/4, d, e | f/(2,4) | P(c, e, g))

    # tie with bar check
    def test7():
        return P(
            S(I, c4/(1,2,4),              +g/4, I),
            S(I, e4/1,        I, t/(2,4), +c/4, I)
        )

    # pause, extend with bar check
    def test8():
        return S(I, c, e, g, c/8, x/16, p/16, b/8, I, P(-c/2,e,g,c))

    # temp %
    def test9():
        return S(c/2, +g%8%v90, c)

    # volume
    def test10():
        return S(c, +c>30, -c>-30, +c, -c%v70, c)

    # volume contour
    def test11():
        I1 = I > [(1/4,20), (1/4,0), (1/4,10), (1/4,0)]
        I2 = I > [(1/2,20), (1/2,10)]
        return S(
            guitar,
            tempo(4,120),
            time(4,4),
            transpose(-12),
            c4%ring, ~I1,+c,g,c,e, I2,g/2,-c, I
        )

    # not sure how relevant this is any more?
    def test12():
        I1 = I > [(1/4,30)]
        return S(guitar, I1, c/4%ring, c/4)

    # volume contour cases
    def test13():
        I1 = I > [(2/4,10), (1/4,20), (1/4,30), (1/4,40)]
        return S(time(5,4), clarinet, ~I1,c/4,d/4,e/4,f/2, I1,g/4>5,a/4,b/4,c/2)
        
    # pitch contour, testing constant and contoured pitch, for elastic and non-elastic inst
    def test14():
        return S(
            *(S(inst, I @ [(1/2,0,2), (1/4,2)], c/(2,4), d)
              for inst in [guitar, clarinet]
            )
        )


def compare(fn1, fn2):

    try:
        clip1 = engine.Clip().read(fn1)
    except:
        print(fn1, "not found")
        clip1 = None

    try:
        clip2 = engine.Clip().read(fn2)
    except:
        print(fn2, "not found")
        clip2 = None

    if clip1 and clip2:
        if len(clip1) == len(clip2):
            #eq = all(b1 == b2 for (b1, b2) in zip(clip1.buf, clip2.buf))
            max_diff = max(abs(clip1.buf-clip2.buf)) / max(abs(clip1.buf))
            #print(f"abs max_diff {max_diff}")
            max_diff = max_diff / max(abs(clip1.buf))
            #print(f"rel max_diff {max_diff}")
            eq = len(clip1)==len(clip2) and max_diff < 1e-2
            err = f"max_diff {max_diff:.3f}"
        else:
            eq = False
            err = f"lengths differ: clip1 {len(clip1)}, clip2 {len(clip2)}"
    else:
        err = "missing clip"
        eq = False

    if eq:
        print("PASS", fn2)
    else:
        print(f"FAIL {fn2} {err}")
        if clip1:
            print("playing", fn1)
            clip1.play()
        if clip2:
            print("playing", fn2)
            clip2.play()
        #exit(-1)

def test(name):
    print("=== running", name)
    fn = f"{name}.mp3"
    tmp_fn = os.path.join("/tmp", fn)
    ref_fn = os.path.join(os.path.dirname(__file__), fn)
    getattr(Test, name)().render().write(tmp_fn)
    compare(ref_fn, tmp_fn)

test("test12")
test("test13")
test("test14")
test("test11")
test("test9")
test("test10")
test("test8")
test("test7")
test("test6")
test("test1")
test("test2")
test("test4")
test("test5")
