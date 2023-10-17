import piece
import sys
import soundfile as sf
import engine
import os
import numpy as np

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
        return S(I, c, e, g, c/8, h/16, p/16, b/8, I, P(-c/2,e,g,c))

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

    def test15():
        R = I > [20,0,10,0]
        return S(
            saxophone,
            R, c*4, R, d/8*4
        )

    def test16():
        lick = S(a2/2, cs, e, a)
        return S(guitar, lick, multi_guitar_f, lick)

    # next
    def test18():
        return S(
            c4,
            E1(g4),
            E2(a)
        ) * 2

    def test19():
        return S(
            P(
                S(a, b),
                S(c, d),
            ) * 2
        )


        


def compare(ref_fn, test_fn):

    try:
        ref_clip = engine.Clip().read(ref_fn)
    except:
        print(ref_fn, "not found")
        ref_clip = None

    try:
        test_clip = engine.Clip().read(test_fn)
    except:
        print(test_fn, "not found")
        test_clip = None

    if ref_clip and test_clip:
        if len(ref_clip) == len(test_clip):
            print(f"ref_clip max {max(abs(ref_clip.buf))} {ref_clip.buf.dtype}")
            print(f"test_clip max {max(abs(test_clip.buf))} {test_clip.buf.dtype}")            
            ref_clip.buf /= max(abs(ref_clip.buf))
            test_clip.buf /= max(abs(test_clip.buf))            
            max_diff = max(abs(ref_clip.buf-test_clip.buf))
            max_diff = max_diff / max(abs(ref_clip.buf))
            eq = len(ref_clip)==len(test_clip) and max_diff < 1e-2
            err = f"max_diff {max_diff:.3f}"
        else:
            eq = False
            err = f"lengths differ: ref_clip {len(ref_clip)}, test_clip {len(test_clip)}"
    else:
        err = "missing clip"
        eq = False

    if eq:
        print(f"max_diff {max_diff:.3f}")
        print("PASS", test_fn)
    else:
        print(f"FAIL {test_fn} {err}")
        if ref_clip:
            print("playing ref", ref_fn)
            ref_clip.play()
        if test_clip:
            print("playing test", test_fn)
            test_clip.play()
        exit(-1)

def test(name):
    print("=== running", name)
    fn = f"{name}.mp3"
    tmp_fn = os.path.join("/tmp", fn)
    ref_fn = os.path.join(os.path.dirname(__file__), fn)
    getattr(Test, name)().render().write(tmp_fn)
    compare(ref_fn, tmp_fn)

def test17():
    print("=== running test17 (musicxcml)")
    fn = "test17"
    xml_fn = os.path.join(os.path.dirname(__file__), "test17.xml")
    tmp_fn = "/tmp/test17.mp3"
    ref_fn = os.path.join(os.path.dirname(__file__), "test17.mp3")
    import musicxml
    musicxml.read(xml_fn).render().write(tmp_fn)
    compare(ref_fn, tmp_fn)

                          
test("test19")
test("test18")
test17() # musicxml
test("test16")
test("test11")
test("test15")
test("test12")
test("test13")
test("test14")
test("test9")
test("test10")
test("test8")
test("test7")
test("test6")
test("test1")
test("test2")
test("test4")
test("test5")
