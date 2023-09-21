Cheat sheet

| Notation | Description |
| --- | --- |
| S(...)                      | Content played sequentially. Content may be notes or instructions that apply to all succeeding notes. |
| P(...)                      | Content like S but played in parallel. S and P may be arbitrarily nested. |
| c0, cs0, df0, ..., cf7, c7  | Notes with absolute pitch. |
| c, cs, d, ... bf, b         | Notes with pitches relative to most recent pitch (LilyPond style).|
| +x, -x                      | Relative pitches one octave higher or lower (LilyPond x' or x,)
| r                           | Rest. |
| x/1, x/2, x/4, ...          | Whole, half, quarter, ... note duration for note or rest x. |
| x/(1,2), x/(1,2,4), x/(2,4), ... | Dotted whole note, double dotted whole note, dotted half note, ... Duration as fraction of a whole note is sum of reciprocals. |
| v0, v1, ..., v100           | Absolute volume instruction for succeeding notes, on a logarithmic scale. |
| x/v0, x/v1, ..., x/v100     | Absolute volume for x and succeeding notes. |
| x>v                         | Volume adjustment v for note x. May be positive or negative. Added to absolute volume. |
| x>(v,w)                     | Volume adjustment varying from v at start to w and end of note x. |
| x>[(dur0,v0),(dur1,v1),...] | Volume adjustment possibly spanning multiple notes starting with x. Volume adjustment v0 for dur0, then v1 for dur1, ... Dur is 1, 1/2, 1/4, ... for whole, half, quarter, ... note. x may be a note or an instruction such as bar check I. |
| x>[(dur0,v0,w0),...]        | Volume adjustment varying from v0 to w0 over period dur0. |
| x@p                         | Pitch adjustment p for note x in units of semitones. May be positive or negative. Added to normal pitch for x. |
| x@(p,q)                     | Pitch adjustment varying from p at start to q and end of note x. |
| x@[(dur0,p0),(dur1,p1),...] | Pitch adjustment possibly spanning multiple notes starting with x. Pitch adjustment p0 for dur0, then p1 for dur1, ... |
| x@[(dur0,p0,q0),...]        | Pitch adjustment varying from p0 to q0 over period dur0, ... |
| x*n                         | Repeat x n times, where x is S(), P(), or a note. |
| I                           | Bar check marker instruction. (LilyPond \|) |
| t/dur                       | Tied note. Extends previous note by dur. Useful for notes that cross bar boundaries. |
| h/dur                       | Hold. Extends previous note by dur but does not count against beat. |
| p/dur                       | Pause. Like a rest but does not count against beat. |
| guitar, clarinet, saxophone | Instrument instruction. Applies to all succeeding notes. |
| time(m,n)                   | Time signature m/n.
| tempo(m,n)                  | Tempo marking: 1/m note is n beats per minute. |
| transpose(n)                | Transpose all succeeding notes by n semitones. |

