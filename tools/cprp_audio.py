"""CPRP audio engine — produced music bed, sound design and mastered mix for 15 s reels.

Usage:
    from cprp_audio import build_mix
    build_mix(narration=[(0.15, "n0.wav"), ...], out="mix.wav",
              impacts=[1.05], reveal=(7.6, 12.2), dur=15.0, seed=1, mood="playful")

Requires: numpy, pedalboard (pip install --break-system-packages pedalboard).
Narration wavs: mono or stereo, any sample rate (resampled).
Output: 44.1 kHz stereo 16-bit WAV, mastered ~-14 LUFS-ish (final loudnorm still done by ffmpeg).
"""
import wave
import numpy as np
from pedalboard import (Pedalboard, Reverb, Compressor, Limiter, LadderFilter, Chorus, Delay,
                        HighpassFilter, LowpassFilter, PeakFilter, HighShelfFilter, Gain)

SR = 44100

# ---------------------------------------------------------------- helpers
def _t(n): return np.arange(n) / SR
def _env(n, a=0.005, d=0.2):
    t = _t(n); e = np.minimum(1, t / max(a, 1e-4)) * np.exp(-t / max(d, 1e-4)); return e
def _place(buf, x, at, gain=1.0):
    i = int(at * SR); L = x.shape[-1]
    if i >= buf.shape[-1] or i + L <= 0: return
    j0 = max(0, -i); i = max(0, i); m = min(L - j0, buf.shape[-1] - i)
    if m <= 0: return
    buf[..., i:i + m] += gain * x[..., j0:j0 + m]
def _pan(x, p):  # p -1..1
    l = np.cos((p + 1) * np.pi / 4); r = np.sin((p + 1) * np.pi / 4)
    return np.stack([x * l, x * r])
def _fx(x, board):  # x: (ch, n)
    return board(x.astype(np.float32), SR)
def _midi(m): return 440.0 * 2 ** ((m - 69) / 12)
def _saw(f, n, detune=0.0, rng=None):
    t = _t(n); ph = (rng.random() if rng is not None else 0)
    return 2 * ((f * (1 + detune) * t + ph) % 1) - 1

def read_wav(path):
    with wave.open(path) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        a = np.frombuffer(w.readframes(w.getnframes()), {2: np.int16, 4: np.int32}[sw]).astype(np.float32)
        a /= 32768.0 if sw == 2 else 2147483648.0
        if ch > 1: a = a.reshape(-1, ch).mean(1)
    if sr != SR:
        n = int(len(a) * SR / sr)
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
    return a

# ---------------------------------------------------------------- instruments
def kick(rng):
    n = int(0.45 * SR); t = _t(n)
    f = 45 + 110 * np.exp(-t * 28)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)
    click = rng.standard_normal(n) * np.exp(-t * 300) * 0.25
    return np.tanh(1.6 * (body + click)) * 0.9

def clap(rng):
    n = int(0.35 * SR); t = _t(n); x = np.zeros(n)
    for k, off in enumerate((0, 0.011, 0.022)):
        i = int(off * SR); m = n - i
        x[i:] += rng.standard_normal(m) * np.exp(-t[:m] * (60 if k < 2 else 18))
    b = Pedalboard([HighpassFilter(900), PeakFilter(1800, 4, 1.2), Reverb(room_size=0.25, wet_level=0.2, dry_level=0.9)])
    return _fx(np.stack([x, x]) * 0.35, b)

def hat(rng, open_=False):
    n = int((0.22 if open_ else 0.05) * SR); t = _t(n)
    x = rng.standard_normal(n) * np.exp(-t * (14 if open_ else 90))
    return _fx(np.stack([x, x]) * 0.18, Pedalboard([HighpassFilter(7000)]))[0]

def bass_note(f, dur, rng):
    n = int(dur * SR); t = _t(n)
    x = 0.6 * _saw(f, n, 0, rng) + 0.4 * np.sin(2 * np.pi * f / 2 * t)
    e = np.minimum(1, t / 0.004) * np.exp(-t * 3.5)
    x = x * e
    b = Pedalboard([LadderFilter(mode=LadderFilter.Mode.LPF24, cutoff_hz=420, resonance=0.35, drive=2.0)])
    return _fx(np.stack([x, x]), b)[0] * 0.55

def pluck(f, rng, dur=0.5):
    n = int(dur * SR); t = _t(n)
    x = sum(a * np.sin(2 * np.pi * f * h * t) for h, a in ((1, 1), (2, 0.35), (3, 0.18), (4.02, 0.08)))
    return x * np.minimum(1, t / 0.002) * np.exp(-t * 9) * 0.22

def pad_chord(notes, dur, rng):
    n = int(dur * SR); t = _t(n); x = np.zeros(n)
    for m in notes:
        for d in (-0.006, 0.0, 0.007):
            x += _saw(_midi(m), n, d, rng)
    x /= (len(notes) * 3)
    e = np.minimum(1, t / 0.35) * np.minimum(1, (dur - t) / 0.4).clip(0, 1)
    st = np.stack([x * e, x * e])
    b = Pedalboard([LowpassFilter(1600), Chorus(rate_hz=0.6, depth=0.35, mix=0.5), Reverb(room_size=0.7, wet_level=0.35, dry_level=0.7, width=1.0)])
    return _fx(st, b) * 0.32

def riser(dur, rng):
    n = int(dur * SR); t = _t(n); p = t / dur
    noise = rng.standard_normal(n) * p ** 2
    tone = np.sin(2 * np.pi * np.cumsum(200 + 1400 * p ** 2) / SR) * p ** 3 * 0.25
    x = np.stack([noise * 0.35 + tone, noise * 0.35 + tone])
    out = np.zeros_like(x); seg = int(0.05 * SR)
    for i in range(0, n, seg):  # sweeping high-pass
        b = Pedalboard([HighpassFilter(300 + 6000 * (i / n) ** 2)])
        out[:, i:i + seg] = _fx(x[:, i:i + seg], b)
    return out * 0.6

def whoosh(dur, rng):
    n = int(dur * SR); t = _t(n); p = t / dur
    x = rng.standard_normal(n) * np.sin(np.pi * p) ** 2
    l = _fx(np.stack([x, x]), Pedalboard([LowpassFilter(2500), HighpassFilter(250)]))
    pan = np.linspace(-0.8, 0.8, n)
    return np.stack([l[0] * np.cos((pan + 1) * np.pi / 4), l[1] * np.sin((pan + 1) * np.pi / 4)]) * 0.9

def impact(rng):
    n = int(1.2 * SR); t = _t(n)
    sub = np.sin(2 * np.pi * np.cumsum(38 + 90 * np.exp(-t * 18)) / SR) * np.exp(-t * 3.2)
    crack = rng.standard_normal(n) * np.exp(-t * 35)
    body = np.sin(2 * np.pi * 160 * t) * np.exp(-t * 14) * 0.4
    x = np.tanh(2.2 * (sub + 0.35 * crack + body)) * 0.85
    return _fx(np.stack([x, x]), Pedalboard([Reverb(room_size=0.6, wet_level=0.25, dry_level=0.9)]))

def stamp(rng):  # patent reveal: paper thud + bright ding
    n = int(1.4 * SR); t = _t(n)
    thud = np.sin(2 * np.pi * np.cumsum(70 + 120 * np.exp(-t * 40)) / SR) * np.exp(-t * 16)
    paper = rng.standard_normal(n) * np.exp(-t * 25) * 0.25
    ding = sum(a * np.sin(2 * np.pi * f * t) for f, a in ((1568, 1), (2352, 0.4), (3136, 0.2))) * np.exp(-t * 3.5) * 0.18
    x = thud * 0.8 + paper + np.r_[np.zeros(int(0.06 * SR)), ding[:-int(0.06 * SR)]]
    return _fx(np.stack([x, x]), Pedalboard([Reverb(room_size=0.5, wet_level=0.25, dry_level=0.9)]))

# ---------------------------------------------------------------- music bed
MOODS = {  # bpm, root midi, progression (scale degrees as chords of midi offsets)
    "playful": (118, 50, [[0, 3, 7, 10], [5, 8, 12, 15], [-2, 2, 5, 9], [3, 7, 10, 14]]),   # Dm7 Gm7 C7sus-ish Fmaj
    "tech":    (112, 45, [[0, 3, 7, 12], [-4, 0, 3, 8], [3, 7, 10, 15], [-2, 2, 5, 10]]),  # Am F C G
    "wonder":  (100, 48, [[0, 4, 7, 11], [-3, 0, 4, 9], [5, 9, 12, 16], [7, 11, 14, 19]]), # Cmaj7 Am7 Fmaj7 G
}

def music(dur, reveal, rng, mood="playful"):
    bpm, root, prog = MOODS.get(mood, MOODS["playful"])
    beat = 60 / bpm; bar = beat * 4
    n = int(dur * SR)
    drums = np.zeros((2, n)); bass = np.zeros((2, n)); keys = np.zeros((2, n)); pads = np.zeros((2, n))
    r0, r1 = reveal if reveal else (dur + 1, dur + 1)
    K = kick(rng); C = clap(rng)
    nb = int(dur / beat) + 2
    for b in range(nb):
        tb = b * beat
        in_rev = r0 - 0.05 <= tb < r1 - 0.02
        if not in_rev:
            _place(drums, np.stack([K, K]), tb, 0.9)
            if b % 2 == 1: _place(drums, C, tb, 0.8)
        elif b % 2 == 0:  # half-time pulse keeps momentum under the reveal
            _place(drums, np.stack([K, K]), tb, 0.55)
        for s in range(4):  # 16th hats with swing + accents
            ts = tb + s * beat / 4 + (0.018 if s % 2 else 0)
            if in_rev and s % 2: continue
            h = hat(rng, open_=(s == 2 and b % 4 == 3))
            _place(drums, _pan(h, 0.35 if s % 2 else -0.25), ts, (0.9 if s == 0 else 0.55) * (0.75 if in_rev else 1))
    # bass + arp follow the chord progression, one chord per bar
    for k in range(int(dur / bar) + 2):
        tb = k * bar; ch = prog[k % len(prog)]
        if r0 - 0.05 <= tb < r1 - 0.02:  # sustained sub under the reveal
            f = _midi(root - 12 + ch[0])
            _place(bass, np.stack([bass_note(f, bar, rng) * 0.6] * 2), tb)
        else:
            for e in range(8):  # 8th-note bass, octave bounce
                te = tb + e * beat / 2
                if r0 - 0.05 <= te < r1: continue
                f = _midi(root - 12 + ch[0] + (12 if e % 4 == 3 else 0))
                _place(bass, np.stack([bass_note(f, beat / 2 * 0.95, rng)] * 2), te)
        arp = [ch[0], ch[1], ch[2], ch[3], ch[2], ch[1], ch[3] + 12, ch[2]]
        for e in range(16):
            te = tb + e * beat / 4
            p = pluck(_midi(root + 12 + arp[e % len(arp)]), rng)
            _place(keys, _pan(p, -0.5 if e % 2 else 0.5), te, 0.8 if r0 <= te < r1 else 1.0)
        _place(pads, pad_chord([root + x for x in ch], bar + 0.5, rng), tb)
    keys = _fx(keys, Pedalboard([Delay(delay_seconds=beat * 0.75, feedback=0.3, mix=0.25), Reverb(room_size=0.4, wet_level=0.2)]))
    # during the reveal: filter the whole bed down (tension), keep pads/keys
    bed = drums * 0.9 + bass * 0.9 + keys * 0.55 + pads * 0.7
    if reveal:
        i0, i1 = int(r0 * SR), int(r1 * SR)
        seg = _fx(bed[:, i0:i1], Pedalboard([LowpassFilter(2600)]))
        bed[:, i0:i1] = seg * 2.0
    bed = _fx(bed, Pedalboard([HighpassFilter(30), Compressor(threshold_db=-16, ratio=3, attack_ms=10, release_ms=120)]))
    return bed

# ---------------------------------------------------------------- voice chain
def voice_chain(v):
    st = np.stack([v, v])
    b = Pedalboard([HighpassFilter(85), PeakFilter(250, -2.0, 1.0), PeakFilter(3200, 3.0, 0.9),
                    HighShelfFilter(9000, 2.0), Compressor(threshold_db=-20, ratio=3.5, attack_ms=4, release_ms=80),
                    Gain(3), Reverb(room_size=0.12, wet_level=0.06, dry_level=1.0, width=0.3)])
    return _fx(st, b)

# ---------------------------------------------------------------- full mix
def build_mix(narration, out, impacts=(), reveal=None, dur=15.0, seed=1, mood="playful", whooshes=()):
    """narration: list of (start_s, wav_path). impacts: hit times (s). reveal: (start, end) of patent card.
    whooshes: transition times (s) for a short whoosh centred on the cut."""
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    v = np.zeros(n, np.float32)
    for st, p in narration:
        a = read_wav(p); i = int(st * SR); v[i:i + len(a)] += a[: max(0, n - i)]
    if np.abs(v).max() > 0: v *= 0.7 / np.abs(v).max()
    voice = voice_chain(v)
    bed = music(dur, reveal, rng, mood)
    sfx = np.zeros((2, n))
    for t in impacts:
        _place(sfx, whoosh(0.32, rng), t - 0.32, 0.8)
        _place(sfx, impact(rng), t, 1.0)
    for t in whooshes:
        _place(sfx, whoosh(0.28, rng), t - 0.18, 0.55)
    if reveal:
        r0, r1 = reveal
        _place(sfx, riser(0.9, rng), r0 - 0.9, 0.7)
        _place(sfx, stamp(rng), r0, 0.9)
        _place(sfx, riser(1.2, rng), r1 - 1.2, 0.9)  # build back into the close
        _place(sfx, impact(rng), r1, 0.55)  # re-entry hit for the close
    # end button: final hit on the downbeat near the end
    _place(sfx, impact(rng) * 0.6, dur - 0.9)
    # sidechain: duck bed under voice (smooth envelope)
    env = np.abs(voice).mean(0)
    k = int(0.06 * SR); env = np.convolve(env, np.ones(k) / k, mode="same")
    duck = 1 - 0.45 * np.clip(env / (env.max() * 0.25 + 1e-9), 0, 1)
    mix = voice * 1.0 + bed * 0.45 * duck + sfx * 0.6 * (1 - 0.3 * (duck < 0.7))
    # fades + master bus
    fi = int(0.02 * SR); fo = int(0.6 * SR)
    mix[:, :fi] *= np.linspace(0, 1, fi); mix[:, -fo:] *= np.linspace(1, 0, fo) ** 1.5
    master = Pedalboard([HighpassFilter(28), Compressor(threshold_db=-14, ratio=2.2, attack_ms=15, release_ms=150),
                         Limiter(threshold_db=-1.5, release_ms=80)])
    mix = _fx(mix, master)
    mix = mix / max(1e-9, np.abs(mix).max()) * 0.84
    with wave.open(out, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((mix.T * 32767).astype(np.int16).tobytes())
    return out
