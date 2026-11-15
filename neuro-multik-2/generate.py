#!/usr/bin/env python3
"""Мультфильм «Что если ты станешь самым богатым человеком на Земле?»
1080x1920, 30 fps, ~60 c. Этапы: voice | audio | frames | mux | all"""
import json
import math
import os
import shutil
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(OUT, "voice_rich")
FRAMES_DIR = os.path.join(OUT, "frames_rich")
FONT_PATH = os.path.join(os.path.dirname(__import__("matplotlib").__file__),
                         "mpl-data", "fonts", "ttf", "DejaVuSans-Bold.ttf")
TIMELINE = os.path.join(OUT, "timeline_rich.json")
MIX = os.path.join(OUT, "final_mix_rich.wav")
ENVELOPE = os.path.join(OUT, "envelope_rich.npy")
VIDEO = os.path.join(OUT, "samyy_bogatyy_tiktok.mp4")

W, H = 1080, 1920
FPS = 30
SS = 2
DW, DH = W * SS, H * SS
SCW, SCH = 2700, 4800
OX, OY = (SCW - DW) // 2, (SCH - DH) // 2
MIN_ZOOM, MAX_ZOOM = 0.8, 2.6

GAP = 0.2
INTRO_HOLD = 0.5
OUTRO_HOLD = 1.8

OUTLINE = (62, 40, 32)
SKIN = (255, 216, 178)
HAIR = (104, 62, 34)
SHIRT = (242, 122, 58)
PANTS = (52, 96, 66)
WHITE = (255, 255, 255)
YELLOW = (255, 214, 64)
GOLD = (255, 200, 60)
ORANGE = (255, 145, 40)
RED = (235, 70, 60)
BLUE = (70, 150, 245)
GREEN = (90, 190, 110)
VIOLET = (150, 90, 230)
DARK = (40, 35, 40)

LINES = [
    "Поздравляю: ты проснулся самым богатым человеком на Земле! Что дальше?",
    "Твой капитал — двести миллиардов долларов. Стопка купюр дотянется до Луны!",
    "Первым делом покупаешь остров: свой пляж, своя пальма — полный кайф!",
    "Дальше — частный самолёт за сто миллионов!",
    "Теперь серьёзно: покупаешь целую компанию!",
    "Яхта, вертолёт, замок в горах — всё твоё за одну неделю!",
    "Но тут поворот: каждый день сотни людей просят у тебя денег!",
    "Бизнес требует внимания: отчёты, суды, советы директоров!",
    "Друзья ведут себя странно, а враги появляются из ниоткуда.",
    "Ты богат, но спишь по четыре часа и забыл, когда смеялся.",
    "Учёные выяснили: после определённой суммы деньги счастье не добавляют!",
    "Счастье — это моменты: друзья, семья, любимое дело. Их не купишь!",
    "Зато без денег туго: они решают денежные проблемы — но не все!",
    "Так можно ли купить счастье? Частично — да. Полностью — увы!",
    "Сохрани это видео — вдруг разбогатеешь!",
    "А что купил бы первым делом ты? Пиши в комменты и подписывайся!",
]

KEYWORDS = [
    {"богатым", "дальше"},
    {"миллиардов", "луны"},
    {"остров", "кайф"},
    {"самолёт", "кофе"},
    {"компанию", "можешь"},
    {"яхта", "неделю"},
    {"поворот", "денег"},
    {"отчёты", "суды"},
    {"друзья", "враги"},
    {"богат", "смеялся"},
    {"учёные", "счастья"},
    {"друзья", "купишь"},
    {"деньги", "не все"},
    {"купить", "частично"},
    {"сохрани"},
    {"комменты", "подписывайся"},
]

RAINBOW = [(235, 60, 60), (250, 140, 40), (250, 210, 50), (90, 200, 90),
           (70, 160, 245), (90, 90, 220), (150, 80, 220)]

_font_cache = {}
_bg_cache = {}


def get_font(size):
    size = max(1, int(size))
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
    return _font_cache[size]


def PX(x):
    return OX + int(round(x * SS))


def PY(y):
    return OY + int(round(y * SS))


def S(v):
    return int(round(v * SS))


def clamp(x, a, b):
    return min(max(x, a), b)


def clamp01(x):
    return clamp(x, 0.0, 1.0)


def smooth(t):
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def ease_out(t):
    t = clamp01(t)
    return 1 - (1 - t) ** 3


def ease_out_back(t, s=1.6):
    t = clamp01(t)
    t -= 1
    return t * t * ((s + 1) * t + s) + 1


# ================================================================ ЭТАП: VOICE

def stage_voice():
    import torch
    model, _ = torch.hub.load("snakers4/silero-models", "silero_tts",
                              language="ru", speaker="v4_ru", trust_repo=True)
    os.makedirs(VOICE_DIR, exist_ok=True)
    timeline = []
    for i, text in enumerate(LINES):
        path = os.path.join(VOICE_DIR, f"line_{i:02d}.wav")
        if os.path.exists(path):
            with wave.open(path, "rb") as f:
                dur = f.getnframes() / f.getframerate()
        else:
            audio = model.apply_tts(text=text, speaker="eugene",
                                    sample_rate=48000, put_accent=True, put_yo=True)
            audio = audio.numpy()
            audio = audio / max(1e-9, np.max(np.abs(audio))) * 0.85
            with wave.open(path, "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(48000)
                f.writeframes((audio * 32767).astype("int16").tobytes())
            dur = len(audio) / 48000
        timeline.append({"i": i, "text": text, "dur": round(dur, 3)})
        print(f"[{i}] {dur:.2f}с  {text}", flush=True)
    with open(TIMELINE, "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=1)


# ================================================================ ЭТАП: AUDIO

def build_timeline():
    with open(TIMELINE, encoding="utf-8") as f:
        tl = json.load(f)
    t = INTRO_HOLD
    for item in tl:
        item["start"] = round(t, 3)
        t += item["dur"] + GAP
    total = tl[-1]["start"] + tl[-1]["dur"] + OUTRO_HOLD
    return tl, round(total, 3)


def synth_music(n, sr):
    audio = np.zeros(n)
    bpm = 126
    beat = 60.0 / bpm
    mel = [523.25, 659.25, 783.99, 659.25, 587.33, 659.25, 880.00, 783.99,
           523.25, 659.25, 783.99, 880.00, 1046.5, 880.00, 783.99, 659.25]
    bass = [130.81, 130.81, 174.61, 196.00]
    t = np.arange(n) / sr

    def tone(freq, start, dur, amp, kind="square"):
        i0 = int(start * sr)
        i1 = min(int((start + dur) * sr), n)
        if i0 >= n:
            return
        tt = t[i0:i1] - start
        if kind == "square":
            sig = np.sign(np.sin(2 * np.pi * freq * tt)) * 0.5 + \
                  np.sin(2 * np.pi * freq * tt) * 0.5
        else:
            sig = np.sin(2 * np.pi * freq * tt)
        audio[i0:i1] += sig * np.exp(-tt * 4.5) * amp

    for g in range(int(n / sr / beat) + 1):
        start = g * beat
        tone(mel[g % len(mel)], start, beat * 0.85, 0.10)
        if g % 2 == 0:
            tone(bass[(g // 2) % len(bass)] / 2, start, beat * 1.5, 0.12, "sine")
        i0 = int(start * sr)
        i1 = min(i0 + int(0.04 * sr), n)
        if i0 < n:
            rng = np.random.default_rng(g)
            audio[i0:i1] += rng.uniform(-1, 1, i1 - i0) * \
                np.exp(-np.arange(i1 - i0) / sr * 70) * 0.05
    return audio


def synth_sfx(kind, sr):
    def env_exp(m, k):
        return np.exp(-np.arange(m) / sr * k)

    if kind == "whoosh":
        m = int(0.4 * sr)
        rng = np.random.default_rng(7)
        noise = rng.uniform(-1, 1, m)
        sweep = np.sin(np.linspace(0.5, 2.2, m) * np.arange(m) / sr * 40)
        return noise * np.hanning(m) * sweep * 1.2
    if kind == "pop":
        m = int(0.12 * sr)
        tt = np.arange(m) / sr
        f = 500 + 2200 * tt / 0.12
        return np.sin(2 * np.pi * f * tt) * env_exp(m, 22)
    if kind == "cash":
        out = np.zeros(int(0.35 * sr))
        for k, f in enumerate([1319, 1760]):
            m = int(0.12 * sr)
            tt = np.arange(m) / sr
            off = int(k * 0.08 * sr)
            out[off:off + m] += np.sin(2 * np.pi * f * tt) * env_exp(m, 14) * 0.8
        return out
    if kind == "sad":
        m = int(0.8 * sr)
        tt = np.arange(m) / sr
        f = 392 * np.exp(-tt * 0.8)
        return np.sin(2 * np.pi * f * tt) * np.hanning(m) * 0.5
    if kind == "sparkle":
        out = np.zeros(int(0.35 * sr))
        for k, f in enumerate([1568, 2093, 2637]):
            m = int(0.15 * sr)
            tt = np.arange(m) / sr
            off = int(k * 0.07 * sr)
            out[off:off + m] += np.sin(2 * np.pi * f * tt) * env_exp(m, 16) * 0.7
        return out
    if kind == "tada":
        out = np.zeros(int(0.7 * sr))
        for k, f in enumerate([523, 659, 784, 1047]):
            m = int(0.3 * sr)
            tt = np.arange(m) / sr
            off = int(k * 0.09 * sr)
            sig = (np.sign(np.sin(2 * np.pi * f * tt)) * 0.4 +
                   np.sin(2 * np.pi * f * tt) * 0.6) * env_exp(m, 7)
            out[off:off + m] += sig * 0.8
        return out
    return np.zeros(1)


def sfx_at(at, kind, buf, sr):
    sig = synth_sfx(kind, sr)
    i0 = int(max(0, at) * sr)
    i1 = min(i0 + len(sig), len(buf))
    if i1 > i0:
        buf[i0:i1] += sig[:i1 - i0]


def stage_audio():
    tl, total = build_timeline()
    sr = 48000
    n = int(total * sr)
    voice = np.zeros(n)
    for item in tl:
        with wave.open(os.path.join(VOICE_DIR, f"line_{item['i']:02d}.wav"), "rb") as f:
            data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16) / 32767.0
        i0 = int(item["start"] * sr)
        i1 = min(i0 + len(data), n)
        voice[i0:i1] += data[:i1 - i0]

    music = synth_music(n, sr)
    sfx = np.zeros(n)
    for item in tl:
        sfx_at(item["start"] - 0.05, "whoosh", sfx, sr)
    events = [
        (0.4, "tada"), (tl[1]["start"] + 0.8, "cash"),
        (tl[2]["start"] + 0.5, "cash"), (tl[3]["start"] + 0.6, "cash"),
        (tl[4]["start"] + 0.8, "cash"), (tl[5]["start"] + 0.4, "pop"),
        (tl[5]["start"] + 1.1, "pop"), (tl[5]["start"] + 1.8, "pop"),
        (tl[6]["start"], "sad"), (tl[7]["start"] + 0.5, "pop"),
        (tl[8]["start"] + 0.6, "sad"), (tl[9]["start"] + 0.5, "sad"),
        (tl[10]["start"] + 1.0, "sparkle"), (tl[11]["start"] + 0.6, "sparkle"),
        (tl[12]["start"] + 0.5, "cash"), (tl[13]["start"], "tada"),
        (tl[14]["start"] + 0.5, "pop"), (tl[15]["start"] + 0.6, "pop"),
        (tl[15]["start"] + 1.5, "pop"),
    ]
    for at, kind in events:
        sfx_at(at, kind, sfx, sr)

    mix = voice + music * 0.55 + sfx * 0.5
    mix = mix / max(1e-9, np.max(np.abs(mix))) * 0.88
    stereo = np.stack([mix, mix], axis=1)
    with wave.open(MIX, "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((stereo * 32767).astype("int16").tobytes())

    win = sr // FPS
    m = len(voice) // win
    env = np.sqrt((voice[:m * win].reshape(m, win) ** 2).mean(axis=1))
    env = env / max(1e-9, env.max())
    env = np.clip(env * 5.0, 0, 1)
    for k in range(1, len(env)):
        env[k] = env[k] * 0.55 + env[k - 1] * 0.45
    np.save(ENVELOPE, env)
    print(f"Аудио готово: {total:.1f}с", flush=True)


# ================================================================ ПЕРСОНАЖ

GESTURES = {
    "idle": (-70, 120, 70, 120),
    "up": (-150, -150, 150, -150),
    "wave": (-70, 120, 170, -130),
    "point_right": (-70, 120, 200, -10),
    "point_left": (-200, -10, 70, 120),
    "open": (-190, 0, 190, 0),
    "think": (-70, 120, 55, -115),
    "cover": (-45, -125, 45, -125),
}


def draw_guy(d, cx, fy, t, emotion="talk", mouth=0.0, gesture="idle", scale=1.0):
    def X(v):
        return PX(cx + v * scale)

    def Y(v):
        return PY(fy + v * scale)

    bob = math.sin(t * 2.1) * 5 * scale
    lw = max(2, S(7 * scale))

    for sgn in (-1, 1):
        d.rounded_rectangle([X(sgn * 52 - 34), Y(-250 + bob), X(sgn * 52 + 34), Y(0)],
                            radius=S(16 * scale), fill=PANTS, outline=OUTLINE, width=lw)
        d.ellipse([X(sgn * 52 - 44), Y(-26 + bob), X(sgn * 52 + 48), Y(14 + bob)],
                  fill=WHITE, outline=OUTLINE, width=lw)

    sh_y = -520 + bob
    dx_l, dy_l, dx_r, dy_r = GESTURES.get(gesture, GESTURES["idle"])
    wob = math.sin(t * 3) * 8 * scale
    for dx, dy in ((dx_l, dy_l + wob), (dx_r, dy_r - wob)):
        sxp = 105 if dx > 0 else -105
        d.line([X(sxp), Y(sh_y), X(dx), Y(sh_y + dy)],
               fill=SHIRT, width=max(2, S(26 * scale)))
    for dx, dy in ((dx_l, dy_l + wob), (dx_r, dy_r - wob)):
        d.ellipse([X(dx - 24), Y(sh_y + dy - 24), X(dx + 24), Y(sh_y + dy + 24)],
                  fill=SKIN, outline=OUTLINE, width=lw)

    d.rounded_rectangle([X(-120), Y(-560 + bob), X(120), Y(-230 + bob)],
                        radius=S(40 * scale), fill=SHIRT, outline=OUTLINE, width=lw)
    d.rounded_rectangle([X(-120), Y(-330 + bob), X(120), Y(-230 + bob)],
                        radius=S(30 * scale), fill=PANTS, outline=OUTLINE, width=lw)

    hy = -660 + bob
    hr = 112
    for sgn in (-1, 1):
        d.ellipse([X(sgn * hr - 14), Y(hy - 16), X(sgn * hr + 18), Y(hy + 20)],
                  fill=SKIN, outline=OUTLINE, width=max(2, S(5 * scale)))
    d.ellipse([X(-hr), Y(hy - hr), X(hr), Y(hy + hr)],
              fill=SKIN, outline=OUTLINE, width=lw)
    d.pieslice([X(-hr - 8), Y(hy - hr - 12), X(hr + 8), Y(hy + 40)],
               180, 360, fill=HAIR)
    d.polygon([(X(-hr + 6), Y(hy - 12)), (X(-70), Y(hy - 50)), (X(-30), Y(hy - 4)),
               (X(5), Y(hy - 54)), (X(45), Y(hy - 2)), (X(80), Y(hy - 48)),
               (X(hr - 4), Y(hy - 10)), (X(hr - 4), Y(hy - 64)), (X(-hr + 6), Y(hy - 64))],
              fill=HAIR)

    ex, ey = 46, hy - 18
    big = 1.18 if emotion == "surprise" else 1.0
    blink = (math.sin(t * 1.9 + 0.7) > 0.977) and emotion not in ("surprise", "tired")
    for sgn in (-1, 1):
        cx_e, cy_e = X(sgn * ex), Y(ey)
        ew, eh = S(30 * scale * big), S(36 * scale * big)
        if emotion == "tired":
            d.line([cx_e - ew, cy_e - eh * 0.5, cx_e + ew, cy_e - eh * 0.5],
                   fill=OUTLINE, width=max(2, S(9 * scale)))
            d.arc([cx_e - ew, cy_e + eh * 0.2, cx_e + ew, cy_e + eh * 1.6],
                  200, 340, fill=OUTLINE, width=max(2, S(4 * scale)))
            continue
        if blink:
            d.line([cx_e - ew, cy_e, cx_e + ew, cy_e], fill=OUTLINE,
                   width=max(2, S(9 * scale)))
            continue
        d.ellipse([cx_e - ew, cy_e - eh, cx_e + ew, cy_e + eh],
                  fill=WHITE, outline=OUTLINE, width=max(2, S(6 * scale)))
        look = S(9 * scale)
        ir = S(19 * scale * big)
        d.ellipse([cx_e + look - ir, cy_e + S(4) - ir, cx_e + look + ir, cy_e + S(4) + ir],
                  fill=(122, 78, 42))
        d.ellipse([cx_e + look - S(9), cy_e + S(4) - S(9), cx_e + look + S(9), cy_e + S(4) + S(9)],
                  fill=(30, 20, 16))
        d.ellipse([cx_e + look - S(14), cy_e - S(8), cx_e + look - S(2), cy_e + S(4)],
                  fill=WHITE)

    by = hy - 74 - (16 if emotion == "surprise" else 0)
    if emotion == "tired":
        for sgn in (-1, 1):
            d.line([X(sgn * 70), Y(by + 6), X(sgn * 20), Y(by + 12)],
                   fill=HAIR, width=max(2, S(13 * scale)))
    else:
        tilt = 14 if emotion == "think" else 0
        for sgn in (-1, 1):
            y0 = Y(by + (tilt if sgn == -1 else 0))
            y1 = Y(by - (tilt if sgn == -1 else 0))
            d.line([X(sgn * 70), y0, X(sgn * 20), y1], fill=HAIR, width=max(2, S(13 * scale)))

    d.arc([X(-12), Y(hy + 2), X(14), Y(hy + 30)], 20, 160, fill=(210, 150, 110),
          width=max(2, S(6 * scale)))

    my = hy + 52
    if emotion == "surprise":
        d.ellipse([X(-22), Y(my - 22), X(22), Y(my + 26)],
                  fill=(120, 50, 45), outline=OUTLINE, width=max(2, S(5 * scale)))
    elif emotion == "smile":
        d.arc([X(-42), Y(my - 26), X(42), Y(my + 30)], 15, 165,
              fill=OUTLINE, width=max(2, S(9 * scale)))
    elif emotion == "tired":
        d.line([X(-26), Y(my + 6), X(26), Y(my + 6)], fill=OUTLINE,
               width=max(2, S(8 * scale)))
    else:
        mh = (6 + 44 * mouth) * scale
        d.rounded_rectangle([X(-30), Y(my - mh / 2), X(30), Y(my + mh / 2 + 6)],
                            radius=S(10 * scale), fill=(120, 50, 45),
                            outline=OUTLINE, width=max(2, S(5 * scale)))

    for sgn in (-1, 1):
        d.ellipse([X(sgn * 74 - 20), Y(hy + 26), X(sgn * 74 + 20), Y(hy + 44)],
                  fill=(255, 170, 150))


# ================================================================ ПОМОЩНИКИ

def scene_bg(key, top, bottom):
    if key in _bg_cache:
        return _bg_cache[key]
    rows = np.linspace(np.array(top, float), np.array(bottom, float), SCH)
    arr = np.repeat(rows[:, None, :], SCW, axis=1).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    _bg_cache[key] = img
    return img


def draw_arrow(d, x0, y0, x1, y1, color, width):
    d.line([PX(x0), PY(y0), PX(x1), PY(y1)], fill=color, width=S(width))
    ang = math.atan2(y1 - y0, x1 - x0)
    for da in (2.6, -2.6):
        d.line([PX(x1), PY(y1),
                PX(x1 - math.cos(ang + da) * 22), PY(y1 - math.sin(ang + da) * 22)],
               fill=color, width=S(width))


def draw_sparkle(d, cx, cy, r, col=YELLOW):
    for ang in (0, math.pi / 2):
        d.line([PX(cx - math.cos(ang) * r), PY(cy - math.sin(ang) * r),
                PX(cx + math.cos(ang) * r), PY(cy + math.sin(ang) * r)],
               fill=col, width=S(6))


def draw_subscribe(d, cx, cy, t):
    pulse = 1 + 0.08 * math.sin(t * 6)
    d.rounded_rectangle([PX(cx - 190), PY(cy - 54), PX(cx + 190), PY(cy + 54)],
                        radius=S(52 * pulse), fill=RED, outline=WHITE, width=S(8))
    d.text((PX(cx), PY(cy)), "ПОДПИШИСЬ", font=get_font(S(50 * pulse)),
           fill=WHITE, anchor="mm")


def draw_confetti(d, t, seed=0, n=26):
    rng = np.random.default_rng(seed)
    for i in range(n):
        x = rng.uniform(0, 1080)
        speed = rng.uniform(120, 260)
        y = ((t * speed + rng.uniform(0, 1920)) % 2100) - 90
        col = RAINBOW[i % len(RAINBOW)]
        cxp, cyp = PX(x + math.sin(t * 2 + i) * 24), PY(y)
        sz = S(11)
        d.rectangle([cxp - sz, cyp - sz // 2, cxp + sz, cyp + sz // 2], fill=col)


def draw_money(d, cx, cy, w=110, h=56, ang=0.0):
    pts = []
    ca, sa = math.cos(ang), math.sin(ang)
    for lx, ly in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        pts.append((PX(cx + lx * ca - ly * sa), PY(cy + lx * sa + ly * ca)))
    d.polygon(pts, fill=(120, 200, 120), outline=OUTLINE, width=S(5))
    r = h * 0.3
    d.ellipse([PX(cx - r), PY(cy - r), PX(cx + r), PY(cy + r)],
              outline=OUTLINE, width=S(4))
    d.text((PX(cx), PY(cy)), "$", font=get_font(S(26)), fill=(40, 90, 40), anchor="mm")


def draw_moneybag(d, cx, cy, r=80, t=0.0):
    sw = math.sin(t * 3) * 4
    d.polygon([(PX(cx - r * 0.45), PY(cy - r * 0.5)), (PX(cx + r * 0.45), PY(cy - r * 0.5)),
               (PX(cx + r), PY(cy + r * 0.9)), (PX(cx - r), PY(cy + r * 0.9))],
              fill=(200, 160, 90), outline=OUTLINE, width=S(6))
    d.rounded_rectangle([PX(cx - r * 0.5), PY(cy - r * 0.8), PX(cx + r * 0.5), PY(cy - r * 0.35)],
                        radius=S(10), fill=(160, 120, 60), outline=OUTLINE, width=S(5))
    d.text((PX(cx + sw), PY(cy + r * 0.15)), "$", font=get_font(S(int(r * 1.0))),
           fill=(60, 40, 20), anchor="mm")


def draw_heart(d, cx, cy, r, col=RED):
    d.ellipse([PX(cx - r), PY(cy - r * 0.9), PX(cx), PY(cy + r * 0.1)], fill=col)
    d.ellipse([PX(cx), PY(cy - r * 0.9), PX(cx + r), PY(cy + r * 0.1)], fill=col)
    d.polygon([(PX(cx - r), PY(cy - r * 0.25)), (PX(cx + r), PY(cy - r * 0.25)),
               (PX(cx), PY(cy + r))], fill=col)


def draw_simple_person(d, cx, fy, s=1.0, col=(90, 70, 60)):
    d.ellipse([PX(cx - 34 * s), PY(fy - 300 * s), PX(cx + 34 * s), PY(fy - 232 * s)],
              fill=col)
    d.rounded_rectangle([PX(cx - 44 * s), PY(fy - 226 * s), PX(cx + 44 * s), PY(fy - 20 * s)],
                        radius=S(20 * s), fill=col)


# ---------------------------------------------------------------- сцены

def sc_hook(img, d, t, dur, mouth):
    img.paste(scene_bg("hook", (255, 236, 180), (255, 190, 120)), (0, 0))
    d.rounded_rectangle([PX(60), PY(1380), PX(1020), PY(1800)], radius=S(40),
                        fill=WHITE, outline=OUTLINE, width=S(8))
    d.rounded_rectangle([PX(120), PY(1240), PX(960), PY(1400)], radius=S(30),
                        fill=(240, 170, 120), outline=OUTLINE, width=S(8))
    rng = np.random.default_rng(3)
    for i in range(16):
        x = rng.uniform(40, 1040)
        y = ((t * rng.uniform(180, 320) + rng.uniform(0, 1920)) % 2100) - 90
        draw_money(d, x, y, w=90, h=48, ang=math.sin(t * 2 + i) * 0.5)
    draw_guy(d, 540, 1560, t, emotion="surprise" if t > dur * 0.45 else "talk",
             mouth=mouth, gesture="up" if t > dur * 0.45 else "idle", scale=0.82)
    p = ease_out_back(clamp01((t - dur * 0.45) / 0.4))
    if p > 0.01:
        d.text((PX(540), PY(360)), "ТЫ МИЛЛИАРДЕР!", font=get_font(S(84 * p)),
               fill=GOLD, anchor="mm", stroke_width=S(10), stroke_fill=OUTLINE)
    d.text((PX(540), PY(560)), "$200 000 000 000", font=get_font(S(64)),
           fill=(60, 45, 30), anchor="mm", stroke_width=S(6), stroke_fill=WHITE)


def sc_moon(img, d, t, dur, mouth):
    img.paste(scene_bg("moon", (18, 24, 60), (50, 40, 80)), (0, 0))
    for i in range(20):
        sx = (i * 173) % 1080
        sy = (i * 97) % 700
        d.ellipse([PX(sx), PY(sy), PX(sx + 6), PY(sy + 6)], fill=WHITE)
    d.ellipse([PX(760), PY(140), PX(1010), PY(390)], fill=(240, 230, 190),
              outline=(200, 185, 140), width=S(6))
    for mx, my, mr in ((850, 220, 26), (930, 300, 18), (800, 320, 14)):
        d.ellipse([PX(mx - mr), PY(my - mr), PX(mx + mr), PY(my + mr)],
                  fill=(215, 200, 155))
    grow = ease_out(clamp01(t / (dur * 0.8)))
    base_y = 1560
    top_y = 620 - (620 - 460) * grow
    n = int(3 + grow * 22)
    for k in range(n):
        yy = base_y - k * ((base_y - top_y) / max(1, n - 1))
        w = 150 - k * 2
        d.ellipse([PX(540 - w), PY(yy - 26), PX(540 + w), PY(yy + 26)],
                  fill=GOLD, outline=(200, 150, 40), width=S(5))
    draw_guy(d, 190, 1620, t, emotion="talk", mouth=mouth, gesture="point_right", scale=0.78)
    p = ease_out_back(clamp01((t - 0.25) / 0.5))
    d.text((PX(330), PY(300)), "до Луны!", font=get_font(S(58 * p)), fill=GOLD,
           anchor="mm", stroke_width=S(8), stroke_fill=OUTLINE)


def sc_island(img, d, t, dur, mouth):
    img.paste(scene_bg("island", (160, 225, 255), (255, 230, 180)), (0, 0))
    d.ellipse([PX(60), PY(120), PX(230), PY(290)], fill=(255, 205, 100),
              outline=(230, 150, 50), width=S(6))
    d.rectangle([PX(0), PY(1180), PX(1080), PY(1920)], fill=(60, 150, 220))
    for k in range(3):
        yy = 1260 + k * 130
        ph = t * 2 + k
        d.arc([PX(-80 + ph * 40 % 200), PY(yy), PX(1160 + ph * 40 % 200), PY(yy + 120)],
              200, 340, fill=(240, 250, 255), width=S(8))
    d.ellipse([PX(120), PY(1520), PX(980), PY(1720)], fill=(245, 215, 150),
              outline=OUTLINE, width=S(7))
    d.arc([PX(700), PY(1050), PX(860), PY(1420)], 60, 180, fill=(120, 80, 40), width=S(22))
    for ang in (-0.9, -0.4, 0.2, 0.7, 1.2):
        lx = 780 + math.cos(ang) * 150
        ly = 1070 + math.sin(ang) * 90
        d.line([PX(780), PY(1070), PX(lx), PY(ly)], fill=(50, 150, 70), width=S(14))
        d.ellipse([PX(lx - 14), PY(ly - 14), PX(lx + 14), PY(ly + 14)],
                  fill=(60, 170, 80), outline=OUTLINE, width=S(4))
    d.rounded_rectangle([PX(300), PY(1420), PX(360), PY(1540)], radius=S(10),
                        fill=(255, 120, 140), outline=OUTLINE, width=S(5))
    d.polygon([(PX(300), PY(1400)), (PX(360), PY(1400)), (PX(330), PY(1360))],
              fill=(255, 120, 140), outline=OUTLINE)
    d.line([PX(330), PY(1360), PX(330), PY(1330)], fill=(230, 240, 250), width=S(6))
    draw_guy(d, 470, 1600, t, emotion="smile" if t > 0.8 else "talk", mouth=mouth,
             gesture="open", scale=0.75)
    p = ease_out_back(clamp01((t - 0.2) / 0.5))
    d.text((PX(540), PY(360)), "СВОЙ ОСТРОВ!", font=get_font(S(76 * p)), fill=GOLD,
           anchor="mm", stroke_width=S(10), stroke_fill=OUTLINE)
    draw_sparkle(d, 180 + math.sin(t * 3) * 30, 500, 26 + math.sin(t * 5) * 6)


def sc_jet(img, d, t, dur, mouth):
    img.paste(scene_bg("jet", (120, 190, 255), (200, 235, 255)), (0, 0))
    for i in range(5):
        cx_c = (i * 260 + t * 60) % 1300 - 110
        cy_c = 300 + (i % 3) * 180
        for ox, oy, r in ((0, 0, 55), (60, 10, 42), (-58, 12, 40)):
            d.ellipse([PX(cx_c + ox - r), PY(cy_c + oy - r * 0.7),
                       PX(cx_c + ox + r), PY(cy_c + oy + r * 0.7)], fill=WHITE)
    jy = 760 + math.sin(t * 1.5) * 20
    d.ellipse([PX(140), PY(jy - 70), PX(940), PY(jy + 70)], fill=WHITE,
              outline=OUTLINE, width=S(8))
    d.polygon([(PX(300), PY(jy - 60)), (PX(430), PY(jy - 200)), (PX(560), PY(jy - 60))],
              fill=WHITE, outline=OUTLINE, width=S(8))
    d.polygon([(PX(300), PY(jy + 60)), (PX(430), PY(jy + 190)), (PX(560), PY(jy + 60))],
              fill=(230, 235, 245), outline=OUTLINE, width=S(8))
    d.polygon([(PX(860), PY(jy - 50)), (PX(1000), PY(jy - 130)), (PX(1000), PY(jy + 20))],
              fill=(230, 235, 245), outline=OUTLINE, width=S(7))
    for k in range(6):
        wx = 620 + k * 52
        d.ellipse([PX(wx - 16), PY(jy - 34), PX(wx + 16), PY(jy + 10)],
                  fill=(140, 200, 250), outline=OUTLINE, width=S(4))
    d.polygon([(PX(140), PY(jy - 20)), (PX(60 + (t * 500) % 60), PY(jy - 8)),
               (PX(140), PY(jy + 20))], fill=(255, 200, 80))
    draw_guy(d, 300, 1660, t, emotion="talk", mouth=mouth, gesture="point_right", scale=0.7)
    p = ease_out_back(clamp01((t - 0.3) / 0.5))
    d.rounded_rectangle([PX(600), PY(1180), PX(1000), PY(1330)], radius=S(24),
                        fill=WHITE, outline=OUTLINE, width=S(7))
    d.text((PX(800), PY(1255)), "$100 000 000", font=get_font(S(52 * p)), fill=(60, 45, 30),
           anchor="mm")


def sc_company(img, d, t, dur, mouth):
    img.paste(scene_bg("company", (255, 225, 170), (255, 190, 130)), (0, 0))
    d.rectangle([PX(60), PY(700), PX(240), PY(1700)], fill=(150, 130, 120),
                outline=OUTLINE, width=S(6))
    d.rectangle([PX(860), PY(820), PX(1030), PY(1700)], fill=(160, 140, 130),
                outline=OUTLINE, width=S(6))
    d.rectangle([PX(300), PY(480), PX(820), PY(1700)], fill=(110, 120, 140),
                outline=OUTLINE, width=S(9))
    for row in range(7):
        for col in range(4):
            wx = 340 + col * 118
            wy = 540 + row * 150
            lit = (row + col) % 3 != 0
            d.rectangle([PX(wx), PY(wy), PX(wx + 78), PY(wy + 100)],
                        fill=(255, 220, 130) if lit else (70, 80, 100),
                        outline=OUTLINE, width=S(4))
    ang = -10
    bw, bh = 560, 130
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    pts = []
    for lx, ly in ((-bw / 2, -bh / 2), (bw / 2, -bh / 2), (bw / 2, bh / 2), (-bw / 2, bh / 2)):
        pts.append((PX(560 + lx * ca - ly * sa), PY(760 + lx * sa + ly * ca)))
    p = ease_out_back(clamp01((t - 0.4) / 0.45))
    if p > 0.01:
        sc = min(1.0, p)
        pts2 = [(560 + (px - 560) * sc, 760 + (py - 760) * sc) for px, py in
                [(v[0] / SS, v[1] / SS) for v in pts]]
        d.polygon([(PX(x), PY(y)) for x, y in pts2], fill=RED, outline=WHITE, width=S(6))
        d.text((PX(560), PY(760)), "ПРОДАНО", font=get_font(S(64 * sc)), fill=WHITE,
               anchor="mm")
    draw_guy(d, 820, 1660, t, emotion="smile", mouth=mouth, gesture="point_left", scale=0.7)


def sc_toys(img, d, t, dur, mouth):
    img.paste(scene_bg("toys", (170, 220, 255), (255, 235, 190)), (0, 0))
    items = [(0.3, 250, 1300, "yacht"), (1.0, 540, 1250, "heli"), (1.7, 840, 1300, "castle")]
    for t0, ix, iy, kind in items:
        p = ease_out_back(clamp01((t - t0) / 0.45))
        if p <= 0.08:
            continue
        if kind == "yacht":
            d.polygon([(PX(ix - 130 * p), PY(iy)), (PX(ix + 130 * p), PY(iy)),
                       (PX(ix + 90 * p), PY(iy + 70)), (PX(ix - 90 * p), PY(iy + 70))],
                      fill=WHITE, outline=OUTLINE, width=S(6))
            d.polygon([(PX(ix), PY(iy - 160 * p)), (PX(ix), PY(iy)), (PX(ix + 90 * p), PY(iy))],
                      fill=(255, 210, 90), outline=OUTLINE, width=S(5))
        elif kind == "heli":
            d.rounded_rectangle([PX(ix - 90 * p), PY(iy - 50), PX(ix + 90 * p), PY(iy + 50)],
                                radius=S(30), fill=RED, outline=OUTLINE, width=S(6))
            d.line([PX(ix - 110), PY(iy - 70), PX(ix + 110), PY(iy - 70)],
                   fill=OUTLINE, width=S(8))
            d.line([PX(ix), PY(iy - 70), PX(ix), PY(iy - 40)], fill=OUTLINE, width=S(8))
            d.line([PX(ix + 90), PY(iy + 50), PX(ix + 90), PY(iy + 90)], fill=OUTLINE, width=S(6))
            d.line([PX(ix + 50), PY(iy + 90), PX(ix + 130), PY(iy + 90)], fill=OUTLINE, width=S(6))
        else:
            d.rectangle([PX(ix - 100 * p), PY(iy - 60), PX(ix + 100 * p), PY(iy + 60)],
                        fill=(190, 170, 150), outline=OUTLINE, width=S(6))
            for tx in (-100, 0, 100):
                d.rectangle([PX(ix + tx * p - 34), PY(iy - 150 * p),
                             PX(ix + tx * p + 34), PY(iy - 40 * p)],
                            fill=(190, 170, 150), outline=OUTLINE, width=S(5))
                d.polygon([(PX(ix + tx * p - 40), PY(iy - 150 * p)),
                           (PX(ix + tx * p), PY(iy - 195 * p)),
                           (PX(ix + tx * p + 40), PY(iy - 150 * p))],
                          fill=(200, 60, 50), outline=OUTLINE, width=S(4))
    draw_guy(d, 540, 1700, t, emotion="talk", mouth=mouth, gesture="open", scale=0.62)
    p = ease_out_back(clamp01((t - 0.1) / 0.4))
    d.text((PX(540), PY(400)), "ВСЁ ТВОЁ!", font=get_font(S(80 * p)), fill=GOLD,
           anchor="mm", stroke_width=S(10), stroke_fill=OUTLINE)


def sc_requests(img, d, t, dur, mouth):
    img.paste(scene_bg("requests", (255, 205, 160), (240, 150, 110)), (0, 0))
    rng = np.random.default_rng(11)
    for i in range(12):
        side = i % 2
        k = i // 2
        x = 40 + k * 190 if side == 0 else 1040 - k * 190
        y = 500 + (i * 97) % 900
        hx = x + (60 if side == 0 else -60)
        ang = math.atan2(900 - y, 540 - x)
        ex = hx + math.cos(ang) * 55
        ey2 = y + math.sin(ang) * 55
        d.line([PX(x), PY(y), PX(ex), PY(ey2)], fill=SKIN, width=S(20))
        d.ellipse([PX(ex - 20), PY(ey2 - 20), PX(ex + 20), PY(ey2 + 20)],
                  fill=SKIN, outline=OUTLINE, width=S(4))
        bx = x + (0 if side == 0 else 0)
        by = y - 90
        pr = 1 + 0.1 * math.sin(t * 4 + i)
        d.ellipse([PX(bx - 40 * pr), PY(by - 30 * pr), PX(bx + 40 * pr), PY(by + 30 * pr)],
                  fill=WHITE, outline=OUTLINE, width=S(4))
        d.text((PX(bx), PY(by)), "$", font=get_font(S(38)), fill=(200, 130, 30), anchor="mm")
    draw_guy(d, 540, 1640, t, emotion="tired", mouth=mouth, gesture="cover", scale=0.78)
    p = ease_out_back(clamp01((t - 0.2) / 0.5))
    d.text((PX(540), PY(280)), "СОТНИ ПРОСЬБАТ!", font=get_font(S(66 * p)), fill=RED,
           anchor="mm", stroke_width=S(9), stroke_fill=WHITE)


def sc_papers(img, d, t, dur, mouth):
    img.paste(scene_bg("papers", (250, 230, 200), (235, 200, 160)), (0, 0))
    n = int(3 + ease_out(clamp01(t / (dur * 0.7))) * 12)
    for k in range(n):
        yy = 1500 - k * 62
        w = 300 + (k % 3) * 12
        d.rectangle([PX(700 - w), PY(yy - 26), PX(700 + w), PY(yy + 26)],
                    fill=WHITE, outline=OUTLINE, width=S(4))
        for ln in range(3):
            d.line([PX(700 - w + 26), PY(yy - 12 + ln * 12),
                    PX(700 + w - 26 - (ln % 2) * 40), PY(yy - 12 + ln * 12)],
                   fill=(170, 170, 175), width=S(3))
    for i, (txt, lx, ly) in enumerate((("ОТЧЁТЫ", 220, 620), ("СУДЫ", 180, 860),
                                       ("СОВЕТ ДИРЕКТОРОВ", 260, 1100))):
        p = ease_out_back(clamp01((t - 0.3 - i * 0.35) / 0.4))
        if p <= 0.01:
            continue
        d.rounded_rectangle([PX(lx - 150 * p), PY(ly - 44), PX(lx + 150 * p), PY(ly + 44)],
                            radius=S(20), fill=WHITE, outline=OUTLINE, width=S(5))
        d.text((PX(lx), PY(ly)), txt, font=get_font(S(36 * p)), fill=(90, 60, 40), anchor="mm")
    draw_guy(d, 300, 1640, t, emotion="tired", mouth=mouth, gesture="think", scale=0.72)


def sc_friends(img, d, t, dur, mouth):
    img.paste(scene_bg("friends", (60, 55, 75), (110, 85, 95)), (0, 0))
    draw_simple_person(d, 260, 1600, s=1.1, col=(70, 60, 75))
    draw_simple_person(d, 850, 1600, s=1.1, col=(70, 60, 75))
    d.arc([PX(180), PY(1180), PX(340), PY(1400)], 240, 30, fill=(70, 60, 75), width=S(9))
    d.arc([PX(770), PY(1180), PX(930), PY(1400)], 150, 300, fill=(70, 60, 75), width=S(9))
    p = ease_out_back(clamp01((t - 0.3) / 0.5))
    if p > 0.01:
        draw_heart(d, 540, 520, 90 * p, col=(150, 60, 70))
        zz = 26 * p
        d.line([PX(540 - zz), PY(520 - zz), PX(540 + zz), PY(520 + zz)],
               fill=(60, 55, 75), width=S(12))
    d.rounded_rectangle([PX(540 - 60), PY(1300), PX(540 + 60), PY(1420)], radius=S(14),
                        fill=(30, 30, 40), outline=OUTLINE, width=S(5))
    d.rectangle([PX(540 - 48), PY(1312), PX(540 + 48), PY(1408)], fill=(90, 140, 200))
    draw_guy(d, 540, 1660, t, emotion="tired", mouth=mouth, gesture="idle", scale=0.72)
    p2 = ease_out_back(clamp01((t - 0.5) / 0.5))
    d.text((PX(540), PY(300)), "ОДИНОКО НА ВЕРШИНЕ", font=get_font(S(56 * p2)),
           fill=(230, 220, 235), anchor="mm", stroke_width=S(8), stroke_fill=OUTLINE)


def sc_tired(img, d, t, dur, mouth):
    img.paste(scene_bg("tired", (40, 40, 60), (70, 55, 70)), (0, 0))
    d.ellipse([PX(660), PY(200), PX(980), PY(520)], fill=WHITE, outline=OUTLINE, width=S(8))
    d.line([PX(820), PY(360), PX(820), PY(250)], fill=OUTLINE, width=S(10))
    d.line([PX(820), PY(360), PX(900), PY(400)], fill=OUTLINE, width=S(10))
    d.ellipse([PX(806), PY(346), PX(834), PY(374)], fill=OUTLINE)
    d.text((PX(820), PY(600)), "04:00", font=get_font(S(56)), fill=(230, 220, 235),
           anchor="mm", stroke_width=S(6), stroke_fill=OUTLINE)
    for k in range(3):
        zx = 200 + math.sin(t * 1.5 + k) * 20
        zy = 700 - k * 90 - (t * 30) % 60
        d.text((PX(zx), PY(zy)), "z", font=get_font(S(44 + k * 12)),
               fill=(200, 200, 230), anchor="mm")
    draw_guy(d, 420, 1660, t, emotion="tired", mouth=mouth, gesture="idle", scale=0.75)
    p = ease_out_back(clamp01((t - 0.25) / 0.5))
    d.text((PX(540), PY(180)), "СОН: 4 ЧАСА", font=get_font(S(60 * p)), fill=(230, 220, 235),
           anchor="mm", stroke_width=S(8), stroke_fill=OUTLINE)


def sc_graph(img, d, t, dur, mouth):
    img.paste(scene_bg("graph", (250, 240, 220), (235, 215, 185)), (0, 0))
    ax0, ay0 = 180, 1450
    draw_arrow(d, ax0, ay0, ax0, 420, OUTLINE, 9)
    draw_arrow(d, ax0, ay0, 980, ay0, OUTLINE, 9)
    d.text((PX(120), PY(500)), "счастье", font=get_font(S(40)), fill=(90, 60, 40),
           anchor="mm")
    d.text((PX(900), PY(1510)), "деньги", font=get_font(S(40)), fill=(90, 60, 40),
           anchor="mm")
    d.line([PX(600), PY(420), PX(600), PY(1450)], fill=ORANGE, width=S(7))
    d.text((PX(600), PY(380)), "ПОРОГ", font=get_font(S(44)), fill=ORANGE, anchor="mm",
           stroke_width=S(5), stroke_fill=WHITE)
    prog = ease_out(clamp01(t / (dur * 0.75)))
    pts = []
    for i in range(41):
        f = i / 40 * prog
        x = 180 + f * 760
        y = 1200 - (1 - math.exp(-f * 3.2)) * 620
        pts.append((PX(x), PY(y)))
    if len(pts) > 1:
        d.line(pts, fill=(70, 130, 200), width=S(10), joint="curve")
    lx = 180 + prog * 760
    ly = 1200 - (1 - math.exp(-prog * 3.2)) * 620
    d.ellipse([PX(lx - 16), PY(ly - 16), PX(lx + 16), PY(ly + 16)],
              fill=GOLD, outline=OUTLINE, width=S(5))
    draw_moneybag(d, 930, 1600, r=70, t=t)
    draw_guy(d, 200, 1700, t, emotion="talk", mouth=mouth, gesture="point_right", scale=0.62)
    p = ease_out_back(clamp01((t - 0.2) / 0.5))
    d.text((PX(540), PY(280)), "БОЛЬШЕ ДЕНЕГ ≠ БОЛЬШЕ СЧАСТЬЯ", font=get_font(S(46 * p)),
           fill=(90, 60, 40), anchor="mm", stroke_width=S(6), stroke_fill=WHITE)


def sc_happiness(img, d, t, dur, mouth):
    img.paste(scene_bg("happiness", (255, 220, 170), (255, 190, 140)), (0, 0))
    p = ease_out_back(clamp01((t - 0.1) / 0.5))
    if p > 0.01:
        pulse = 1 + 0.06 * math.sin(t * 5)
        draw_heart(d, 540, 480, 150 * p * pulse, col=RED)
        draw_sparkle(d, 300 + math.sin(t * 3) * 40, 350, 24)
        draw_sparkle(d, 780 + math.cos(t * 2.5) * 40, 420, 20)
    ph = ease_out(clamp01((t - 0.4) / 0.6))
    for k, px in enumerate((300, 540, 780)):
        pk = clamp01(ph * 3 - k)
        if pk <= 0.01:
            continue
        draw_simple_person(d, px, 1560, s=0.9 * pk, col=(120, 90, 70))
    draw_guy(d, 540, 1760, t, emotion="smile", mouth=mouth, gesture="open", scale=0.55)
    p2 = ease_out_back(clamp01((t - 0.6) / 0.5))
    d.text((PX(540), PY(820)), "ЭТО НЕ КУПИШЬ", font=get_font(S(64 * p2)), fill=(90, 55, 35),
           anchor="mm", stroke_width=S(8), stroke_fill=WHITE)


def sc_scales(img, d, t, dur, mouth):
    img.paste(scene_bg("scales", (245, 235, 210), (220, 200, 170)), (0, 0))
    tilt = math.sin(clamp01((t - 0.3) / 0.8) * math.pi / 2) * 0.18
    px_, py_ = 540, 700
    d.rectangle([PX(px_ - 16), PY(py_), PX(px_ + 16), PY(py_ + 420)],
                fill=(120, 90, 60), outline=OUTLINE, width=S(6))
    d.ellipse([PX(px_ - 110), PY(py_ + 400), PX(px_ + 110), PY(py_ + 500)],
              fill=(120, 90, 60), outline=OUTLINE, width=S(6))
    bx, by = px_ - 260 * math.cos(tilt), py_ - 260 * math.sin(tilt)
    qx, qy = px_ + 260 * math.cos(tilt), py_ + 260 * math.sin(tilt)
    d.line([PX(bx), PY(by), PX(qx), PY(qy)], fill=(120, 90, 60), width=S(14))
    d.line([PX(bx), PY(by), PX(bx), PY(by + 70)], fill=(120, 90, 60), width=S(8))
    d.line([PX(qx), PY(qy), PX(qx), PY(qy + 70)], fill=(120, 90, 60), width=S(8))
    d.rounded_rectangle([PX(bx - 110), PY(by + 70), PX(bx + 110), PY(by + 130)],
                        radius=S(12), fill=WHITE, outline=OUTLINE, width=S(6))
    d.rounded_rectangle([PX(qx - 110), PY(qy + 70), PX(qx + 110), PY(qy + 130)],
                        radius=S(12), fill=WHITE, outline=OUTLINE, width=S(6))
    draw_moneybag(d, bx, by - 30, r=62, t=t)
    draw_heart(d, qx, qy - 20, 50, col=RED)
    draw_guy(d, 900, 1660, t, emotion="talk", mouth=mouth, gesture="point_left", scale=0.68)
    p = ease_out_back(clamp01((t - 0.5) / 0.5))
    d.text((PX(420), PY(300)), "ЧАСТИЧНО — ДА", font=get_font(S(62 * p)), fill=(90, 60, 40),
           anchor="mm", stroke_width=S(8), stroke_fill=WHITE)


def sc_final(img, d, t, dur, mouth):
    img.paste(scene_bg("finalL", (235, 90, 80), (200, 60, 60)), (0, 0))
    rows = np.linspace(np.array((80, 170, 110), float), np.array((60, 140, 90), float), SCH)
    right = np.repeat(rows[:, None, :], SCW // 2, axis=1).astype(np.uint8)
    img.paste(Image.fromarray(right, "RGB"), (SCW // 2, 0))
    d.text((PX(270), PY(430)), "ДА", font=get_font(S(130)), fill=WHITE, anchor="mm",
           stroke_width=S(10), stroke_fill=OUTLINE)
    d.text((PX(810), PY(430)), "НЕТ", font=get_font(S(130)), fill=WHITE, anchor="mm",
           stroke_width=S(10), stroke_fill=OUTLINE)
    d.text((PX(270), PY(580)), "частично", font=get_font(S(44)), fill=WHITE, anchor="mm")
    d.text((PX(810), PY(580)), "полностью", font=get_font(S(44)), fill=WHITE, anchor="mm")
    draw_guy(d, 540, 1560, t, emotion="talk", mouth=mouth, gesture="open", scale=0.8)
    draw_sparkle(d, 150 + math.sin(t * 3) * 30, 900, 24)
    draw_sparkle(d, 930 + math.cos(t * 2.7) * 30, 950, 20)


def sc_save(img, d, t, dur, mouth):
    img.paste(scene_bg("save", (255, 200, 120), (250, 160, 90)), (0, 0))
    pulse = 1 + 0.12 * math.sin(t * 6)
    cy_a = 760
    d.polygon([(PX(540 - 90 * pulse), PY(cy_a - 160 * pulse)),
               (PX(540 + 90 * pulse), PY(cy_a - 160 * pulse)),
               (PX(540 + 90 * pulse), PY(cy_a - 40 * pulse)),
               (PX(540 + 150 * pulse), PY(cy_a - 40 * pulse)),
               (PX(540), PY(cy_a + 120 * pulse)),
               (PX(540 - 150 * pulse), PY(cy_a - 40 * pulse)),
               (PX(540 - 90 * pulse), PY(cy_a - 40 * pulse))],
              fill=WHITE, outline=OUTLINE, width=S(8))
    d.rounded_rectangle([PX(540 - 120 * pulse), PY(cy_a + 180), PX(540 + 120 * pulse),
                         PY(cy_a + 240)], radius=S(16), fill=WHITE, outline=OUTLINE, width=S(7))
    draw_guy(d, 540, 1660, t, emotion="smile", mouth=mouth, gesture="up", scale=0.72)
    p = ease_out_back(clamp01((t - 0.15) / 0.5))
    d.text((PX(540), PY(360)), "СОХРАНИ ВИДЕО!", font=get_font(S(76 * p)), fill=WHITE,
           anchor="mm", stroke_width=S(10), stroke_fill=OUTLINE)
    draw_confetti(d, t, seed=5, n=18)


def sc_cta(img, d, t, dur, mouth):
    img.paste(scene_bg("cta", (255, 215, 150), (255, 175, 110)), (0, 0))
    draw_confetti(d, t, seed=9, n=30)
    p = ease_out_back(clamp01((t - 0.1) / 0.5))
    if p > 0.01:
        bx, by = 540, 420
        d.rounded_rectangle([PX(bx - 320 * p), PY(by - 90), PX(bx + 320 * p), PY(by + 90)],
                            radius=S(40), fill=WHITE, outline=OUTLINE, width=S(8))
        d.polygon([(PX(bx - 40), PY(by + 88)), (PX(bx + 40), PY(by + 88)), (PX(bx), PY(by + 140))],
                  fill=WHITE, outline=OUTLINE, width=S(6))
        d.text((PX(bx), PY(by)), "Что купишь первым?", font=get_font(S(50 * p)),
               fill=(90, 55, 35), anchor="mm")
    draw_guy(d, 540, 1560, t, emotion="smile", mouth=mouth, gesture="wave", scale=0.8)
    draw_subscribe(d, 540, 1150, t)
    draw_heart(d, 900, 950, 40 + math.sin(t * 4) * 6, col=RED)


SCENES = [sc_hook, sc_moon, sc_island, sc_jet, sc_company, sc_toys, sc_requests,
          sc_papers, sc_friends, sc_tired, sc_graph, sc_happiness, sc_scales,
          sc_final, sc_save, sc_cta]
SCENE_NAMES = ["hook", "moon", "island", "jet", "company", "toys", "requests",
               "papers", "friends", "tired", "graph", "happiness", "scales",
               "final", "save", "cta"]

CAM = {
    "hook": [(0.0, 1.0, 540, 960), (0.5, 1.3, 540, 900), (1.0, 1.12, 540, 940)],
    "moon": [(0.0, 1.15, 540, 800), (0.5, 0.95, 540, 1000), (1.0, 1.2, 620, 700)],
    "island": [(0.0, 1.25, 540, 1000), (0.5, 1.0, 540, 1000), (1.0, 1.15, 460, 950)],
    "jet": [(0.0, 1.35, 540, 800), (0.5, 1.0, 540, 950), (1.0, 1.2, 540, 900)],
    "company": [(0.0, 1.4, 540, 900), (0.5, 1.05, 540, 950), (1.0, 1.2, 540, 980)],
    "toys": [(0.0, 1.0, 540, 900), (0.5, 1.15, 540, 950), (1.0, 0.95, 540, 1000)],
    "requests": [(0.0, 1.3, 540, 900), (0.5, 1.0, 540, 950), (1.0, 1.25, 540, 950)],
    "papers": [(0.0, 1.2, 400, 950), (0.5, 1.0, 540, 950), (1.0, 1.3, 700, 850)],
    "friends": [(0.0, 1.25, 540, 850), (0.5, 1.0, 540, 950), (1.0, 1.15, 540, 900)],
    "tired": [(0.0, 1.35, 540, 800), (0.5, 1.05, 540, 950), (1.0, 1.2, 540, 900)],
    "graph": [(0.0, 1.3, 540, 900), (0.5, 1.0, 540, 950), (1.0, 1.15, 540, 900)],
    "happiness": [(0.0, 1.2, 540, 700), (0.5, 0.95, 540, 950), (1.0, 1.1, 540, 900)],
    "scales": [(0.0, 1.35, 540, 800), (0.5, 1.0, 540, 950), (1.0, 1.2, 540, 950)],
    "final": [(0.0, 1.4, 540, 800), (0.5, 1.05, 540, 950), (1.0, 1.2, 540, 950)],
    "save": [(0.0, 1.25, 540, 800), (0.5, 1.0, 540, 950), (1.0, 1.15, 540, 950)],
    "cta": [(0.0, 1.25, 540, 850), (0.5, 1.0, 540, 950), (1.0, 1.1, 540, 900)],
}


def camera_at(name, frac):
    keys = CAM[name]
    for k in range(len(keys) - 1):
        f0, z0, cx0, cy0 = keys[k]
        f1, z1, cx1, cy1 = keys[k + 1]
        if f0 <= frac <= f1:
            g = smooth((frac - f0) / max(1e-6, f1 - f0))
            return (z0 + (z1 - z0) * g, cx0 + (cx1 - cx0) * g, cy0 + (cy1 - cy0) * g)
    return keys[-1][1], keys[-1][2], keys[-1][3]


# ================================================================ СУБТИТРЫ

def chunk_text(text, maxlen=26):
    words = text.split()
    chunks, cur = [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if len(cand) > maxlen and cur:
            chunks.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        chunks.append(cur)
    return chunks


def draw_subtitle(d, text, keywords):
    f = get_font(46)
    words = text.split()
    space_w = f.getlength(" ")
    widths = [f.getlength(w) for w in words]
    total = sum(widths) + space_w * (len(words) - 1)
    x = 540 - total / 2
    y = 1660
    for w, wd in zip(words, widths):
        base = w.lower().strip(",.!?—")
        col = ORANGE if any(k in base for k in keywords) else WHITE
        d.text((x, y), w, font=f, fill=col, stroke_width=6,
               stroke_fill=(30, 20, 15), anchor="lm")
        x += wd + space_w


# ================================================================ ЭТАП: FRAMES

def stage_frames(start=0, end=None):
    tl, total = build_timeline()
    n_frames = int(round(total * FPS))
    if end is None:
        end = n_frames
    env = np.load(ENVELOPE)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    segs = [(item["start"], item["start"] + item["dur"] + GAP, idx)
            for idx, item in enumerate(tl)]

    for fi in range(start, end):
        out_path = os.path.join(FRAMES_DIR, f"f_{fi:05d}.png")
        if os.path.exists(out_path):
            continue
        t = fi / FPS
        img = Image.new("RGB", (SCW, SCH), (20, 20, 30))
        d = ImageDraw.Draw(img)

        cur = segs[0]
        for s0, s1, idx in segs:
            if s0 <= t < s1:
                cur = (s0, s1, idx)
                break
        s0, s1, idx = cur
        t_rel = t - s0
        dur = s1 - s0
        mouth = float(env[fi]) if fi < len(env) else 0.0

        SCENES[idx](img, d, t_rel, dur, mouth)

        frac = clamp01(t_rel / dur)
        punch = 1.0
        if t_rel < 0.22 and idx > 0:
            punch = 1.0 + 0.10 * (1 - t_rel / 0.22)
        z, ccx, ccy = camera_at(SCENE_NAMES[idx], frac)
        z = clamp(z * punch, MIN_ZOOM, MAX_ZOOM)
        win_w, win_h = DW / z, DH / z
        wx = clamp(OX + (ccx * SS - win_w / 2), 0, SCW - win_w)
        wy = clamp(OY + (ccy * SS - win_h / 2), 0, SCH - win_h)
        frame = img.crop((int(wx), int(wy), int(wx + win_w), int(wy + win_h)))
        frame = frame.resize((W, H), Image.LANCZOS)

        if t_rel < 0.22 and idx > 0:
            a = int(200 * (1 - t_rel / 0.22))
            overlay = Image.new("RGBA", (W, H), (255, 255, 255, a))
            frame = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")

        dd = ImageDraw.Draw(frame)
        item = tl[idx]
        if item["start"] - 0.05 <= t <= item["start"] + item["dur"]:
            chunks = chunk_text(item["text"])
            span = item["dur"] * 0.95
            ctime = clamp(t - item["start"], 0, span)
            ci = min(int(ctime / span * len(chunks)), len(chunks) - 1)
            draw_subtitle(dd, chunks[ci], KEYWORDS[idx])
        frame.save(out_path)
        if fi % 100 == 0:
            print(f"кадр {fi}/{end}", flush=True)
    print(f"Кадры {start}-{end} готовы", flush=True)


# ================================================================ ЭТАП: MUX

def stage_mux():
    tl, total = build_timeline()
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [ffmpeg, "-y", "-framerate", str(FPS),
           "-i", os.path.join(FRAMES_DIR, "f_%05d.png"),
           "-i", MIX,
           "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           "-shortest", "-movflags", "+faststart", VIDEO]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Готово: {VIDEO} ({os.path.getsize(VIDEO)/1e6:.1f} МБ, {total:.1f} с)", flush=True)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "voice":
        stage_voice()
    elif stage == "audio":
        stage_audio()
    elif stage == "frames":
        start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        end = int(sys.argv[3]) if len(sys.argv) > 3 else None
        stage_frames(start, end)
    elif stage == "mux":
        stage_mux()
    elif stage == "all":
        stage_voice()
        stage_audio()
        stage_frames()
        stage_mux()


if __name__ == "__main__":
    main()
