#!/usr/bin/env python3
"""2D нейро-мультик для TikTok в формате «Что если…?»
«Что если гравитация исчезнет на 5 секунд?» — 1080x1920, 30 fps, 12 c."""
import math
import os
import shutil
import subprocess
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib

W, H = 1080, 1920
FPS = 30
DURATION = 12.0
N_FRAMES = int(FPS * DURATION)
SS = 2  # суперсэмплинг для сглаживания
DW, DH = W * SS, H * SS

T_GRAV_OFF = 3.0   # гравитация исчезает
T_GRAV_ON = 8.0    # гравитация возвращается

FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
FONT_BOLD = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(OUT_DIR, "frames")
VIDEO_PATH = os.path.join(OUT_DIR, "chto_esli_gravitacia_tiktok.mp4")
AUDIO_PATH = os.path.join(OUT_DIR, "music.wav")

# ---------------------------------------------------------------- утилиты

def ease_out_back(t, s=1.70158):
    t = min(max(t, 0.0), 1.0)
    t -= 1.0
    return t * t * ((s + 1) * t + s) + 1.0


def ease_out_cubic(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def ease_in_cubic(t):
    t = min(max(t, 0.0), 1.0)
    return t ** 3


def clamp01(x):
    return min(max(x, 0.0), 1.0)


def lerp(a, b, t):
    return a + (b - a) * t


def S(v):
    return int(round(v * SS))


# ---------------------------------------------------------------- фон

def make_background():
    img = Image.new("RGB", (DW, DH))
    top = (150, 120, 220)     # сиреневый «космос» сверху
    bottom = (255, 200, 160)  # тёплый закатный низ
    px = img.load()
    for y in range(DH):
        t = y / (DH - 1)
        px_row = (int(lerp(top[0], bottom[0], t)),
                  int(lerp(top[1], bottom[1], t)),
                  int(lerp(top[2], bottom[2], t)))
        for x in range(DW):
            px[x, y] = px_row
    return img


BG = None


def draw_sun(d, t):
    cx, cy, r = S(880), S(300), S(110)
    for i in range(12):
        ang = t * 0.3 + i * math.pi / 6
        x1 = cx + math.cos(ang) * (r + S(26))
        y1 = cy + math.sin(ang) * (r + S(26))
        x2 = cx + math.cos(ang) * (r + S(70))
        y2 = cy + math.sin(ang) * (r + S(70))
        d.line([x1, y1, x2, y2], fill=(255, 180, 120), width=S(12))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 200, 110))
    d.ellipse([cx - r + S(16), cy - r + S(16), cx + r - S(16), cy + r - S(16)],
              fill=(255, 224, 140))


def draw_stars_sky(d, t, k):
    """Звёздочки на небе, проявляются вместе с «антигравитацией»."""
    rng = math.sin(1.7)  # детерминированный сид
    for i in range(18):
        x = S(60 + (i * 197) % 960)
        y = S(120 + (i * 331) % 800)
        tw = 0.5 + 0.5 * math.sin(t * 3 + i * 1.3)
        r = S(4 + 4 * tw)
        d.ellipse([x - r, y - r, x + r, y + r],
                  fill=(255, 255, int(220 + 35 * tw)))


def draw_cloud(d, cx, cy, scale, tint=(255, 255, 255)):
    def puff(dx, dy, r):
        d.ellipse([S(cx + dx - r), S(cy + dy - r), S(cx + dx + r), S(cy + dy + r)],
                  fill=tint)
    puff(0, 0, 55 * scale)
    puff(-60 * scale, 15 * scale, 40 * scale)
    puff(60 * scale, 15 * scale, 42 * scale)
    puff(-25 * scale, -30 * scale, 38 * scale)


def draw_ground(d):
    d.rectangle([0, S(1560), DW, DH], fill=(146, 214, 145))
    d.rectangle([0, S(1560), DW, S(1590)], fill=(120, 195, 120))
    for i in range(7):
        x = S(60 + i * 165)
        y = S(1650 + (i % 3) * 80)
        r = S(24)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(120, 200, 120))


# ---------------------------------------------------------------- персонаж

def draw_cat(d, t, cx, cat_y, mode="idle", wave_t=0.0, tilt=0.0):
    """mode: idle | float | fall | landed"""
    hy = cat_y - S(300)   # центр головы
    hx = cx + tilt * S(40)
    hr = S(165)
    sx = 1.0 + tilt * 0.08
    sy = 1.0 - tilt * 0.08

    if mode == "landed":
        sq = 0.82
        sx, sy = 1.12, sq
        hy = cat_y - S(260)

    # ---- хвост
    tail_wag = math.sin(t * 5.0) * 40
    tx, ty = hx - S(120), cat_y - S(110)
    pts = []
    for i in range(13):
        f = i / 12
        px_ = tx - f * S(120) + math.sin(f * 2.5 + t * 5) * S(28)
        py_ = ty + f * S(30) - math.sin(f * math.pi) * S(70) + tail_wag * f * f
        pts.append((px_, py_))
    d.line(pts, fill=(255, 168, 92), width=S(34), joint="curve")
    tip = pts[-1]
    d.ellipse([tip[0] - S(24), tip[1] - S(24), tip[0] + S(24), tip[1] + S(24)],
              fill=(255, 168, 92))

    # ---- лапы-ноги
    if mode == "float":
        for sgn in (-1, 1):
            d.ellipse([hx + sgn * S(70) - S(36), cat_y - S(60) + math.sin(t * 4 + sgn) * S(14),
                       hx + sgn * S(70) + S(36), cat_y + S(10) + math.sin(t * 4 + sgn) * S(14)],
                      fill=(255, 150, 80))
    else:
        for dx in (-S(60), S(60)):
            d.ellipse([hx + dx - S(42), cat_y - S(40), hx + dx + S(42), cat_y + S(30)],
                      fill=(255, 150, 80))

    # ---- тело
    bw, bh = S(150 * sx), S(115 * sy)
    by = cat_y - S(130)
    d.ellipse([hx - bw, by - bh, hx + bw, by + bh], fill=(255, 184, 108))
    d.ellipse([hx - S(85), by - S(90), hx + S(85), by + S(20)], fill=(255, 226, 180))

    # ---- голова
    d.ellipse([hx - hr * sx, hy - hr * sy, hx + hr * sx, hy + hr * sy],
              fill=(255, 184, 108))

    # ---- уши
    for sgn in (-1, 1):
        ex = hx + sgn * hr * 0.62 * sx
        ey = hy - hr * 0.72 * sy
        d.polygon([(ex - sgn * S(52), ey + S(30)), (ex + sgn * S(18), ey + S(44)),
                   (ex + sgn * S(6), ey - S(78))], fill=(255, 184, 108))
        d.polygon([(ex - sgn * S(30), ey + S(22)), (ex + sgn * S(4), ey + S(30)),
                   (ex, ey - S(40))], fill=(255, 150, 170))

    # ---- полоска на голове
    d.polygon([(hx - S(16), hy - hr * sy + S(6)), (hx + S(16), hy - hr * sy + S(6)),
               (hx + S(8), hy - hr * sy + S(54)), (hx - S(8), hy - hr * sy + S(54))],
              fill=(240, 140, 70))

    # ---- глаза
    eye_dx = S(62)
    eye_y = hy - S(6)
    if mode == "float":
        # огромные удивлённые глаза
        for sgn in (-1, 1):
            ex = hx + sgn * eye_dx
            r = S(34)
            d.ellipse([ex - r, eye_y - r, ex + r, eye_y + r], fill=(60, 40, 40))
            d.ellipse([ex - S(14), eye_y - S(20), ex + S(8), eye_y + S(2)],
                      fill=(255, 255, 255))
            d.ellipse([ex - S(4), eye_y + S(6), ex + S(10), eye_y + S(20)],
                      fill=(255, 255, 255))
    elif mode == "fall":
        # зажмуренные испуганные глаза
        for sgn in (-1, 1):
            ex = hx + sgn * eye_dx
            d.line([ex - S(28), eye_y, ex + S(28), eye_y], fill=(60, 40, 40), width=S(12))
    else:
        blink = math.sin(t * 1.7) > 0.985
        for sgn in (-1, 1):
            ex = hx + sgn * eye_dx
            if blink:
                d.line([ex - S(26), eye_y, ex + S(26), eye_y],
                       fill=(60, 40, 40), width=S(12))
            else:
                r = S(26)
                d.ellipse([ex - r, eye_y - r, ex + r, eye_y + r], fill=(60, 40, 40))
                d.ellipse([ex - S(10), eye_y - S(16), ex + S(6), eye_y],
                          fill=(255, 255, 255))

    # ---- румянец
    for sgn in (-1, 1):
        ex = hx + sgn * S(105)
        d.ellipse([ex - S(26), eye_y + S(28), ex + S(26), eye_y + S(54)],
                  fill=(255, 150, 160))

    # ---- нос и рот
    d.polygon([(hx - S(14), eye_y + S(44)), (hx + S(14), eye_y + S(44)),
               (hx, eye_y + S(62))], fill=(255, 120, 130))
    if mode == "float":
        d.ellipse([hx - S(16), eye_y + S(56), hx + S(16), eye_y + S(84)],
                  fill=(60, 40, 40))
    elif mode == "fall":
        d.ellipse([hx - S(20), eye_y + S(60), hx + S(20), eye_y + S(92)],
                  fill=(60, 40, 40))
    else:
        mw = S(26)
        d.arc([hx - mw, eye_y + S(52), hx - S(4), eye_y + S(82)], 0, 140,
              fill=(60, 40, 40), width=S(8))
        d.arc([hx + S(4), eye_y + S(52), hx + mw, eye_y + S(82)], 40, 180,
              fill=(60, 40, 40), width=S(8))

    # ---- усы
    for sgn in (-1, 1):
        for dy in (-S(10), S(6), S(22)):
            d.line([hx + sgn * S(60), eye_y + S(52) + dy,
                    hx + sgn * S(150), eye_y + S(40) + dy * 1.6],
                   fill=(90, 60, 50), width=S(6))

    # ---- передние лапы
    if mode == "float":
        # раскинуты в стороны, шевелятся
        for sgn in (-1, 1):
            ang = -0.9 * sgn + math.sin(t * 5 + sgn) * 0.25
            px_ = hx + sgn * S(140) + math.cos(ang + math.pi / 2 * sgn) * 0
            py_ = hy + S(90)
            ex_ = px_ + sgn * S(90) * math.cos(0.5)
            ey_ = py_ - S(70) * math.sin(0.5) + math.sin(t * 5 + sgn) * S(16)
            d.line([px_, py_, ex_, ey_], fill=(255, 184, 108), width=S(42))
            d.ellipse([ex_ - S(34), ey_ - S(34), ex_ + S(34), ey_ + S(34)],
                      fill=(255, 184, 108))
    elif wave_t > 0:
        ang = math.sin(t * 10) * 0.5
        sxp, syp = hx + S(150), hy + S(40)
        px_ = sxp + math.cos(-1.0 + ang) * S(110)
        py_ = syp + math.sin(-1.0 + ang) * S(110)
        d.line([sxp, syp, px_, py_], fill=(255, 184, 108), width=S(44))
        d.ellipse([px_ - S(38), py_ - S(38), px_ + S(38), py_ + S(38)],
                  fill=(255, 184, 108))


# ---------------------------------------------------------------- предметы

def draw_bowl(d, cx, cy, rot=0.0, t=0.0):
    d.polygon([(cx - S(70), cy - S(30)), (cx + S(70), cy - S(30)),
               (cx + S(50), cy + S(40)), (cx - S(50), cy + S(40))],
              fill=(230, 90, 90))
    d.ellipse([cx - S(70), cy - S(45), cx + S(70), cy - S(5)], fill=(255, 130, 130))
    for i in range(3):
        draw_fish_food(d, cx - S(30) + i * S(30), cy - S(55))


def draw_fish_food(d, cx, cy):
    r = S(9)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(150, 100, 60))


def draw_fish(d, cx, cy, rot=0.0, t=0.0):
    wag = math.sin(t * 8) * S(10)
    d.ellipse([cx - S(60), cy - S(32), cx + S(50), cy + S(32)], fill=(90, 170, 230))
    d.polygon([(cx + S(45), cy), (cx + S(90), cy - S(35) + wag),
               (cx + S(90), cy + S(35) + wag)], fill=(90, 170, 230))
    d.ellipse([cx - S(40), cy - S(14), cx - S(18), cy + S(8)], fill=(255, 255, 255))
    d.ellipse([cx - S(34), cy - S(8), cx - S(22), cy + S(4)], fill=(40, 40, 60))


def draw_bone(d, cx, cy, rot=0.0, t=0.0):
    d.line([cx - S(50), cy, cx + S(50), cy], fill=(250, 245, 230), width=S(24))
    for sgn in (-1, 1):
        for dy in (-1, 1):
            d.ellipse([cx + sgn * S(50) - S(22), cy + dy * S(14) - S(22),
                       cx + sgn * S(50) + S(22), cy + dy * S(14) + S(22)],
                      fill=(250, 245, 230))


def draw_flowerpot(d, cx, cy, rot=0.0, t=0.0):
    d.polygon([(cx - S(45), cy - S(20)), (cx + S(45), cy - S(20)),
               (cx + S(32), cy + S(50)), (cx - S(32), cy + S(50))],
              fill=(210, 120, 80))
    d.rectangle([cx - S(50), cy - S(38), cx + S(50), cy - S(12)], fill=(230, 140, 95))
    d.line([cx, cy - S(38), cx, cy - S(100)], fill=(90, 170, 90), width=S(10))
    for ang in (2.4, 3.9):
        px_ = cx + math.cos(ang) * S(45)
        py_ = cy - S(70) + math.sin(ang) * S(45)
        d.ellipse([px_ - S(26), py_ - S(16), px_ + S(26), py_ + S(16)],
                  fill=(110, 190, 110))
    draw_heart_flower(d, cx, cy - S(115))


def draw_heart_flower(d, cx, cy):
    r = S(18)
    d.ellipse([cx - r, cy - r, cx, cy], fill=(255, 100, 130))
    d.ellipse([cx, cy - r, cx + r, cy], fill=(255, 100, 130))
    d.polygon([(cx - r, cy - r * 0.15), (cx + r, cy - r * 0.15), (cx, cy + r * 1.1)],
              fill=(255, 100, 130))


def draw_book(d, cx, cy, rot=0.0, t=0.0):
    d.rectangle([cx - S(55), cy - S(40), cx + S(55), cy + S(40)], fill=(150, 100, 220))
    d.rectangle([cx - S(45), cy - S(32), cx + S(45), cy + S(32)], fill=(255, 255, 255))
    d.line([cx, cy - S(32), cx, cy + S(32)], fill=(220, 220, 230), width=S(5))


# ---------------------------------------------------------------- эффекты

def draw_star(d, cx, cy, r, rot, color=(255, 220, 90)):
    pts = []
    for i in range(10):
        ang = rot + i * math.pi / 5 - math.pi / 2
        rr = r if i % 2 == 0 else r * 0.45
        pts.append((cx + math.cos(ang) * rr, cy + math.sin(ang) * rr))
    d.polygon(pts, fill=color)


def draw_qmark(d, cx, cy, size, color):
    font = get_font(S(size))
    d.text((cx, cy), "?", font=font, fill=color, anchor="mm",
           stroke_width=S(4), stroke_fill=(255, 255, 255))


def draw_impact(d, cx, cy, p):
    """«Бум» при падении."""
    r = S(40 + 160 * p)
    for i in range(8):
        ang = i * math.pi / 4 + 0.3
        x1 = cx + math.cos(ang) * r * 0.7
        y1 = cy + math.sin(ang) * r * 0.5
        x2 = cx + math.cos(ang) * r
        y2 = cy + math.sin(ang) * r * 0.75
        d.line([x1, y1, x2, y2], fill=(255, 200, 60), width=S(16))


def draw_sweat_drop(d, cx, cy):
    d.polygon([(cx, cy - S(30)), (cx - S(16), cy + S(6)), (cx + S(16), cy + S(6))],
              fill=(140, 200, 255))


# ---------------------------------------------------------------- текст

_font_cache = {}

def get_font(size, bold=True):
    size = max(1, int(size))
    key = (size, bold)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(FONT_BOLD, size)
    return _font_cache[key]


def draw_text_center(d, text, cy, size, fill, stroke, scale=1.0, y_off=0):
    font = get_font(S(int(size * scale)))
    d.text((DW // 2, S(cy) + S(y_off)), text, font=font, fill=fill,
           stroke_width=S(6), stroke_fill=stroke, anchor="mm")


# ---------------------------------------------------------------- музыка

def make_music(path, duration):
    sr = 44100
    n = int(sr * duration)
    t = np.arange(n) / sr
    audio = np.zeros(n)

    bpm = 126
    beat = 60.0 / bpm

    melody_notes = [523.25, 587.33, 659.25, 783.99, 880.00,
                    783.99, 659.25, 587.33, 523.25, 659.25]
    bass_notes = [130.81, 130.81, 174.61, 196.00]

    def tone(freq, start, dur, amp, kind="square"):
        i0 = int(start * sr)
        i1 = min(int((start + dur) * sr), n)
        if i0 >= n:
            return
        tt = t[i0:i1] - start
        if kind == "square":
            sig = np.sign(np.sin(2 * np.pi * freq * tt)) * 0.6 + \
                  np.sin(2 * np.pi * freq * tt) * 0.4
        else:
            sig = np.sin(2 * np.pi * freq * tt)
        env = np.exp(-tt * 5.0) * (tt < dur)
        audio[i0:i1] += sig * env * amp

    def whoosh(start, dur=0.8, amp=0.25, up=True):
        i0 = int(start * sr)
        i1 = min(int((start + dur) * sr), n)
        if i0 >= n:
            return
        m = i1 - i0
        rng = np.random.default_rng(int(start * 100))
        noise = rng.uniform(-1, 1, m)
        env = np.hanning(m) * amp
        audio[i0:i1] += noise * env

    def thud(start, amp=0.5):
        i0 = int(start * sr)
        i1 = min(i0 + int(0.3 * sr), n)
        if i0 >= n:
            return
        tt = t[i0:i1] - start
        sig = np.sin(2 * np.pi * (60 * np.exp(-tt * 8)) * tt)
        audio[i0:i1] += sig * np.exp(-tt * 12) * amp

    n_beats = int(duration / beat)
    for b in range(n_beats):
        start = b * beat
        tone(melody_notes[b % len(melody_notes)], start, beat * 0.9, 0.15)
        if b % 2 == 0:
            tone(bass_notes[(b // 2) % len(bass_notes)], start, beat * 1.6, 0.18, "sine")
        i0 = int(start * sr)
        i1 = min(i0 + int(0.05 * sr), n)
        if i0 < n:
            rng = np.random.default_rng(b)
            audio[i0:i1] += rng.uniform(-1, 1, i1 - i0) * \
                np.exp(-np.arange(i1 - i0) / sr * 60) * 0.06

    whoosh(T_GRAV_OFF - 0.2)          # взлёт
    whoosh(T_GRAV_ON - 0.4, up=False) # падение
    thud(T_GRAV_ON + 0.55)            # приземление

    audio = audio / max(1e-9, np.max(np.abs(audio))) * 0.7
    stereo = np.stack([audio, audio], axis=1)
    pcm = (stereo * 32767).astype(np.int16)

    with wave.open(path, "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- физика сцены

GRAVITY = 2600.0  # px/s^2 в координатах финального кадра


def object_height(base_y, t, wobble_seed):
    """Высота предмета над своим местом с учётом «отключения» гравитации."""
    if t < T_GRAV_OFF:
        return 0.0
    if t < T_GRAV_ON:
        ft = t - T_GRAV_OFF
        rise = ease_out_cubic(clamp01(ft / 1.6)) * 620
        wob = math.sin(t * 2.2 + wobble_seed) * 24 + math.sin(t * 1.1) * 14
        return rise + wob
    ft = t - T_GRAV_ON
    fall = ease_in_cubic(clamp01(ft / 0.55)) * 620
    if ft >= 0.55:
        return 0.0
    return 620 - fall


def object_extra_after_land(t):
    """Отскок после приземления."""
    ft = t - T_GRAV_ON
    if ft < 0.55:
        return 0.0
    ft2 = ft - 0.55
    if ft2 < 0.5:
        return -abs(math.sin(ft2 * math.pi / 0.5)) * 40 * (1 - ft2 * 2)
    return 0.0


def cat_state(t):
    """(y, mode, tilt) для котика."""
    ground_y = 1450.0
    if t < T_GRAV_OFF:
        return ground_y, "idle", 0.0
    if t < T_GRAV_ON:
        ft = t - T_GRAV_OFF
        rise = ease_out_cubic(clamp01(ft / 1.4)) * 480
        y = ground_y - rise + math.sin(t * 2.0) * 22 + math.sin(t * 0.9) * 14
        tilt = math.sin(t * 1.6) * 0.5
        return y, "float", tilt
    ft = t - T_GRAV_ON
    if ft < 0.55:
        y = ground_y - 480 + ease_in_cubic(ft / 0.55) * 480
        return y, "fall", 0.0
    if ft < 0.95:
        sq = 1 - abs(math.sin((ft - 0.55) * math.pi / 0.4)) * 0.18
        return ground_y, "landed", (1 - sq) * 4
    return ground_y, "idle", 0.0


# ---------------------------------------------------------------- кадры

def render_frame(idx):
    t = idx / FPS
    img = BG.copy()
    d = ImageDraw.Draw(img)

    # интенсивность «антигравитации» — для неба и звёзд
    ag = 0.0
    if T_GRAV_OFF <= t < T_GRAV_ON:
        ag = clamp01((t - T_GRAV_OFF) / 0.8)
    elif t >= T_GRAV_ON:
        ag = clamp01(1 - (t - T_GRAV_ON) / 0.4)

    if ag > 0:
        draw_stars_sky(d, t, ag)
    draw_sun(d, t)
    cloud_lift = object_height(0, t, 0.0) * 0.4
    draw_cloud(d, 260, 430 - cloud_lift, 1.2)
    draw_cloud(d, 760, 620 - cloud_lift * 0.7, 0.9)
    draw_ground(d)

    cat_y, mode, tilt = cat_state(t)
    cat_y += object_extra_after_land(t) * 0.4

    # ---- летающие предметы
    items = [
        (draw_bowl,      200, 1520, 0.5),
        (draw_fish,      880, 1500, 1.9),
        (draw_bone,      130, 1480, 3.1),
        (draw_flowerpot, 950, 1520, 4.4),
        (draw_book,      320, 1500, 5.6),
    ]
    for fn, bx, by, seed in items:
        dy = object_height(by, t, seed) + object_extra_after_land(t)
        fn(d, S(bx), S(by - dy), 0.0, t)

    # ---- котик
    draw_cat(d, t, 540, S(cat_y), mode=mode, tilt=tilt)

    # ---- вопросики в начале
    if 1.4 <= t < 3.0:
        for i, (qx, qy, qs) in enumerate([(180, 900, 70), (880, 850, 90), (300, 760, 55)]):
            p = ease_out_back(clamp01((t - 1.4 - i * 0.15) / 0.4))
            fade = clamp01((3.0 - t) / 0.3)
            if p > 0.01 and fade > 0.01:
                wob = math.sin(t * 4 + i) * 12
                qy2 = qy - (qy - 500) * (1 - p) + wob
                col = (int(120 + 60 * i), 80, 200)
                d.text((S(qx), S(qy2)), "?", font=get_font(S(int(qs * p))),
                       fill=col, anchor="mm", stroke_width=S(5),
                       stroke_fill=(255, 255, 255))

    # ---- звёзды-эффект при приземлении
    if T_GRAV_ON + 0.4 <= t < T_GRAV_ON + 1.2:
        p = clamp01((t - (T_GRAV_ON + 0.4)) / 0.8)
        draw_impact(d, S(540), S(1430), 1 - p)
        for i in range(6):
            ang = i * math.pi / 3 + 0.4
            r = 100 + p * 260
            draw_star(d, S(540 + math.cos(ang) * r),
                      S(1400 + math.sin(ang) * r * 0.5),
                      S(int(30 * (1 - p))), t * 4 + i)

    # ---- искры восторга в полёте
    if T_GRAV_OFF + 0.6 <= t < T_GRAV_ON - 0.3:
        for i in range(5):
            ft = ((t - T_GRAV_OFF) * 0.6 + i * 0.2) % 1.0
            sx_ = 180 + i * 190 + math.sin(ft * 6 + i * 2) * 40
            sy_ = 1300 - ft * 500
            draw_star(d, S(sx_), S(sy_), S(int(20 * (1 - ft * 0.5))), t * 3 + i,
                      color=(255, 230, 120))

    # ---- титры
    if t < 1.5:
        p = ease_out_back(clamp01(t / 0.6))
        draw_text_center(d, "ЧТО ЕСЛИ...", 240, 110, (255, 255, 255), (90, 60, 160),
                         scale=p)
    elif t < 3.1:
        p = ease_out_back(clamp01((t - 1.6) / 0.5))
        draw_text_center(d, "гравитация", 240, 78, (255, 255, 255), (90, 60, 160),
                         scale=p)
        draw_text_center(d, "исчезнет на 5 секунд?", 330, 62, (255, 240, 150),
                         (90, 60, 160), scale=p)
    elif t < 5.2:
        p = ease_out_back(clamp01((t - 3.2) / 0.5))
        draw_text_center(d, "Ух ты, мы летим!", 620, 64, (60, 60, 90), (255, 255, 255),
                         scale=p)
    elif t < 7.9:
        p = ease_out_back(clamp01((t - 5.4) / 0.5))
        draw_text_center(d, "Это потрясающе!", 620, 64, (90, 70, 180), (255, 255, 255),
                         scale=p)
    elif t < 9.6:
        p = ease_out_back(clamp01((t - 8.1) / 0.4))
        shake = math.sin(t * 40) * 6 * (1 - clamp01((t - 8.5) / 0.5))
        draw_text_center(d, "БАМ! Гравитация", 620, 66, (230, 60, 60), (255, 255, 255),
                         scale=p, y_off=shake)
        draw_text_center(d, "вернулась!", 700, 66, (230, 60, 60), (255, 255, 255),
                         scale=p, y_off=shake)
    else:
        p = ease_out_back(clamp01((t - 9.8) / 0.5))
        draw_text_center(d, "А ты бы что делал?", 620, 62, (60, 60, 90), (255, 255, 255),
                         scale=p)
        pulse = 1.0 + 0.06 * math.sin(t * 6)
        draw_text_center(d, "Пиши в комментах!", 710, 56, (60, 60, 90), (255, 255, 255),
                         scale=p)
        draw_text_center(d, "ПОДПИШИСЬ", 260, 92, (255, 255, 255), (230, 60, 90),
                         scale=p * pulse)

    frame = img.resize((W, H), Image.LANCZOS)
    return frame


# ---------------------------------------------------------------- сборка

def main():
    global BG
    BG = make_background()

    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    os.makedirs(FRAMES_DIR)

    print(f"Рендер {N_FRAMES} кадров...")
    for i in range(N_FRAMES):
        frame = render_frame(i)
        frame.save(os.path.join(FRAMES_DIR, f"f_{i:04d}.png"))
        if i % 60 == 0:
            print(f"  кадр {i}/{N_FRAMES}")

    print("Генерация музыки...")
    make_music(AUDIO_PATH, DURATION)

    print("Сборка видео (ffmpeg)...")
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg, "-y",
        "-framerate", str(FPS), "-i", os.path.join(FRAMES_DIR, "f_%04d.png"),
        "-i", AUDIO_PATH,
        "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-shortest", "-movflags", "+faststart",
        VIDEO_PATH,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    size_mb = os.path.getsize(VIDEO_PATH) / 1e6
    print(f"Готово: {VIDEO_PATH} ({size_mb:.1f} МБ)")


if __name__ == "__main__":
    main()
