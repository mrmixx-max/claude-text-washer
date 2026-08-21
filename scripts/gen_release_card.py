#!/usr/bin/env python3
"""Generate X release card for Claude Text Washer v1.0 Windows app."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont
from collections import Counter

W, H = 1600, 900
OUT = r"C:\Users\webma\Downloads\claude-washer-release-card.png"

# Colors
BG      = (17, 17, 24)
GOLD    = (201, 168, 76)
WHITE   = (240, 237, 228)
MUTED   = (130, 135, 150)
GREEN   = (0, 230, 118)
RED     = (233, 69, 96)
TEAL    = (0, 180, 216)

FONT_DIR = "C:/Windows/Fonts"

def load(name: str, size: int):
    try:
        return ImageFont.truetype(FONT_DIR + "/" + name, size)
    except:
        return ImageFont.load_default()

# Fonts
f_title    = load("calibrib.ttf", 52)
f_subtitle = load("calibri.ttf", 28)
f_tag      = load("calibri.ttf", 20)
f_feature  = load("calibri.ttf", 22)
f_small    = load("calibri.ttf", 18)
f_stat     = load("calibrib.ttf", 36)
f_stat_l   = load("calibri.ttf", 16)

img = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img)

# --- Background grid ---
for x in range(0, W, 60):
    draw.line([(x, 0), (x, H)], fill=(30, 30, 40), width=1)
for y in range(0, H, 60):
    draw.line([(0, y), (W, y)], fill=(30, 30, 40), width=1)

# --- Glow orb top-right ---
for r in range(400, 0, -2):
    alpha = int(30 * (1 - r / 400))
    color = (0, 180, 216, alpha)
    draw.ellipse([W - 500 + r, -200 + r, W - 500 + r + 4, -200 + r + 4], fill=(0, 30, 40))

# --- Tag pill ---
draw.rounded_rectangle([80, 40, 260, 80], radius=20, fill=(0, 180, 216))
draw.text((98, 47), "v1.0 RELEASE", fill=(17, 17, 24), font=f_tag)

# --- Title ---
draw.text((80, 120), "Claude Text Washer", fill=WHITE, font=f_title)
x_after = 80 + draw.textlength("Claude Text Washer", font=f_title)
draw.text((x_after, 120), " for Windows", fill=GOLD, font=f_title)

# --- Subtitle ---
draw.text((80, 190), "Strip AI watermarks. Rewrite in organic prose.", fill=MUTED, font=f_subtitle)

# --- Feature list (left column) ---
features = [
    ("✓", "Statistical AI detection (green-list, burstiness, entropy)"),
    ("✓", "Multi-pass rewrite via local LLM"),
    ("✓", "Any LLM backend (Ollama, vLLM, OpenRouter)"),
    ("✓", "Sampling controls (temp, top-p, max tokens)"),
    ("✓", "Anti-watermark prompt generator"),
    ("✓", "Free + open source (MIT)"),
]

y = 260
for icon, text in features:
    draw.text((80, y), icon, fill=GREEN, font=f_feature)
    draw.text((115, y), text, fill=WHITE, font=f_feature)
    y += 42

# --- Stats row ---
stats = [
    ("3", "Tabs"),
    ("8.2", "MB EXE"),
    ("235+", "Tests"),
    ("MIT", "License"),
]

x_stat = 80
y_stat = 520
for val, label in stats:
    draw.text((x_stat, y_stat), val, fill=GOLD, font=f_stat)
    draw.text((x_stat, y_stat + 42), label, fill=MUTED, font=f_stat_l)
    x_stat += 180

# --- Comparison box (right side) ---
# Box background
draw.rounded_rectangle([850, 120, 1520, 420], radius=16, fill=(25, 25, 35), outline=GOLD, width=2)
draw.text((880, 140), "Before vs After", fill=WHITE, font=f_subtitle)

# Before (red)
draw.rounded_rectangle([880, 190, 1180, 260], radius=8, fill=(40, 25, 30), outline=RED, width=1)
draw.text((900, 198), "Generic cleaner:", fill=RED, font=f_small)
draw.text((900, 222), "Removes invisible Unicode only", fill=MUTED, font=f_small)

# After (green)
draw.rounded_rectangle([880, 275, 1480, 345], radius=8, fill=(25, 40, 30), outline=GREEN, width=1)
draw.text((900, 283), "Claude Text Washer:", fill=GREEN, font=f_small)
draw.text((900, 307), "Detects + removes statistical AI markers", fill=MUTED, font=f_small)

# Arrow
draw.text((1195, 220), "→", fill=GOLD, font=f_subtitle)

# --- Score box ---
draw.rounded_rectangle([850, 440, 1520, 560], radius=16, fill=(25, 25, 35), outline=TEAL, width=1)
draw.text((880, 455), "AI Score reduction", fill=MUTED, font=f_small)
draw.text((880, 480), "87 → 12", fill=GOLD, font=f_stat)
draw.text((1050, 490), "typical after 2 passes", fill=MUTED, font=f_small)

# --- CTA bar ---
draw.rounded_rectangle([850, 580, 1520, 640], radius=12, fill=(0, 180, 216))
draw.text((920, 595), "github.com/mrmixx-max/claude-text-washer", fill=(17, 17, 24), font=f_feature)

# --- Author bar ---
draw.rounded_rectangle([80, 800, 1520, 870], radius=0, fill=(25, 25, 35))
draw.text((110, 818), "@webma", fill=WHITE, font=f_feature)
draw.text((220, 820), "Claude Text Washer v1.0 for Windows", fill=MUTED, font=f_small)

# --- Brand ---
draw.text((1200, 820), "Free. Open Source. Any LLM.", fill=MUTED, font=f_small)

img.save(OUT, "PNG")

# --- Verification ---
img_check = Image.open(OUT).convert("RGB")
pix = [img_check.getpixel((x, y)) for y in range(0, H, 4) for x in range(0, W, 4)]
c = Counter(pix)
print(f"Card saved: {OUT}")
print(f"Unique colors: {len(c)}")
print(f"Size: {img_check.size}")
# Check key colors present
def near(col, target, tol=30):
    return all(abs(a - b) <= tol for a, b in zip(col, target))
for name, t in [("Gold", GOLD), ("Green", GREEN), ("Teal", TEAL), ("White", WHITE)]:
    pct = sum(v for p, v in c.items() if near(p, t)) / sum(c.values()) * 100
    print(f"  {name}: {pct:.1f}%")
