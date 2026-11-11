"""Compares two captures of (nearly) the same view. usage: stab.py a.png b.png [max_shift]"""
import sys
from PIL import Image, ImageChops, ImageStat
a = Image.open(sys.argv[1]).convert('RGB')
b = Image.open(sys.argv[2]).convert('RGB')
m = int(sys.argv[3]) if len(sys.argv) > 3 else 0
w, h = a.size
box = (200, 60, w - 200, h - 60)
ca = a.crop(box)
best = None
for s in range(-m, m + 1):
    d = ImageChops.difference(ca, b.crop((box[0] + s, box[1], box[2] + s, box[3])))
    v = sum(ImageStat.Stat(d).mean) / 3
    if best is None or v < best[0]:
        best = (v, s, d)
v, s, d = best
hist = d.convert('L').histogram()
total = sum(hist)
def pct(p):
    acc = 0
    for value, count in enumerate(hist):
        acc += count
        if acc >= total * p:
            return value
print(f'{sys.argv[1]} vs {sys.argv[2]}: shift {s}px, mean {v:.2f}/255, p99 {pct(0.99)}, p99.9 {pct(0.999)}, max {pct(1.0)}')
d.point(lambda x: min(255, x * 8)).save('shots/diff.png')
