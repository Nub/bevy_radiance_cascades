"""Scores the cascades against the path-traced reference in the Cornell box.

usage: python cornell.py <name> [VAR=value ...]

Renders the `cornell` example with the cascades (and the given environment)
to shots/cl_<name>.png, untonemapped and with a dim panel so that nothing
clips, and prints each region's brightness relative to shots/cl_ref.png, which
is rendered first if it is not there (`python cornell.py ref` renders it
again). 1.00 is the path tracer's.
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, 'target', 'debug', 'examples', 'cornell.exe')
PANEL = '0.25'
# (row, column) in a 960 x 960 view.
REGIONS = {
    'floor front': (850, 480), 'floor left': (840, 200), 'floor by short box': (800, 740),
    'back wall mid': (350, 600), 'back wall top': (260, 480), 'back wall low': (640, 620),
    'left wall': (480, 120), 'right wall': (480, 840), 'ceiling': (100, 300), 'ceiling by panel': (150, 640),
    'tall box front': (600, 400), 'tall box side': (600, 297), 'tall box top shadow': (700, 250),
    'short box top': (615, 590), 'short box front': (740, 580), 'short box shadow': (790, 700),
}


def render(name, env):
    path = os.path.join(HERE, 'shots', 'cl_%s.png' % name)
    if os.path.exists(path):
        os.remove(path)
    full = dict(os.environ, RC_LINEAR='1', RC_PANEL=PANEL, RC_SHOT=path)
    full.update(env)
    out = subprocess.run([EXE], cwd=HERE, env=full, capture_output=True, text=True, timeout=400)
    for line in (out.stdout + out.stderr).splitlines():
        if ('ERROR' in line or 'panic' in line) and 'gilrs' not in line:
            print(line)
    return path


def linear(path):
    a = np.asarray(Image.open(path).convert('RGB')).astype(np.float64) / 255
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def main():
    name = sys.argv[1]
    env = dict(a.split('=', 1) for a in sys.argv[2:])
    # (A reference per view, `RC_VIEW`.)
    view = env.get('RC_VIEW', '')
    ref_name = 'ref' + ('_' + view if view else '')
    ref_path = os.path.join(HERE, 'shots', 'cl_%s.png' % ref_name)
    if name == 'ref' or not os.path.exists(ref_path):
        render(ref_name, dict({'RC_MODE': 'reference'}, **({'RC_VIEW': view} if view else {})))
        if name == 'ref':
            return
    env.setdefault('RC_MODE', 'on')
    got = linear(render(name, env))
    ref = linear(ref_path)
    if env.get('RC_BOUNCE') == '0':
        # One bounce: against the panel's light worked out exactly.
        from cornell_direct import direct
        ratios = []
        for region, (y, x) in REGIONS.items():
            r = direct(x, y, panel=float(env.get('RC_PANEL', PANEL))).mean()
            v = got[y - 8:y + 8, x - 8:x + 8].mean()
            if r > 1e-4:
                ratios.append(v / r)
                print('%-20s direct %.3f  %5.2f' % (region, r, ratios[-1]))
            else:
                print('%-20s direct 0      (%.3f)' % (region, v))
        print('%-20s one bounce: mean %.2f  log spread %.2f' % (name, np.mean(ratios), np.std(np.log(np.maximum(ratios, 1e-3)))))
        return
    ratios = []
    for region, (y, x) in ({} if view else REGIONS).items():
        r = ref[y - 8:y + 8, x - 8:x + 8].mean()
        v = got[y - 8:y + 8, x - 8:x + 8].mean()
        ratios.append(v / max(r, 1e-6))
        print('%-20s ref %.3f  %5.2f' % (region, r, ratios[-1]))
    # Over the whole box (not the black around it, nor the panel).
    inside = (ref.mean(2) > 0.004) & (ref.max(2) < 0.95)
    whole = got[inside].mean() / ref[inside].mean()
    error = np.abs(got - ref)[inside].mean() / ref[inside].mean()
    print('%-20s whole %.2f  mean abs error %.2f' % (name, whole, error))
    # Side by side, brightened: the reference, this, and this over it (blue
    # darker, red brighter, up to twice).
    r = np.where(inside, got.mean(2) / np.maximum(ref.mean(2), 1e-6), 1.0)
    lo, hi = np.clip(r, 0, 1), np.clip(r - 1, 0, 1)
    colour = np.stack([np.where(r > 1, 1, lo), np.where(r > 1, 1 - hi, lo), np.where(r > 1, 1 - hi, 1)], -1)
    show = lambda x: (np.clip(x * 3, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    sheet = np.concatenate([show(ref), show(got), (colour * 255).astype(np.uint8)], 1)
    Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(os.path.join(HERE, 'shots', 'cs_%s.png' % name))


main()
