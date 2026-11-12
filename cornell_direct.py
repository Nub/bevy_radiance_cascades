"""The light the panel of the `cornell` example sends straight to a point of
the image (no bounces), worked out exactly: what the cascades should show there
with `RC_BOUNCE=0`. Used by cornell.py."""
import numpy as np

PANEL = np.array([34000.0, 28000.0, 19000.0])
PANEL_SIDE = 1.2
EXPOSURE = 1.0 / (2.0 ** 9.7 * 1.2)
WHITE, RED, GREEN = (0.73, 0.73, 0.73), (0.65, 0.05, 0.05), (0.12, 0.45, 0.15)
# size, centre, turn about y, colour: as in examples/cornell.rs.
BOXES = [
    ((4.4, 0.2, 4.4), (0.0, -0.1, 0.0), 0.0, WHITE),
    ((4.4, 0.2, 4.4), (0.0, 4.1, 0.0), 0.0, WHITE),
    ((4.4, 4.4, 0.2), (0.0, 2.0, -2.1), 0.0, WHITE),
    ((0.2, 4.4, 4.4), (-2.1, 2.0, 0.0), 0.0, RED),
    ((0.2, 4.4, 4.4), (2.1, 2.0, 0.0), 0.0, GREEN),
    ((1.2, 2.4, 1.2), (-0.65, 1.2, -0.6), 0.3, WHITE),
    ((1.2, 1.2, 1.2), (0.7, 0.6, 0.6), -0.3, WHITE),
]
EYE = np.array([0.0, 2.0, 7.3])


def to_local(v, turn):
    c, s = np.cos(turn), np.sin(turn)
    # The inverse of a turn about y.
    return np.array([c * v[0] - s * v[2], v[1], s * v[0] + c * v[2]])


def from_local(v, turn):
    return to_local(v, -turn)


def hit(origin, direction, reach=1e9):
    """Nearest box along a ray: (distance, normal, colour) or None."""
    best = None
    for size, centre, turn, colour in BOXES:
        o = to_local(origin - np.array(centre), turn)
        d = to_local(direction, turn)
        half = np.array(size) / 2
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = (-half - o) / d
            t2 = (half - o) / d
        near, far = np.minimum(t1, t2), np.maximum(t1, t2)
        near = np.where(np.isnan(near), -np.inf, near)
        far = np.where(np.isnan(far), np.inf, far)
        enter, leave = near.max(), far.min()
        if enter <= leave and 1e-4 < enter < reach and (best is None or enter < best[0]):
            axis = int(near.argmax())
            normal = np.zeros(3)
            normal[axis] = -np.sign(d[axis])
            best = (enter, from_local(normal, turn), np.array(colour))
    return best


def direct(x, y, size=960, panel=0.25, samples=24):
    """Linear colour at pixel (x, y) of a `size` square view."""
    tan = np.tan(np.pi / 8)
    ray = np.array([(2 * (x + 0.5) / size - 1) * tan, (1 - 2 * (y + 0.5) / size) * tan, -1.0])
    ray /= np.linalg.norm(ray)
    first = hit(EYE, ray)
    if first is None:
        return np.zeros(3)
    distance, normal, colour = first
    point = EYE + ray * distance + normal * 1e-3
    total = 0.0
    for i in range(samples):
        for j in range(samples):
            on = np.array([((i + 0.5) / samples - 0.5) * PANEL_SIDE, 3.99, ((j + 0.5) / samples - 0.5) * PANEL_SIDE])
            to = on - point
            r2 = to @ to
            to /= np.sqrt(r2)
            cosines = max(normal @ to, 0.0) * max(to[1], 0.0)
            if cosines > 0.0 and hit(point, to, np.sqrt(r2) - 2e-3) is None:
                total += cosines / r2
    irradiance = total * PANEL_SIDE ** 2 / samples ** 2
    return colour / np.pi * irradiance * PANEL * panel * EXPOSURE


def _hit_many(origins, directions, reach):
    """Vectorised `hit`: (distance, normal, colour) of the nearest box per ray."""
    n = len(origins)
    best = np.full(n, np.inf)
    normals = np.zeros((n, 3))
    colours = np.zeros((n, 3))
    for size, centre, turn, colour in BOXES:
        c, s = np.cos(turn), np.sin(turn)
        rot = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        o = (origins - np.array(centre)) @ rot.T
        d = directions @ rot.T
        half = np.array(size) / 2
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = (-half - o) / d
            t2 = (half - o) / d
        near = np.nan_to_num(np.minimum(t1, t2), nan=-np.inf)
        far = np.nan_to_num(np.maximum(t1, t2), nan=np.inf)
        enter, leave = near.max(1), far.min(1)
        ok = (enter <= leave) & (enter > 1e-4) & (enter < reach) & (enter < best)
        axis = near.argmax(1)
        local = np.zeros((n, 3))
        local[np.arange(n), axis] = -np.sign(d[np.arange(n), axis])
        best = np.where(ok, enter, best)
        normals[ok] = (local @ rot)[ok]
        colours[ok] = colour
    return best, normals, colours


def direct_image(size=240, panel=0.25, samples=12):
    """The whole view, `size` square: linear colour per pixel."""
    tan = np.tan(np.pi / 8)
    ys, xs = np.mgrid[0:size, 0:size]
    rays = np.stack([(2 * (xs + 0.5) / size - 1) * tan, (1 - 2 * (ys + 0.5) / size) * tan, -np.ones_like(xs, dtype=float)], -1).reshape(-1, 3)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    eyes = np.tile(EYE, (len(rays), 1))
    distance, normals, colours = _hit_many(eyes, rays, np.inf)
    seen = np.isfinite(distance)
    points = eyes + rays * np.where(seen, distance, 0.0)[:, None] + normals * 1e-3
    total = np.zeros(len(rays))
    for i in range(samples):
        for j in range(samples):
            on = np.array([((i + 0.5) / samples - 0.5) * PANEL_SIDE, 3.99, ((j + 0.5) / samples - 0.5) * PANEL_SIDE])
            to = on - points
            r = np.linalg.norm(to, axis=1)
            to /= r[:, None]
            cosines = np.maximum((normals * to).sum(1), 0.0) * np.maximum(to[:, 1], 0.0)
            blocked, _, _ = _hit_many(points, to, r - 2e-3)
            total += np.where(np.isfinite(blocked), 0.0, cosines / (r * r))
    irradiance = total * PANEL_SIDE ** 2 / samples ** 2
    image = colours / np.pi * irradiance[:, None] * PANEL * panel * EXPOSURE
    image[~seen] = 0.0
    return image.reshape(size, size, 3)


def ratio_sheet(path, out, panel=0.25):
    """Writes `out`: the truth, the image at `path`, and the latter over the
    former (blue: too dark, white: right, red: too bright, up to twice)."""
    from PIL import Image
    import os
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shots', 'direct_%g.npy' % panel)
    if os.path.exists(cache):
        truth = np.load(cache)
    else:
        truth = direct_image(panel=panel)
        np.save(cache, truth)
    a = np.asarray(Image.open(path).convert('RGB')).astype(np.float64) / 255
    a = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    n = truth.shape[0]
    got = a.reshape(n, a.shape[0] // n, n, a.shape[1] // n, 3).mean((1, 3))
    t, g = truth.mean(2), got.mean(2)
    r = np.where(t > 0.004, g / np.maximum(t, 1e-6), 1.0)
    lo, hi = np.clip(r, 0, 1), np.clip(r - 1, 0, 1)
    colour = np.stack([np.where(r > 1, 1, lo), np.where(r > 1, 1 - hi, lo), np.where(r > 1, 1 - hi, 1)], -1)
    show = lambda x: (np.clip(x * 3, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    sheet = np.concatenate([show(truth), show(got), (colour * 255).astype(np.uint8)], 1)
    Image.fromarray(sheet).resize((n * 3 * 2, n * 2)).save(out)
    lit = t > 0.004
    return g[lit].mean() / t[lit].mean(), np.abs(g - t)[lit].mean() / t[lit].mean()
