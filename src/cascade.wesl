//! Traces one cascade's radiance interval, against the depth buffer and then
//! the world cache, and merges it with the (already merged) cascade above.
//! Dispatched from the top cascade down.

import package::common::{
    params, view, emitters, oct_texel, PI, specular_history, depth_tex, gbuffer_tex, scene_color, history, cascade_in, linear_sampler, output,
    depth_pyramid, color_pyramid, voxels, trace_voxels, MISS,
    oct_dir, view_size_f, linear_depth, pixel_footprint, ndc_to_pixel, world_position,
    gbuffer_normal, gbuffer_valid, gbuffer_emissive, surface_normal, probe_pixel, probe_weights, probe_index, MAX_RADIANCE,
};
import bevy_pbr::render::rgb9e5::rgb9e5_to_vec3_;

struct Trace {
    /// rgb = radiance, a = transmittance.
    result: vec4<f32>,
    /// The ray went where the screen cannot say what is there: off its edge,
    /// or behind something.
    unknown: bool,
    /// How far along the ray it was stopped, if it was.
    distance: f32,
    /// How far along the ray the screen first could not say what is there.
    unknown_from: f32,
}

/// Where no part of a ray is unknown.
const KNOWN: f32 = 1.0e30;

/// How far along a ray a point of its line on screen is. `u` is how far the
/// point is along the line, which is not linear in distance, though the
/// clip-space segment (of depths `w0` to `w1`, `ta` to `tb` along the ray) it
/// came from is.
fn ray_distance(u: f32, w0: f32, w1: f32, ta: f32, tb: f32) -> f32 {
    let k = u * w0 / max(w1 - u * (w1 - w0), 1e-12);
    return mix(ta, tb, clamp(k, 0.0, 1.0));
}

/// How far past its own interval a cascade keeps tracing, in interval lengths.
/// What it hits out there it shares with the cascade above, which sees the same
/// place: more of its own the nearer, more of the other's the farther. The two
/// see small lights differently (more or fewer rays find them), and where one
/// simply took over from the other that showed as an edge in the lighting.
const FADE: f32 = 1.0;

/// The lit scene around a pixel, as the camera sees it. `width` is how many
/// pixels across the ray's cone is where it hits.
fn scene_radiance(pix: vec2<i32>, pixel: vec2<f32>, width: f32) -> vec3<f32> {
    // Level 0 of the color pyramid is two pixels a texel.
    let lod = clamp(log2(max(width, 1.0)) - 1.0, 0.0, params.color_levels - 1.0);
    let filtered = textureSampleLevel(color_pyramid, linear_sampler, pixel * 0.5 / params.color_size, lod);
    if filtered.a > 0.05 {
        return filtered.rgb / filtered.a;
    }
    return textureLoad(scene_color, pix, 0).rgb;
}

/// Last frame's indirect light leaving the surface at a pixel: adding it is
/// what makes the bounces accumulate over frames instead of stopping at one.
fn bounced_radiance(pix: vec2<i32>, depth: f32) -> vec3<f32> {
    if params.bounce <= 0.0 {
        return vec3<f32>(0.0);
    }
    let p = world_position(vec2<f32>(pix) + 0.5, depth);
    let prev = params.prev_clip_from_world * vec4<f32>(p, 1.0);
    if prev.w <= 0.0 {
        return vec3<f32>(0.0);
    }
    let uv = prev.xy / prev.w * vec2<f32>(0.5, -0.5) + 0.5;
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        return vec3<f32>(0.0);
    }
    // (And what it mirrored towards the eye, where that is kept apart: taken
    // for what it sends every way, which is all a metal bounces.)
    var left = textureSampleLevel(history, linear_sampler, uv, 0.0).rgb;
    if params.specular_separate != 0u {
        left += textureSampleLevel(specular_history, linear_sampler, uv, 0.0).rgb;
    }
    return params.bounce * voxels.feedback_scale * left;
}

/// What the ray finds at the pixel where it meets the depth buffer.
fn shade_hit(pix: vec2<i32>, pixel: vec2<f32>, width: f32, dir: vec3<f32>) -> vec4<f32> {
    let depth = textureLoad(depth_tex, pix, 0);
    let g = textureLoad(gbuffer_tex, pix, 0);
    var radiance = scene_radiance(pix, pixel, width);
    if gbuffer_valid(g) {
        // A ray that arrives at the far side of what the camera sees has hit
        // a side the camera does not: its lamps are unknown, but what glows is
        // assumed to glow all around, and the bounced light to reach there as
        // it reaches the near side. (Bevy keeps emissive in the same exposed
        // units as the lit scene.) The two meet gradually: a hard line between
        // them is a hard line across everything the surface lights.
        // (No more of the glow than the ray's cone sees of the lit scene
        // there: the glow is this one pixel's, and a wide ray that happens on
        // a pixel of a thin glowing trim would take all of it for the trim's
        // brightness, one frame in a few as the view is jittered.)
        let behind = smoothstep(-0.2, 0.2, dot(gbuffer_normal(g), dir));
        radiance = mix(radiance, min(gbuffer_emissive(g), radiance), behind);
    }
    radiance += bounced_radiance(pix, depth);
    // Also swallows NaNs: min/max return the non-NaN operand.
    return vec4<f32>(clamp(radiance, vec3<f32>(0.0), vec3<f32>(MAX_RADIANCE)), 0.0);
}

/// Traces `origin + dir * [t0, t1]` against the depth buffer, exactly: the
/// depth pyramid lets the ray skip whole blocks of pixels it provably passes in
/// front of or behind, and it only ever steps pixel by pixel next to geometry.
fn trace(origin: vec3<f32>, dir: vec3<f32>, t0: f32, t1: f32, probe_pixel: vec2<f32>, spread: f32) -> Trace {
    let near = view.clip_from_view[3][2];

    var c0 = view.clip_from_world * vec4<f32>(origin + dir * t0, 1.0);
    var c1 = view.clip_from_world * vec4<f32>(origin + dir * t1, 1.0);

    // Clip to the near plane (clip w is the linear depth).
    if c0.w < near && c1.w < near {
        return Trace(MISS, true, t1, t0);
    }
    var unknown = c0.w < near || c1.w < near;
    var unknown_from = KNOWN;
    // The distances along the ray of the ends of what is left of it.
    var ta = t0;
    var tb = t1;
    if c0.w < near {
        let k = (c1.w - near) / (c1.w - c0.w);
        c0 = mix(c1, c0, k);
        ta = mix(t1, t0, k);
        unknown_from = t0;
    } else if c1.w < near {
        let k = (c0.w - near) / (c0.w - c1.w);
        c1 = mix(c0, c1, k);
        tb = mix(t0, t1, k);
        unknown_from = tb;
    }

    // NDC xyz is affine along a screen-space line, so from here on the ray is
    // a 2D line with linearly varying NDC depth.
    let n0 = c0.xyz / c0.w;
    let n1 = c1.xyz / c1.w;
    let d = n1 - n0;

    // Clip to the screen. Whatever lies outside is unknown, and treated as empty.
    var s0 = 0.0;
    var s1 = 1.0;
    for (var a = 0u; a < 2u; a++) {
        if abs(d[a]) < 1e-9 {
            if abs(n0[a]) >= 1.0 {
                return Trace(MISS, true, t1, t0);
            }
        } else {
            let a0 = (-1.0 - n0[a]) / d[a];
            let a1 = (1.0 - n0[a]) / d[a];
            s0 = max(s0, min(a0, a1));
            s1 = min(s1, max(a0, a1));
        }
    }
    if s0 >= s1 {
        return Trace(MISS, true, t1, t0);
    }
    unknown = unknown || s0 > 0.0 || s1 < 1.0;
    if s0 > 0.0 {
        // It starts off the screen.
        unknown_from = t0;
    } else if s1 < 1.0 {
        // It leaves the screen.
        unknown_from = min(unknown_from, ray_distance(s1, c0.w, c1.w, ta, tb));
    }

    let p0 = n0 + d * s0;
    let p1 = n0 + d * s1;
    let start = ndc_to_pixel(p0.xy);
    let delta = ndc_to_pixel(p1.xy) - start;
    let len = length(delta);
    let size = vec2<i32>(view_size_f());
    let max_level = i32(params.pyramid_levels) - 1;
    let step_dir = select(delta / len, vec2<f32>(1.0, 0.0), len < 1e-4);
    let inv_dir = 1.0 / select(step_dir, vec2<f32>(1e-12), abs(step_dir) < vec2<f32>(1e-12));
    let inv_len = select(1.0 / len, 0.0, len < 1e-4);

    const EPS: f32 = 0.01;
    var s = 0.0;
    var level = clamp(i32(log2(max(len, 1.0))) - 2, 0, max_level);
    for (var i = 0u; i < params.max_steps; i++) {
        // The cell of the current pyramid level the ray is in, and where it leaves it.
        let p = start + step_dir * min(s + EPS, len);
        let cell_size = f32(1u << u32(level));
        let cell = floor(p / cell_size);
        let exit = (cell + vec2<f32>(step_dir > vec2<f32>(0.0))) * cell_size;
        // (Along an axis the line does not move on, it never leaves.)
        let to_exit = select((exit - p) * inv_dir, vec2<f32>(1.0e30), abs(step_dir) < vec2<f32>(1e-12));
        let s_end = min(s + EPS + max(min(to_exit.x, to_exit.y), 0.0), len);

        let z_a = max(mix(p0.z, p1.z, s * inv_len), 1e-12);
        let z_b = max(mix(p0.z, p1.z, s_end * inv_len), 1e-12);
        let lo = near / max(z_a, z_b);
        let hi = near / min(z_a, z_b);

        let level_size = max(size >> vec2<u32>(u32(level)), vec2<i32>(1));
        let bounds = textureLoad(depth_pyramid, clamp(vec2<i32>(cell), vec2<i32>(0), level_size - 1), level).rg;
        // Nothing in the cell (r is the closest depth, 0 is background), the
        // ray stays in front of all of it, or behind all of it.
        var pass_by = bounds.r <= 0.0;
        if !pass_by {
            pass_by = hi <= near / bounds.r * (1.0 + params.depth_bias);
            if !pass_by && bounds.g > 0.0 && lo >= near / bounds.g + params.thickness {
                pass_by = true;
                if !unknown || unknown_from > t1 {
                    // (Only looked at up to `t1`: later is as good as never.)
                    unknown_from = min(unknown_from, ray_distance(s0 + (s1 - s0) * s * inv_len, c0.w, c1.w, ta, tb));
                }
                unknown = true;
            }
        }

        if pass_by {
            if s_end >= len {
                return Trace(MISS, unknown, t1, unknown_from);
            }
            s = s_end;
            level = min(level + 1, max_level);
        } else if level == 0 {
            // The cone is as wide as the ray is long on screen, times its spread.
            let width = distance(p, probe_pixel) * spread;
            // Where on the ray this is: the screen-space line is not linear in
            // distance, but the clip-space one it came from is.
            let hit_distance = ray_distance(s0 + (s1 - s0) * s * inv_len, c0.w, c1.w, ta, tb);
            return Trace(shade_hit(clamp(vec2<i32>(cell), vec2<i32>(0), size - 1), p, width, dir), false, hit_distance, KNOWN);
        } else {
            level -= 1;
        }
    }
    // Out of steps: nothing is known past where it got to.
    return Trace(MISS, true, t1, min(unknown_from, ray_distance(s0 + (s1 - s0) * s * inv_len, c0.w, c1.w, ta, tb)));
}

/// Radiance of the cascade above, towards the four directions that subdivide
/// `dir_texel`, interpolated between the four probes around this one.
fn merge_upper(pixel: vec2<f32>, position: vec3<f32>, normal: vec3<f32>, footprint: f32, dir_texel: vec2<u32>) -> vec3<f32> {
    let upper_spacing = params.probe_spacing * 2u;
    let upper_res = params.dir_res * 2u;
    // Far light varies slowly from place to place: the higher the cascade, the
    // less a probe on the wrong surface is wrong, and the more a seam shows.
    let min_weight = min(0.02 * f32(1u << params.cascade_index), 0.5);
    let weights = probe_weights(pixel, position, normal, footprint, upper_spacing, params.upper_probe_count, min_weight);
    // (With how much of each the probes there know, in the alpha: see
    // `HORIZON`.)
    var radiance = vec4<f32>(0.0);
    if params.cascade_index == 0u {
        // The first cascade is what the gather integrates over each surface's
        // hemisphere, and its directions are few: one covers a sixteenth of
        // the sphere, across which the cosine of the angle to the normal goes
        // from 1 to half that. So its four finer directions are each weighed
        // by their own cosine here, and the gather weighs the whole by their
        // sum: a lamp straight overhead then counts in full, where the plain
        // mean had it at the mean cosine of the direction it fell in, and
        // what is below the horizon of a direction that straddles it does not
        // count at all.
        var cosines: array<f32, 4>;
        var total = 0.0;
        for (var k = 0u; k < 4u; k++) {
            let child = dir_texel * 2u + vec2<u32>(k & 1u, k >> 1u);
            cosines[k] = max(dot(normal, oct_dir(child, upper_res)), 0.0);
            total += cosines[k];
        }
        if total > 1e-4 {
            for (var j = 0u; j < 4u; j++) {
                let probe = probe_index(weights.base, j, params.upper_probe_count) * i32(upper_res);
                for (var k = 0u; k < 4u; k++) {
                    let child = vec2<i32>(dir_texel * 2u + vec2<u32>(k & 1u, k >> 1u));
                    radiance += (weights.w[j] * cosines[k] / total) * textureLoad(cascade_in, probe + child, 0);
                }
            }
            return known(radiance);
        }
    }
    // The shared corner of the 2x2 child texels: one bilinear tap averages them.
    let texel = vec2<f32>(dir_texel * 2u + 1u);
    for (var j = 0u; j < 4u; j++) {
        let probe = probe_index(weights.base, j, params.upper_probe_count);
        let uv = (vec2<f32>(probe * i32(upper_res)) + texel) / vec2<f32>(params.upper_texture_size);
        radiance += weights.w[j] * textureSampleLevel(cascade_in, linear_sampler, uv, 0.0);
    }
    return known(radiance);
}

/// A probe above the first cascade knows nothing of the directions that lead
/// into the surface it stands on (this far below its horizon, as the cosine
/// of the angle to its normal): its ray would start inside the wall. It
/// leaves them empty, alpha and all, and whoever merges with it goes by the
/// probes around it that do know. Counted as dark instead, they were: the
/// probes of the far cascades are far apart, and of those around a small
/// thing half stand on the wall behind it, which had a box top lit from
/// straight above by half of what reached it.
const HORIZON: f32 = -0.1;

/// The mean of what is known of a sum of radiances that are empty where they
/// are not (see `HORIZON`).
fn known(sum: vec4<f32>) -> vec3<f32> {
    if sum.a < 1e-3 {
        // Nothing is: a small thing against the sky, say, with every probe
        // around it on the background.
        return params.sky_radiance * view.exposure;
    }
    return sum.rgb / sum.a;
}

/// An emitter lights a surface if the point this far off it, in world units,
/// is in the emitter's box, and nothing by this far out of it. The walls of
/// its room stand on the box, or a little outside it, and their faces look
/// in: lifted off them, the point is inside. Measured at the surface itself
/// with as much allowance, the floor of the next room was lit along the foot
/// of the wall between them, and whatever stood there.
const EMITTER_BOUNDS_LIFT: f32 = 0.5;
const EMITTER_BOUNDS_NONE: f32 = 0.1;

/// How much of an emitter at `distance` falls to a cascade whose interval
/// starts at `start`, as against the one below it. Not all or nothing at the
/// border: where a probe stands shifts a little with every frame's jitter, and
/// an emitter that went from one cascade to the next with it changed hands
/// between rays that see it differently, and between cascades that are not
/// all traced in the same frame, which then both had it or neither.
fn emitter_share(distance: f32, start: f32) -> f32 {
    return smoothstep(start * 0.8, start * 1.25, distance);
}

/// The light of the emitters the rays are told of (`RadianceEmitter`) that
/// falls to this ray: of those whose distance is in its interval, a share by
/// how near its direction is to theirs. An emitter is mostly smaller than a
/// texel of the direction map, so no ray would hit it but by chance: instead
/// its light, all of what it sends this way, is shared out among the texels
/// around its direction, each of whose rays says whether it is in sight.
fn emitted(
    position: vec3<f32>,
    normal: vec3<f32>,
    dir_texel: vec2<u32>,
    ray_dir: vec3<f32>,
    t0: f32,
    t1: f32,
    last: bool,
    screen: Trace,
) -> vec3<f32> {
    var light = vec3<f32>(0.0);
    let res = params.dir_res;
    // The solid angle of a texel of the direction map.
    let texel_angle = 4.0 * PI / f32(res * res);
    for (var i = 0u; i < emitters.count; i++) {
        let emitter = emitters.items[i];
        // Only what is in its box (its room).
        let lifted = position + normal * EMITTER_BOUNDS_LIFT;
        let outside = max(emitter.bounds_min - lifted, lifted - emitter.bounds_max);
        let out = max(outside.x, max(outside.y, outside.z));
        if out >= EMITTER_BOUNDS_NONE {
            continue;
        }
        // (A fill light, which nothing shadows, is the gather's: `fill_light`.)
        if emitter.clearance >= emitter.range {
            continue;
        }
        let to = emitter.position - position;
        let distance = length(to);
        if distance >= emitter.range || distance < 1e-4 {
            continue;
        }
        // This cascade's part of it: what is left of it after the one below,
        // less what the one above takes.
        var part = 1.0;
        if params.cascade_index > 0u {
            part = emitter_share(distance, t0);
        }
        if !last {
            part *= 1.0 - emitter_share(distance, t1);
        }
        if part <= 0.0 {
            continue;
        }
        let towards = to / distance;
        var facing = 1.0;
        if any(emitter.normal != vec3<f32>(0.0)) {
            facing = dot(emitter.normal, -towards);
        }
        // (Nor does it light a surface from behind.)
        if facing <= 0.0 || dot(normal, towards) <= 0.0 {
            continue;
        }
        let at = oct_texel(towards, res) - (vec2<f32>(dir_texel) + 0.5);
        let share = max(1.0 - abs(at.x), 0.0) * max(1.0 - abs(at.y), 0.0);
        if share <= 0.0 {
            continue;
        }
        // Whether anything stands before it along this ray: on screen, or
        // where the screen cannot say, in the world cache.
        // (Short of it by its clearance, and of the voxels it is in.)
        let reach = distance - max(emitter.clearance, voxels.voxel_size * 1.5);
        var sight = 1.0;
        if screen.result.a < 1.0 && screen.distance < reach {
            sight = 0.0;
        } else if screen.unknown && screen.unknown_from < reach && voxels.levels > 0u {
            let start = max(max(t0, voxels.voxel_size * 1.5), screen.unknown_from - voxels.voxel_size * 2.0);
            if reach > start {
                sight = trace_voxels(position + normal * voxels.voxel_size, ray_dir, start, reach).a;
            }
        }
        if sight <= 0.0 {
            continue;
        }
        // (Softened where it is nearer than it is wide, and at its range.)
        let angle = emitter.area * facing / (distance * distance + emitter.area);
        var fade = 1.0 - smoothstep(0.75 * emitter.range, emitter.range, distance);
        fade *= 1.0 - smoothstep(0.0, EMITTER_BOUNDS_NONE, out);
        light += emitter.radiance * (view.exposure * angle / texel_angle * share * sight * fade * part);
    }
    return min(light, vec3<f32>(MAX_RADIANCE));
}

@compute @workgroup_size(8, 8, 1)
fn cascade(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= params.texture_size) {
        return;
    }
    let res = params.dir_res;
    let probe = vec2<i32>(id.xy / res);
    let dir_texel = id.xy % res;
    let dir = oct_dir(dir_texel, res);
    let sky = params.sky_radiance * view.exposure;
    let last = params.cascade_index + 1u >= params.cascade_count;

    let pix = probe_pixel(probe, params.probe_spacing);
    let depth = textureLoad(depth_tex, pix, 0);
    if depth <= 0.0 {
        // Probe sits on the background: it stands nowhere, and knows nothing
        // of what arrives at the surfaces beside it on screen (see `HORIZON`).
        textureStore(output, id.xy, vec4<f32>(0.0));
        return;
    }

    let pixel = vec2<f32>(pix) + 0.5;
    let position = world_position(pixel, depth);
    let normal = surface_normal(pix, textureLoad(gbuffer_tex, pix, 0), depth);
    let footprint = pixel_footprint(linear_depth(depth));

    var ray_dir = dir;
    if params.cascade_index > 0u && dot(normal, dir) < HORIZON {
        textureStore(output, id.xy, vec4<f32>(0.0));
        return;
    }
    if params.cascade_index == 0u {
        // The first interval starts on the surface, so directions into it see
        // nothing. A texel that straddles the horizon stands for the part of
        // it above: bend its ray up there, so that the result does not flip as
        // the normal turns (which shows as facets on curved surfaces).
        let cosine = dot(normal, dir);
        let texel_radius = 2.0 / f32(res);
        if cosine <= -texel_radius {
            textureStore(output, id.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
            return;
        }
        let lift = max(0.5 * (cosine + texel_radius), 0.05) - cosine;
        ray_dir = normalize(dir + normal * max(lift, 0.0));
    }

    // Intervals are measured in pixel footprints at the probe, so they scale
    // with the probe spacing in world space wherever the probe happens to be.
    let t0 = params.interval_start * footprint;
    let t1 = params.interval_end * footprint;
    let origin = position + normal * (params.normal_bias * footprint);

    // A texel of the direction map covers a solid angle of 4 pi / res^2: a
    // cone about this wide per unit of distance.
    let spread = 3.54 / f32(res);
    let fade_end = select(t1 + (t1 - t0) * FADE, t1, last);
    let screen = trace(origin, ray_dir, t0, fade_end, pixel, spread);
    var result = screen.result;
    if result.a <= 0.0 && screen.distance > t1 {
        let keep = 1.0 - (screen.distance - t1) / (fade_end - t1);
        result = vec4<f32>(result.rgb * keep, 1.0 - keep);
    }
    if screen.unknown && result.a >= 1.0 && voxels.levels > 0u {
        // The world cache knows nothing finer than a voxel: intervals that
        // end closer than that are left to the screen, and the rest start a
        // voxel clear of the surface they leave. Nor is it asked about what
        // the screen has just shown to be empty: only from (a little before)
        // where the ray left the screen or went behind something. What it
        // remembers of there may be out of date.
        let start = max(max(t0, voxels.voxel_size * 1.5), screen.unknown_from - voxels.voxel_size * 2.0);
        if t1 > start {
            result = trace_voxels(position + normal * voxels.voxel_size, ray_dir, start, t1);
        }
    }
    result = vec4<f32>(result.rgb + emitted(position, normal, dir_texel, ray_dir, t0, t1, last, screen), result.a);
    if result.a > 0.0 {
        if last {
            result = vec4<f32>(result.rgb + result.a * sky, 1.0);
        } else {
            result = vec4<f32>(result.rgb + result.a * merge_upper(pixel, position, normal, footprint, dir_texel), 1.0);
        }
    }
    textureStore(output, id.xy, vec4<f32>(result.rgb, 1.0));
}
