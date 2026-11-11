//! Final gather: integrates cascade 0 over the hemisphere of every pixel and
//! turns it into the diffuse indirect light leaving that pixel. Also draws the
//! debug views.

import package::common::{
    params, view, voxels, depth_tex, gbuffer_tex, cascade_in, output, trace_voxels,
    history, ambient_occlusion, recent_history, recent_out,
    irradiance_history, irradiance_out, motion_vectors, ndc_to_pixel, view_size_f,
    oct_dir, view_size_u, linear_depth, pixel_footprint, world_position,
    gbuffer_valid, gbuffer_normal, surface_normal, gbuffer_reflectance, probe_weights, probe_index, MAX_RADIANCE,
    DEBUG_IRRADIANCE, DEBUG_SURFACES, DEBUG_WORLD_CACHE,
};

/// Per axis, per direction texel.
const SUBSAMPLES: u32 = 3u;

/// The indirect light arriving at a pixel, as a white diffuse surface would
/// reflect it (irradiance / pi).
fn incoming(pix: vec2<i32>, g: vec4<u32>, depth: f32) -> vec3<f32> {
    let normal = surface_normal(pix, g, depth);
    let pixel = vec2<f32>(pix) + 0.5;
    let position = world_position(pixel, depth);
    let footprint = pixel_footprint(linear_depth(depth));
    let weights = probe_weights(pixel, position, normal, footprint, params.probe_spacing, params.probe_count, 1e-4);

    let res = params.dir_res;
    var probes: array<vec2<i32>, 4>;
    for (var j = 0u; j < 4u; j++) {
        probes[j] = probe_index(weights.base, j, params.probe_count) * i32(res);
    }

    // Every texel covers the same solid angle, so the cosine-weighted mean of
    // the radiance is irradiance / pi.
    var sum = vec3<f32>(0.0);
    var cosine_sum = 0.0;
    for (var y = 0u; y < res; y++) {
        for (var x = 0u; x < res; x++) {
            // The cosine lobe integrated over the texel rather than sampled at
            // its centre, or the few directions show as facets on curved surfaces.
            var cosine = 0.0;
            for (var k = 0u; k < SUBSAMPLES * SUBSAMPLES; k++) {
                let sub = vec2<u32>(x, y) * SUBSAMPLES + vec2<u32>(k % SUBSAMPLES, k / SUBSAMPLES);
                cosine += max(dot(normal, oct_dir(sub, res * SUBSAMPLES)), 0.0);
            }
            if cosine <= 0.0 {
                continue;
            }
            let texel = vec2<i32>(i32(x), i32(y));
            var radiance = vec3<f32>(0.0);
            for (var j = 0u; j < 4u; j++) {
                radiance += weights.w[j] * textureLoad(cascade_in, probes[j] + texel, 0).rgb;
            }
            sum += radiance * cosine;
            cosine_sum += cosine;
        }
    }
    let light = sum / max(cosine_sum, 1e-6) * params.intensity;
    return clamp(light, vec3<f32>(0.0), vec3<f32>(MAX_RADIANCE));
}

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

/// What `accumulate` makes of a pixel: the light, how many frames of it have
/// accumulated, and the light over only the last few of them.
struct Accumulated {
    light: vec3<f32>,
    frames: f32,
    recent: vec3<f32>,
}

/// The share of a frame in the light of the last few frames.
const RECENT_BLEND: f32 = 0.25;
/// How far from the light of the last few frames, in standard deviations of
/// this frame's light around the pixel and as a share of it, the accumulated
/// light may be before it is taken for stale.
const HISTORY_DEVIATIONS: f32 = 2.0;
const HISTORY_SLACK: f32 = 0.5;
/// How many times brighter than it has been, and by how much besides, a
/// frame may make a place that has been in view a while. A probe now and then
/// catches a small bright light square on for a frame; unchecked, a patch of
/// wall lights up with it and takes a second to fade. (A light that does come
/// on still gets there in a few frames, each as many times the last.)
const FIREFLY_GAIN: f32 = 3.0;
const FIREFLY_FLOOR: f32 = 0.02;
const FIREFLY_FRAMES: f32 = 4.0;

/// Blends this frame's incoming light into what has accumulated at the same
/// place in the world. A view that is jittered for temporal anti-aliasing draws
/// small lights differently every frame, and what they light flickers with
/// them; accumulated, it is as steady as the anti-aliased image.
///
/// Two lengths of memory are kept: the light as it accumulates at the pace of
/// `temporal_blend`, which is what is shown, and the light of only the last few
/// frames. The long one is held within reach of the short one, so that light
/// which has moved on leaves no trail behind it. (Held within reach of this
/// frame alone, as it once was, it was thrown away whenever a frame differed
/// from the rest: a small bright light that the rays of one jitter in eight
/// miss made what it lit drop every eighth frame and creep back.)
///
/// `frames` is kept in the alpha of the indirect light, `history` being last
/// frame's. A place that has only just come into view takes the mean of what
/// it has seen so far, and only settles into `temporal_blend` once it has seen
/// as many frames as that stands for: it is neither left with a frame's worth
/// of flicker nor slow to lose it.
fn accumulate(traced: vec3<f32>, deviation: vec3<f32>, pix: vec2<i32>, depth: f32) -> Accumulated {
    var light = traced;
    let fresh = Accumulated(light, 1.0, light);
    if params.temporal_blend >= 1.0 {
        return fresh;
    }
    let position = world_position(vec2<f32>(pix) + 0.5, depth);
    let clip = params.prev_clip_from_world * vec4<f32>(position, 1.0);
    if clip.w <= 0.0 {
        return fresh;
    }
    // Perspective: w is the distance along the view axis, as stored.
    let expected = clip.w;
    // Where this place was on screen last frame: where the camera's movement
    // puts it or, where the scene says how its pixels moved, where that does,
    // which also follows what moves on its own.
    var at = ndc_to_pixel(clip.xy / clip.w) - 0.5;
    if params.has_motion_vectors != 0u {
        let motion = textureLoad(motion_vectors, pix, 0).rg;
        at = vec2<f32>(pix) - motion * view_size_f();
    }
    let base = vec2<i32>(floor(at));
    let f = at - floor(at);
    let size = vec2<i32>(view_size_u());

    var sum = vec3<f32>(0.0);
    var sum_recent = vec3<f32>(0.0);
    var frames = 0.0;
    var weight = 0.0;
    for (var j = 0; j < 4; j++) {
        let off = vec2<i32>(j & 1, j >> 1);
        let p = base + off;
        if any(p < vec2<i32>(0)) || any(p >= size) {
            continue;
        }
        let old = textureLoad(irradiance_history, p, 0);
        // Only what arrived at this surface: not at what stood before it then.
        if old.a <= 0.0 || abs(old.a - expected) > 0.1 * expected {
            continue;
        }
        let w = select(1.0 - f.x, f.x, off.x == 1) * select(1.0 - f.y, f.y, off.y == 1);
        sum += old.rgb * w;
        sum_recent += textureLoad(recent_history, p, 0).rgb * w;
        frames += textureLoad(history, p, 0).a * w;
        weight += w;
    }
    if weight < 0.05 {
        return fresh;
    }
    let count = min(frames / weight + 1.0, 1.0 / params.temporal_blend);
    let share = max(params.temporal_blend, 1.0 / count);
    // (The history is in last frame's exposure.)
    let old_long = sum / weight * voxels.history_scale;
    let old_recent = sum_recent / weight * voxels.history_scale;
    if frames / weight >= FIREFLY_FRAMES {
        let before = max(luminance(old_long), luminance(old_recent));
        light *= min(1.0, (before * FIREFLY_GAIN + FIREFLY_FLOOR) / max(luminance(light), 1e-6));
    }
    let recent = mix(old_recent, light, max(share, RECENT_BLEND));
    let range = deviation * HISTORY_DEVIATIONS + recent * HISTORY_SLACK;
    let old = clamp(old_long, max(recent - range, vec3<f32>(0.0)), recent + range);
    return Accumulated(mix(old, light, share), count, recent);
}

@compute @workgroup_size(8, 8, 1)
fn gather(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= view_size_u()) {
        return;
    }
    let pix = vec2<i32>(id.xy);
    let depth = textureLoad(depth_tex, pix, 0);
    let g = textureLoad(gbuffer_tex, pix, 0);
    // What was drawn forward has no albedo to reflect the light with.
    if depth <= 0.0 || !gbuffer_valid(g) {
        textureStore(output, pix, vec4<f32>(0.0));
        return;
    }
    // As traced, for `resolve` to steady: the light, and the distance of the
    // surface it arrives at.
    textureStore(output, pix, vec4<f32>(incoming(pix, g, depth), linear_depth(depth)));
}

/// Taps to each side of a pixel that `resolve` filters over.
const FILTER_RADIUS: i32 = 2;
/// Steadies what `gather` traced, in space and then in time, and turns it into
/// the light leaving each pixel. Here `cascade_in` is the output of `gather`.
///
/// In space: each pixel takes the mean of the light around it on the same
/// surface. The probes are a few pixels apart and what they resolve is coarser
/// still, so little is lost, and what differs from the probes of one pixel to
/// those of the next, which is what crawls as the view moves, evens out.
@compute @workgroup_size(8, 8, 1)
fn resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= view_size_u()) {
        return;
    }
    let pix = vec2<i32>(id.xy);
    let center = textureLoad(cascade_in, pix, 0);
    let g = textureLoad(gbuffer_tex, pix, 0);
    if center.a <= 0.0 || !gbuffer_valid(g) {
        textureStore(output, pix, vec4<f32>(0.0));
        textureStore(irradiance_out, pix, vec4<f32>(0.0));
        textureStore(recent_out, pix, vec4<f32>(0.0));
        return;
    }
    let depth = textureLoad(depth_tex, pix, 0);
    let normal = gbuffer_normal(g);
    let position = world_position(vec2<f32>(pix) + 0.5, depth);
    let size = vec2<i32>(view_size_u());
    let stride = max(i32(params.probe_spacing) / 2, 1);
    // How far off the surface of the pixel a neighbour may lie.
    let tolerance = pixel_footprint(center.a) * f32(stride) * 2.0;

    var sum = vec3<f32>(0.0);
    var sum_squares = vec3<f32>(0.0);
    var weight = 0.0;
    for (var y = -FILTER_RADIUS; y <= FILTER_RADIUS; y++) {
        for (var x = -FILTER_RADIUS; x <= FILTER_RADIUS; x++) {
            let p = pix + vec2<i32>(x, y) * stride;
            if any(p < vec2<i32>(0)) || any(p >= size) {
                continue;
            }
            let tap = textureLoad(cascade_in, p, 0);
            let tap_g = textureLoad(gbuffer_tex, p, 0);
            if tap.a <= 0.0 || !gbuffer_valid(tap_g) {
                continue;
            }
            let tap_position = world_position(vec2<f32>(p) + 0.5, textureLoad(depth_tex, p, 0));
            let off_plane = dot(normal, tap_position - position) / tolerance;
            let facing = max(dot(normal, gbuffer_normal(tap_g)), 0.0);
            let facing_2 = facing * facing;
            let facing_4 = facing_2 * facing_2;
            let w = exp(-0.25 * f32(x * x + y * y) - off_plane * off_plane) * facing_4 * facing_4;
            sum += tap.rgb * w;
            sum_squares += tap.rgb * tap.rgb * w;
            weight += w;
        }
    }
    // (The pixel itself always counts, in full.)
    // The creases and contacts between the probes, which they are too far
    // apart to resolve. (Before the light is accumulated: ambient occlusion
    // is traced through the jittered view like the rest, and flickers alike.)
    var open = 1.0;
    if params.ambient_occlusion > 0.0 {
        // Pixel for pixel: it is computed for the pixels the view renders,
        // which a texture as large as the whole render target may hold in
        // its corner (where the main pass renders at a lower resolution).
        let at = min(pix, vec2<i32>(textureDimensions(ambient_occlusion)) - 1);
        open = mix(1.0, saturate(textureLoad(ambient_occlusion, at, 0).r), params.ambient_occlusion);
    }
    let mean = sum / weight * open;
    let deviation = sqrt(max(sum_squares / weight * open * open - mean * mean, vec3<f32>(0.0)));

    let accumulated = accumulate(mean, deviation, pix, depth);
    let light = accumulated.light;
    textureStore(irradiance_out, pix, vec4<f32>(light, center.a));
    textureStore(recent_out, pix, vec4<f32>(accumulated.recent, 1.0));
    let to_eye = normalize(view.world_position - position);
    textureStore(output, pix, vec4<f32>(light * gbuffer_reflectance(g, normal, to_eye), accumulated.frames));
}

/// Draws the debug view that replaces the image. Same bindings as `gather`,
/// but `output` is a texture of its own, so that what is shown never feeds
/// back into the lighting, and `irradiance_history` is what `gather` has just
/// accumulated.
@compute @workgroup_size(8, 8, 1)
fn debug_view(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= view_size_u()) {
        return;
    }
    let pix = vec2<i32>(id.xy);
    let depth = textureLoad(depth_tex, pix, 0);
    let g = textureLoad(gbuffer_tex, pix, 0);
    var color = vec3<f32>(0.0);

    if params.debug == DEBUG_WORLD_CACHE {
        // The world cache as the camera would see it, were it the scene.
        let far = world_position(vec2<f32>(pix) + 0.5, 1e-6);
        let dir = normalize(far - view.world_position);
        let hit = trace_voxels(view.world_position, dir, 0.0, 1.0e6);
        color = hit.rgb + hit.a * vec3<f32>(0.02, 0.0, 0.03);
    } else if params.debug == DEBUG_SURFACES {
        if depth > 0.0 {
            let shade = 0.1 + 0.9 * fract(log2(linear_depth(depth)));
            color = select(vec3<f32>(shade, 0.0, 0.0), vec3<f32>(0.0, shade, 0.0), gbuffer_valid(g));
        }
    } else if params.debug == DEBUG_IRRADIANCE {
        color = textureLoad(irradiance_history, pix, 0).rgb;
    }
    textureStore(output, pix, vec4<f32>(color, 1.0));
}
