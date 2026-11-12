//! Final gather: integrates cascade 0 over the hemisphere of every pixel and
//! turns it into the diffuse indirect light leaving that pixel. Also draws the
//! debug views.

import package::common::{
    params, view, voxels, depth_tex, gbuffer_tex, cascade_in, output, trace_voxels,
    history, ambient_occlusion, recent_history, recent_out, emitters, PI,
    irradiance_history, irradiance_out, motion_vectors, ndc_to_pixel, view_size_f,
    oct_dir, view_size_u, linear_depth, pixel_footprint, world_position,
    gbuffer_valid, gbuffer_normal, surface_normal, gbuffer_diffuse_color, gbuffer_f0, gbuffer_roughness, gbuffer_specular_color, specular_share, specular_in, specular_out, specular_history, highlight_out, linear_sampler, oct_texel, probe_weights, probe_index, probe_pixel, MAX_RADIANCE,
    DEBUG_IRRADIANCE, DEBUG_SURFACES, DEBUG_WORLD_CACHE, DEBUG_REFLECTANCE, DEBUG_MATERIALS,
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
    // And the uniform fill (`RadianceCascades::ambient`), as what a white
    // surface would send on of it: it bounces from here on like the rest.
    let ambient = vec3<f32>(params.ambient * view.exposure / PI) * params.intensity;
    return clamp(light + fill_light(position, normal) + ambient, vec3<f32>(0.0), vec3<f32>(MAX_RADIANCE));
}

/// The light of the emitters that nothing shadows (`RadianceEmitter::clearance`
/// as much as its range): on every surface in their box that faces them, as an
/// unshadowed lamp's. As the rest of what `incoming` returns, the radiance a
/// white surface would send on: the light arriving, over pi.
fn fill_light(position: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    var light = vec3<f32>(0.0);
    for (var i = 0u; i < emitters.count; i++) {
        let emitter = emitters.items[i];
        if emitter.clearance < emitter.range {
            continue;
        }
        // (In its box, measured as the cascades do: see `emitted`.)
        let lifted = position + normal * 0.5;
        let outside = max(emitter.bounds_min - lifted, lifted - emitter.bounds_max);
        let out = max(outside.x, max(outside.y, outside.z));
        if out >= 0.1 {
            continue;
        }
        let to = emitter.position - position;
        let distance = length(to);
        if distance >= emitter.range || distance < 1e-4 {
            continue;
        }
        let towards = to / distance;
        var facing = 1.0;
        if any(emitter.normal != vec3<f32>(0.0)) {
            facing = max(dot(emitter.normal, -towards), 0.0);
        }
        let cosine = max(dot(normal, towards), 0.0);
        let angle = emitter.area * facing / (distance * distance + emitter.area);
        var fade = 1.0 - smoothstep(0.75 * emitter.range, emitter.range, distance);
        fade *= 1.0 - smoothstep(0.0, 0.1, out);
        light += emitter.radiance * (view.exposure * angle * cosine * fade / PI);
    }
    return light * params.intensity;
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
    /// What the place mirrored last frame, and whether that is known.
    mirrored: vec3<f32>,
    known: bool,
    /// The share this frame has in what is accumulated.
    share: f32,
}

/// The least share a frame has in what a surface mirrors (see `resolve`).
const SPECULAR_BLEND: f32 = 0.25;

/// How many times the light arriving at a surface from all round what it
/// mirrors of the cascades may be (see `resolve`).
const MIRROR_GAIN: f32 = 1.5;

/// The highlights of the emitters on a surface: each as a small round light
/// of its size, its light arriving from where it is. Not shadowed (what the
/// cascades know of what stands between is too coarse for a highlight), but
/// only in its box and on what faces it; and not the fill lights, which stand
/// for light that has no one place to be mirrored from.
fn highlights(g: vec4<u32>, position: vec3<f32>, normal: vec3<f32>, to_eye: vec3<f32>, perceptual_roughness: f32) -> vec3<f32> {
    if params.specular <= 0.0 {
        return vec3<f32>(0.0);
    }
    let f0 = gbuffer_f0(g);
    let roughness = max(perceptual_roughness * perceptual_roughness, 0.03);
    let n_dot_v = max(dot(normal, to_eye), 1e-3);
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < emitters.count; i++) {
        let emitter = emitters.items[i];
        if emitter.clearance >= emitter.range {
            continue;
        }
        let lifted = position + normal * 0.5;
        let outside = max(emitter.bounds_min - lifted, lifted - emitter.bounds_max);
        if max(outside.x, max(outside.y, outside.z)) >= 0.1 {
            continue;
        }
        let to = emitter.position - position;
        let distance = length(to);
        if distance >= emitter.range || distance < 1e-3 {
            continue;
        }
        let towards = to / distance;
        var facing = 1.0;
        if any(emitter.normal != vec3<f32>(0.0)) {
            facing = max(dot(emitter.normal, -towards), 0.0);
        }
        let n_dot_l = dot(normal, towards);
        if facing <= 0.0 || n_dot_l <= 0.0 {
            continue;
        }
        // A light of some size reflects as a wider highlight than a point's,
        // of the same light in all (Karis: the roughness widened by the angle
        // it fills, and the highlight dimmed as much).
        let radius = sqrt(emitter.area * facing / PI);
        let wide = saturate(roughness + radius / (2.0 * distance));
        let spread = roughness / wide;
        let half_way = normalize(towards + to_eye);
        let n_dot_h = saturate(dot(normal, half_way));
        let v_dot_h = saturate(dot(to_eye, half_way));
        let a2 = wide * wide;
        let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
        let distribution = a2 / (PI * d * d) * spread * spread;
        let k = wide * 0.5;
        let visible = 0.25 / ((n_dot_v * (1.0 - k) + k) * (n_dot_l * (1.0 - k) + k));
        let fresnel = f0 + (1.0 - f0) * pow(1.0 - v_dot_h, 5.0);
        let arriving = emitter.area * facing / (distance * distance + emitter.area);
        let fade = 1.0 - smoothstep(0.75 * emitter.range, emitter.range, distance);
        sum += emitter.radiance * fresnel * (view.exposure * arriving * n_dot_l * distribution * visible * fade);
    }
    return min(sum * params.specular * params.intensity, vec3<f32>(MAX_RADIANCE));
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
    let fresh = Accumulated(light, 1.0, light, vec3<f32>(0.0), false, 1.0);
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
    var sum_mirrored = vec3<f32>(0.0);
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
        sum_mirrored += textureLoad(specular_history, p, 0).rgb * w;
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
    return Accumulated(mix(old, light, share), count, recent, sum_mirrored / weight * voxels.history_scale, true, share);
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

/// The cascades' light from around the direction a surface mirrors to the
/// eye: read from `specular_in`, a cascade above the first, whose directions
/// are finer (and which lacks only what is nearest the surface).
fn reflected(pix: vec2<i32>, position: vec3<f32>, normal: vec3<f32>, to_eye: vec3<f32>, distance: f32, roughness: f32) -> vec3<f32> {
    var dir = reflect(-to_eye, normal);
    // (Kept off the surface: a direction into it sees nothing.)
    dir = normalize(dir + normal * max(0.05 - dot(dir, normal), 0.0));
    let spacing = params.probe_spacing << params.spec_level;
    let res = params.dir_res << params.spec_level;
    // From the probes around the pixel, three a side and weighed smoothly (a
    // quadratic B-spline): each probe has one ray a direction, which a lamp or
    // a lit edge is in for one probe and not for the next, and blended
    // linearly between the nearest four that difference showed as creases
    // along the probes' grid, which crawled as the view moved.
    let s = f32(spacing);
    let g = (vec2<f32>(pix) + 0.5 - 0.5 - f32(spacing / 2u)) / s;
    let nearest = floor(g + 0.5);
    let f = g - nearest;
    let wx = vec3<f32>(0.5 * (0.5 - f.x) * (0.5 - f.x), 0.75 - f.x * f.x, 0.5 * (0.5 + f.x) * (0.5 + f.x));
    let wy = vec3<f32>(0.5 * (0.5 - f.y) * (0.5 - f.y), 0.75 - f.y * f.y, 0.5 * (0.5 + f.y) * (0.5 + f.y));
    let tolerance = pixel_footprint(distance) * s;
    // And from around the mirrored direction, the wider the rougher the
    // surface (which is what it mirrors, and evens out the one ray too).
    let around = clamp(roughness * 2.0, 0.5, 1.5);
    let texel = oct_texel(dir, res);
    let low = vec2<f32>(0.5);
    let high = vec2<f32>(f32(res) - 0.5);
    // (Its alpha is how much of it the probes know: directions into the
    // surface a probe stands on are empty there.)
    var sum = vec4<f32>(0.0);
    for (var j = 0; j < 9; j++) {
        let off = vec2<i32>(j % 3, j / 3);
        let probe = clamp(vec2<i32>(nearest) + off - 1, vec2<i32>(0), vec2<i32>(params.spec_probe_count) - 1);
        let at = probe_pixel(probe, spacing);
        let d = textureLoad(depth_tex, at, 0);
        var similarity = 0.0;
        if d > 0.0 {
            let off_plane = dot(normal, world_position(vec2<f32>(at) + 0.5, d) - position) / tolerance;
            similarity = exp(-off_plane * off_plane);
        }
        let w = wx[off.x] * wy[off.y] * max(similarity, 0.02);
        // (Within the probe's own map: its neighbours' lie right beside it.)
        let corner = vec2<f32>(probe * i32(res));
        var seen = vec4<f32>(0.0);
        for (var k = 0; k < 4; k++) {
            let side = vec2<f32>(f32(k & 1), f32(k >> 1)) * 2.0 - 1.0;
            let uv = (corner + clamp(texel + side * around, low, high)) / vec2<f32>(params.spec_texture_size);
            seen += textureSampleLevel(specular_in, linear_sampler, uv, 0.0);
        }
        sum += seen * (0.25 * w);
    }
    if sum.a < 1e-6 {
        return vec3<f32>(0.0);
    }
    return sum.rgb / sum.a * params.intensity;
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
        textureStore(specular_out, pix, vec4<f32>(0.0));
        textureStore(highlight_out, pix, vec4<f32>(0.0));
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
    // What the surface mirrors. A rough one mirrors about all that arrives
    // at it; a glossy one what comes from around the mirrored direction,
    // which is what tells a floor under a dark ceiling from one under a lit
    // wall.
    let to_eye = normalize(view.world_position - position);
    let roughness = gbuffer_roughness(g);
    let tint = gbuffer_specular_color(g, normal, to_eye);
    var mirrored = light;
    let glossy = 1.0 - smoothstep(0.35, 0.75, roughness);
    if params.specular > 0.0 && glossy > 0.0 {
        // (No more than a few times what arrives at it from all round: a
        // small bright lamp is in one direction of one probe's map, and what
        // mirrors that direction took all of it, a blot of light that came and
        // went as the view moved. The lamps' highlights are drawn from the
        // lamps themselves, below.)
        var seen = reflected(pix, position, normal, to_eye, center.a, roughness) * open;
        seen *= min(1.0, (luminance(light) * MIRROR_GAIN + 1e-4) / max(luminance(seen), 1e-6));
        mirrored = mix(light, seen, glossy);
    }
    // Steadied in time like the rest, at the same place in the world: read
    // straight from the cascades, it is as coarse as their probes, and from
    // one frame to the next of a moving view it is other probes' light, in
    // stripes that follow their grid. (A short memory: what is mirrored
    // changes with the view, and with as long a one as the rest has it
    // trailed behind a turn of the camera, further off than the stripes were.)
    var specular = mirrored * tint;
    if accumulated.known {
        specular = mix(accumulated.mirrored, specular, max(accumulated.share, SPECULAR_BLEND));
    }
    textureStore(specular_out, pix, vec4<f32>(specular, 1.0));
    // The lamps' highlights, straight into the image: they are the lamps'
    // own light, not something mirrored for a reflection pass to weigh
    // against what its rays find (which left them out whenever a ray hit
    // anything, and so whenever the view moved). Not in the debug views that
    // are not the lighting.
    var shine = vec3<f32>(0.0);
    if params.debug <= 1u {
        shine = highlights(g, position, normal, to_eye, roughness) * open;
    }
    textureStore(highlight_out, pix, vec4<f32>(shine, 1.0));
    var shown = light * gbuffer_diffuse_color(g);
    if params.specular_separate == 0u {
        shown += specular * specular_share(roughness);
    }
    textureStore(output, pix, vec4<f32>(shown, accumulated.frames));
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
    } else if params.debug == DEBUG_REFLECTANCE {
        // How much of the light arriving at each surface it sends on to the
        // eye, its diffuse and its specular side together.
        if depth > 0.0 && gbuffer_valid(g) {
            let position = world_position(vec2<f32>(pix) + 0.5, depth);
            let to_eye = normalize(view.world_position - position);
            color = gbuffer_diffuse_color(g) + gbuffer_specular_color(g, gbuffer_normal(g), to_eye);
        }
    } else if params.debug == DEBUG_MATERIALS {
        // Red its roughness, green how metallic, blue its baked occlusion.
        if depth > 0.0 && gbuffer_valid(g) {
            let props = unpack4x8unorm(g.b);
            color = vec3<f32>(gbuffer_roughness(g), props.g, props.b);
        }
    }
    textureStore(output, pix, vec4<f32>(color, 1.0));
}
