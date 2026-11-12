//! Bindings and helpers shared by the cascade and final-gather passes.
//!
//! Layout of a cascade texture: `probe_count` tiles of `dir_res x dir_res`
//! texels. A tile is one probe; a texel within it is one direction, mapped to
//! the sphere with an equal-area octahedral map so every texel subtends the same
//! solid angle (which is what lets four child directions merge by plain
//! averaging).

import bevy_render::view::View;
import bevy_render::utils::octahedral_decode;
import bevy_pbr::deferred::types::unpack_24bit_normal;
import package::voxel::{
    VoxelParams, voxel_texel, voxel_mask_texel, voxel_normal, voxel_occupied, voxel_confidence,
    VOXEL_RADIANCE_SCALE, VOXEL_CONFIDENCE_MAX, VOXEL_SOFT_OCCUPANCY, VOXEL_BLEND_LOOKUP, VOXEL_SOFT_FACING,
};
import bevy_pbr::render::rgb9e5::rgb9e5_to_vec3_;

struct RcParams {
    prev_clip_from_world: mat4x4<f32>,
    sky_radiance: vec3<f32>,
    bounce: f32,
    probe_count: vec2<u32>,
    upper_probe_count: vec2<u32>,
    texture_size: vec2<u32>,
    upper_texture_size: vec2<u32>,
    probe_spacing: u32,
    dir_res: u32,
    cascade_index: u32,
    cascade_count: u32,
    interval_start: f32,
    interval_end: f32,
    thickness: f32,
    max_steps: u32,
    intensity: f32,
    debug: u32,
    normal_bias: f32,
    depth_bias: f32,
    pyramid_levels: u32,
    /// Size of level 0 of the color pyramid, and its number of levels.
    color_size: vec2<f32>,
    color_levels: f32,
    /// How much of this frame's light the accumulated light takes on.
    temporal_blend: f32,
    /// Whether `motion_vectors` is the motion vector prepass (and not a
    /// placeholder).
    has_motion_vectors: u32,
    /// How much of the screen-space ambient occlusion the indirect light
    /// takes on (0 where there is none).
    ambient_occlusion: f32,
    /// How much of the light arriving at a surface its specular side
    /// reflects, and the perceptual roughness over which that fades in.
    specular: f32,
    specular_roughness: vec2<f32>,
    /// What the G-buffer's emissive is multiplied by to be in the units of
    /// the lit scene, and whether the exposure is still to be applied to it.
    emissive_scale: f32,
    emissive_unexposed: u32,
    /// Whether what surfaces mirror goes to `specular_out` for a reflection
    /// pass to add (and not into the image here).
    specular_separate: u32,
    /// The cascade `specular_in` is, and its probes and size.
    spec_level: u32,
    spec_probe_count: vec2<u32>,
    spec_texture_size: vec2<u32>,
    /// A uniform light arriving at every surface (lux, before the exposure).
    ambient: f32,
}

@group(0) @binding(0) var<uniform> params: RcParams;
@group(0) @binding(1) var<uniform> view: View;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var gbuffer_tex: texture_2d<u32>;
@group(0) @binding(4) var scene_color: texture_2d<f32>;
@group(0) @binding(5) var history: texture_2d<f32>;
@group(0) @binding(6) var cascade_in: texture_2d<f32>;
@group(0) @binding(7) var linear_sampler: sampler;
@group(0) @binding(8) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var depth_pyramid: texture_2d<f32>;
@group(0) @binding(10) var<uniform> voxels: VoxelParams;
@group(0) @binding(11) var voxel_radiance: texture_3d<u32>;
@group(0) @binding(12) var voxel_meta: texture_3d<u32>;
@group(0) @binding(13) var color_pyramid: texture_2d<f32>;
/// The accumulated incoming light of last frame: rgb, and the linear depth of
/// the surface it arrived at in a.
@group(0) @binding(14) var irradiance_history: texture_2d<f32>;
@group(0) @binding(15) var irradiance_out: texture_storage_2d<rgba16float, write>;
/// How far each pixel has moved across the view since last frame, in UV.
@group(0) @binding(16) var motion_vectors: texture_2d<f32>;
/// Which blocks of voxels have anything in them (see `voxel_mask_texel`).
@group(0) @binding(17) var voxel_mask: texture_3d<u32>;
/// Screen-space ambient occlusion, where the camera has it: how open each
/// pixel is to its nearest surroundings, finer than the probes resolve.
@group(0) @binding(18) var ambient_occlusion: texture_2d<f32>;
/// The incoming light over only the last few frames, as of last frame, and
/// this frame's (see `accumulate` in `gather.wesl`).
@group(0) @binding(19) var recent_history: texture_2d<f32>;
@group(0) @binding(20) var recent_out: texture_storage_2d<rgba16float, write>;

/// A flat glowing surface the rays are told of (`RadianceEmitter`).
struct Emitter {
    position: vec3<f32>,
    /// How far its light reaches.
    range: f32,
    /// The side it shines from (zero: all round).
    normal: vec3<f32>,
    area: f32,
    /// Its luminance, before the exposure.
    radiance: vec3<f32>,
    /// How near it things are not in its way.
    clearance: f32,
    /// The box its light stays in.
    bounds_min: vec3<f32>,
    bounds_max: vec3<f32>,
}

struct Emitters {
    count: u32,
    items: array<Emitter, 64>,
}

@group(0) @binding(21) var<uniform> emitters: Emitters;
/// What each pixel mirrors of the cascades' light, weighed by how much its
/// surface reflects: written by `resolve` (see `RadianceCascades::specular_separate`).
@group(0) @binding(22) var specular_out: texture_storage_2d<rgba16float, write>;
/// A cascade with finer directions than the first, merged, for `resolve` to
/// read what surfaces mirror from.
@group(0) @binding(23) var specular_in: texture_2d<f32>;
/// What each pixel mirrored last frame (`specular_out` then), for `resolve` to
/// steady this frame's with, and for the cascades to bounce.
@group(0) @binding(24) var specular_history: texture_2d<f32>;
/// The emitters' highlights on each pixel (see `highlights` in `gather.wesl`):
/// added to the image with the indirect light, and kept apart from it so
/// that they are not bounced on as if the surface sent them every way.
@group(0) @binding(25) var highlight_out: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265358979;
/// What a ray finds: rgb = radiance arriving from its interval, a =
/// transmittance past it.
const MISS: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 1.0);
const OCCLUDED: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
/// Largest value we let into an rgba16float texture.
const MAX_RADIANCE: f32 = 60000.0;
/// `RadianceCascadesDebug::Irradiance`.
const DEBUG_IRRADIANCE: u32 = 2u;
/// `RadianceCascadesDebug::Surfaces`.
const DEBUG_SURFACES: u32 = 3u;
/// `RadianceCascadesDebug::WorldCache`.
const DEBUG_WORLD_CACHE: u32 = 4u;
/// `RadianceCascadesDebug::Reflectance`.
const DEBUG_REFLECTANCE: u32 = 5u;
/// `RadianceCascadesDebug::Materials`.
const DEBUG_MATERIALS: u32 = 6u;

/// Equal-area octahedral map (Clarberg), texel centre -> unit direction.
fn oct_dir(texel: vec2<u32>, res: u32) -> vec3<f32> {
    let p = (vec2<f32>(texel) + 0.5) / f32(res);
    let u = 2.0 * p.x - 1.0;
    let v = 2.0 * p.y - 1.0;
    let up = abs(u);
    let vp = abs(v);
    let signed_distance = 1.0 - (up + vp);
    let d = abs(signed_distance);
    let r = 1.0 - d;
    var phi = PI / 4.0;
    if r > 0.0 {
        phi = ((vp - up) / r + 1.0) * PI / 4.0;
    }
    let z = (1.0 - r * r) * sign_not_zero(signed_distance);
    let s = r * sqrt(max(2.0 - r * r, 0.0));
    return vec3<f32>(
        cos(phi) * sign_not_zero(u) * s,
        sin(phi) * sign_not_zero(v) * s,
        z,
    );
}

/// Where in a direction map `res` texels a side a direction falls, in texels:
/// the inverse of `oct_dir`.
fn oct_texel(dir: vec3<f32>, res: u32) -> vec2<f32> {
    let r = sqrt(max(1.0 - abs(dir.z), 0.0));
    let phi = atan2(abs(dir.y), max(abs(dir.x), 1e-20));
    let difference = r * (phi * 4.0 / PI - 1.0);
    let sum = select(2.0 - r, r, dir.z >= 0.0);
    let u = 0.5 * (sum - difference) * sign_not_zero(dir.x);
    let v = 0.5 * (sum + difference) * sign_not_zero(dir.y);
    return (vec2<f32>(u, v) * 0.5 + 0.5) * f32(res);
}

fn sign_not_zero(x: f32) -> f32 {
    return select(-1.0, 1.0, x >= 0.0);
}

fn view_size_f() -> vec2<f32> {
    return view.main_pass_viewport.zw;
}

fn view_size_u() -> vec2<u32> {
    return vec2<u32>(view.main_pass_viewport.zw);
}

/// Positive distance along the view axis. Perspective, infinite reverse-z.
fn linear_depth(depth: f32) -> f32 {
    return view.clip_from_view[3][2] / depth;
}

/// World-space size of one pixel at the given linear depth.
fn pixel_footprint(lin: f32) -> f32 {
    return lin * 2.0 / (view.clip_from_view[1][1] * view.main_pass_viewport.w);
}

fn ndc_to_pixel(ndc: vec2<f32>) -> vec2<f32> {
    return (ndc * vec2<f32>(0.5, -0.5) + 0.5) * view_size_f();
}

/// `pixel` is a continuous pixel coordinate (pixel centres at +0.5).
fn world_position(pixel: vec2<f32>, depth: f32) -> vec3<f32> {
    let uv = pixel / view_size_f();
    let ndc = (uv - 0.5) * vec2<f32>(2.0, -2.0);
    let p = view.world_from_clip * vec4<f32>(ndc, depth, 1.0);
    return p.xyz / p.w;
}

fn gbuffer_normal(g: vec4<u32>) -> vec3<f32> {
    return octahedral_decode(unpack_24bit_normal(g.a));
}

/// Whether a pixel was drawn through the deferred prepass. What was drawn
/// forward is in the depth buffer and the lit scene, so it still occludes and
/// bounces light, but nothing else is known about it.
fn gbuffer_valid(g: vec4<u32>) -> bool {
    return g.a != 0u;
}

/// The normal of the surface at a pixel: the G-buffer's, or failing that the
/// one the depth buffer implies.
fn surface_normal(pix: vec2<i32>, g: vec4<u32>, depth: f32) -> vec3<f32> {
    if gbuffer_valid(g) {
        return gbuffer_normal(g);
    }
    let max_pix = vec2<i32>(view_size_u()) - 1;
    let center = world_position(vec2<f32>(pix) + 0.5, depth);
    var edges: array<vec3<f32>, 2>;
    for (var axis = 0; axis < 2; axis++) {
        // Of the two neighbours along the axis, the one closer in depth is
        // more likely to lie on the same surface.
        var step = vec2<i32>(0);
        step[axis] = 1;
        let a = clamp(pix + step, vec2<i32>(0), max_pix);
        let b = clamp(pix - step, vec2<i32>(0), max_pix);
        let depth_a = textureLoad(depth_tex, a, 0);
        let depth_b = textureLoad(depth_tex, b, 0);
        if abs(depth_a - depth) <= abs(depth_b - depth) {
            edges[axis] = world_position(vec2<f32>(a) + 0.5, depth_a) - center;
        } else {
            edges[axis] = center - world_position(vec2<f32>(b) + 0.5, depth_b);
        }
    }
    // Pixel x runs right and pixel y runs down, so this faces the camera.
    let n = cross(edges[1], edges[0]);
    let len = length(n);
    if len < 1e-20 {
        return normalize(view.world_position - center);
    }
    return n / len;
}

/// Diffuse albedo, with the material's baked ambient occlusion folded in.
fn gbuffer_diffuse_color(g: vec4<u32>) -> vec3<f32> {
    let base_color = pow(unpack4x8unorm(g.r).rgb, vec3<f32>(2.2));
    let props = unpack4x8unorm(g.b);
    return base_color * (1.0 - props.g) * props.b;
}

/// How rough a surface is, perceptually (0 a mirror, 1 matte).
fn gbuffer_roughness(g: vec4<u32>) -> f32 {
    return unpack4x8unorm(g.r).a;
}

/// How much of the light a surface mirrors towards the eye it reflects, by
/// colour: none of it where `specular` is off. A metal has no diffuse side at
/// all: without this the bounced light falls on it and is never seen.
fn gbuffer_specular_color(g: vec4<u32>, normal: vec3<f32>, to_eye: vec3<f32>) -> vec3<f32> {
    if params.specular <= 0.0 {
        return vec3<f32>(0.0);
    }
    let base = unpack4x8unorm(g.r);
    let base_color = pow(base.rgb, vec3<f32>(2.2));
    let props = unpack4x8unorm(g.b);
    let roughness = base.a;
    let f0 = mix(vec3<f32>(0.16 * props.r * props.r), base_color, props.g);
    // The split-sum environment term, as fitted by Karis for mobile.
    let n_dot_v = saturate(dot(normal, to_eye));
    let r = roughness * vec4<f32>(-1.0, -0.0275, -0.572, 0.022) + vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    // And what scatters more than once between the facets of a rough surface
    // before it leaves (Fdez-Aguera), which the term above leaves out: on a
    // rough metal that is half of what it reflects.
    let single = f0 * ab.x + ab.y;
    let lost = 1.0 - (ab.x + ab.y);
    let average = f0 + (1.0 - f0) / 21.0;
    let multiple = single * average / (1.0 - lost * average) * lost;
    return (single + multiple) * props.b * params.specular;
}

/// What a surface reflects at normal incidence, by colour.
fn gbuffer_f0(g: vec4<u32>) -> vec3<f32> {
    let base_color = pow(unpack4x8unorm(g.r).rgb, vec3<f32>(2.2));
    let props = unpack4x8unorm(g.b);
    return mix(vec3<f32>(0.16 * props.r * props.r), base_color, props.g);
}

/// How much of what a surface mirrors is added to the image here, by its
/// roughness, where it is not left to a reflection pass altogether: the smooth
/// are left to screen-space reflections.
fn specular_share(roughness: f32) -> f32 {
    return smoothstep(params.specular_roughness.x, max(params.specular_roughness.y, params.specular_roughness.x + 1e-3), roughness);
}

/// What glows at a pixel, in the units of the lit scene.
fn gbuffer_emissive(g: vec4<u32>) -> vec3<f32> {
    let scale = params.emissive_scale * select(1.0, view.exposure, params.emissive_unexposed != 0u);
    return rgb9e5_to_vec3_(g.g) * scale;
}

/// The pixel a probe of a cascade with the given spacing sits on.
fn probe_pixel(probe: vec2<i32>, spacing: u32) -> vec2<i32> {
    let s = i32(spacing);
    return min(probe * s + s / 2, vec2<i32>(view_size_u()) - 1);
}

struct ProbeWeights {
    base: vec2<i32>,
    w: vec4<f32>,
}

/// Bilinear weights of the four probes (of a cascade with the given spacing)
/// around a continuous pixel coordinate, attenuated by how far each probe lies
/// off the tangent plane of the receiving surface. That keeps a surface from
/// borrowing radiance from probes that sit on unrelated geometry which merely
/// happens to be adjacent on screen.
///
/// `min_weight` is the least a probe counts for however far off the plane it is.
/// Where no probe fits, a low floor picks whichever fits best, which shows as
/// seams between probes; a high one blends them all, which leaks.
fn probe_weights(
    pixel: vec2<f32>,
    position: vec3<f32>,
    normal: vec3<f32>,
    footprint: f32,
    spacing: u32,
    count: vec2<u32>,
    min_weight: f32,
) -> ProbeWeights {
    let s = f32(spacing);
    let g = (pixel - 0.5 - f32(spacing / 2u)) / s;
    let b = floor(g);
    let f = g - b;
    let base = vec2<i32>(b);
    let tolerance = footprint * s;

    var out: ProbeWeights;
    out.base = base;
    var sum = 0.0;
    for (var j = 0u; j < 4u; j++) {
        let off = vec2<i32>(i32(j & 1u), i32(j >> 1u));
        let idx = clamp(base + off, vec2<i32>(0), vec2<i32>(count) - 1);
        let bil = select(1.0 - f.x, f.x, off.x == 1) * select(1.0 - f.y, f.y, off.y == 1);
        let pix = probe_pixel(idx, spacing);
        let d = textureLoad(depth_tex, pix, 0);
        var similarity = 0.0;
        if d > 0.0 {
            let pj = world_position(vec2<f32>(pix) + 0.5, d);
            let off_plane = dot(normal, pj - position) / tolerance;
            similarity = exp(-off_plane * off_plane);
        }
        let w = bil * max(similarity, min_weight);
        out.w[j] = w;
        sum += w;
    }
    out.w /= max(sum, 1e-20);
    return out;
}

fn probe_index(base: vec2<i32>, j: u32, count: vec2<u32>) -> vec2<i32> {
    let off = vec2<i32>(i32(j & 1u), i32(j >> 1u));
    return clamp(base + off, vec2<i32>(0), vec2<i32>(count) - 1);
}

/// Whether a level of the world cache covers a point.
fn voxel_level_covers(p: vec3<f32>, level: i32) -> bool {
    let size = voxels.voxel_size * f32(1u << u32(level));
    let rel = vec3<i32>(floor(p / size)) - voxels.origin[level].xyz;
    return all(rel >= vec3<i32>(0)) && all(rel < vec3<i32>(i32(voxels.resolution)));
}

/// How far along a ray from `p`, which is outside a level of the world cache,
/// it enters that level (or very far, if it never does).
fn voxel_level_entry(p: vec3<f32>, inv_dir: vec3<f32>, level: i32) -> f32 {
    let size = voxels.voxel_size * f32(1u << u32(level));
    let low = vec3<f32>(voxels.origin[level].xyz) * size;
    let high = low + f32(voxels.resolution) * size;
    let a = (low - p) * inv_dir;
    let b = (high - p) * inv_dir;
    let near = min(a, b);
    let far = max(a, b);
    let enter = max(near.x, max(near.y, near.z));
    let leave = min(far.x, min(far.y, far.z));
    return select(1.0e30, enter, enter > 0.0 && leave >= enter);
}

fn voxel_stored(texel: vec3<i32>) -> vec3<f32> {
    return rgb9e5_to_vec3_(textureLoad(voxel_radiance, texel, 0).r);
}

/// What a ray along `dir` that has hit the voxel `index` of a level at `p`
/// takes from it, where `WorldCache::blend_lookup` is on: not that voxel's
/// color alone but the colors of the voxels around the point, weighted by how
/// near each is, so that the cache does not show as blocks in what it lights;
/// and near the edge of the level, more and more the color of the next coarser
/// level there, so that there is no step in the lighting where one level takes
/// over from another.
fn voxel_blended(p: vec3<f32>, index: vec3<i32>, level: i32, dir: vec3<f32>, own: vec3<f32>) -> vec3<f32> {
    let n = i32(voxels.resolution);
    let size = voxels.voxel_size * f32(1u << u32(level));
    let g = p / size - 0.5;
    let base = floor(g);
    let f = g - base;
    var sum = vec3<f32>(0.0);
    var weight = 0.0;
    for (var k = 0u; k < 8u; k++) {
        let off = vec3<u32>(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u);
        let neighbour = vec3<i32>(base) + vec3<i32>(off);
        let rel = neighbour - voxels.origin[level].xyz;
        if any(rel < vec3<i32>(0)) || any(rel >= vec3<i32>(n)) {
            continue;
        }
        let texel = voxel_texel(neighbour, u32(level), voxels.resolution);
        let info = textureLoad(voxel_meta, texel, 0).r;
        // Only what faces the ray: not the far side of a thin wall.
        if !voxel_occupied(info) || dot(voxel_normal(info), dir) >= 0.0 {
            continue;
        }
        let w3 = select(1.0 - f, f, off == vec3<u32>(1u));
        let w = w3.x * w3.y * w3.z + 1e-3;
        sum += voxel_stored(texel) * w;
        weight += w;
    }
    var radiance = own;
    if weight > 0.0 {
        radiance = sum / weight;
    }
    if level + 1 < i32(voxels.levels) {
        // How many voxels in from the edge of the level this is.
        let rel = index - voxels.origin[level].xyz;
        let inward = min(rel, vec3<i32>(n - 1) - rel);
        let fine = saturate(f32(min(inward.x, min(inward.y, inward.z))) / 8.0);
        if fine < 1.0 {
            let coarse = vec3<i32>(floor(p / (size * 2.0)));
            let coarse_rel = coarse - voxels.origin[level + 1].xyz;
            if all(coarse_rel >= vec3<i32>(0)) && all(coarse_rel < vec3<i32>(n)) {
                let texel = voxel_texel(coarse, u32(level + 1), voxels.resolution);
                if voxel_occupied(textureLoad(voxel_meta, texel, 0).r) {
                    radiance = mix(voxel_stored(texel), radiance, fine);
                }
            }
        }
    }
    return radiance;
}

/// Traces `origin + dir * [t0, t1]` through the world cache: always in the
/// finest level that covers where the ray is, a voxel at a time where there is
/// something and a block of voxels at a time where the level's occupancy
/// masks say there is nothing.
fn trace_voxels(origin: vec3<f32>, dir: vec3<f32>, t0: f32, t1: f32) -> vec4<f32> {
    if voxels.levels == 0u {
        return MISS;
    }
    let top = i32(voxels.levels) - 1;
    let inv_dir = 1.0 / select(dir, vec3<f32>(1e-12), abs(dir) < vec3<f32>(1e-12));
    let soft_occupancy = (voxels.options & VOXEL_SOFT_OCCUPANCY) != 0u;
    let soft_facing = (voxels.options & VOXEL_SOFT_FACING) != 0u;
    let blend_lookup = (voxels.options & VOXEL_BLEND_LOOKUP) != 0u;
    var t = t0;
    var level = 0;
    // What the ray has gathered so far, and how much of it is still to be
    // stopped. (All or nothing at the first voxel, unless an option softens it.)
    var gathered = vec3<f32>(0.0);
    var through = 1.0;
    // The ray may start inside the voxels of the surface it left, or of one
    // that already stopped the cascades below: it has to get clear first.
    var clear = false;
    for (var i = 0u; i < voxels.max_steps; i++) {
        if t >= t1 {
            break;
        }
        let p = origin + dir * t;

        // The finest level that covers the point.
        while level > 0 && voxel_level_covers(p, level - 1) {
            level -= 1;
        }
        while level <= top && !voxel_level_covers(p, level) {
            level += 1;
        }
        if level > top {
            // Beyond what the cache covers.
            break;
        }
        let size = voxels.voxel_size * f32(1u << u32(level));
        let index = vec3<i32>(floor(p / size));

        // What the ray crosses in this step: a voxel, or a block of them
        // that has nothing in it.
        var shift = 0u;
        if textureLoad(voxel_mask, voxel_mask_texel(index, u32(level), voxels.resolution, 4u), 0).r == 0u {
            shift = 4u;
        } else if textureLoad(voxel_mask, voxel_mask_texel(index, u32(level), voxels.resolution, 2u), 0).r == 0u {
            shift = 2u;
        }
        if shift != 0u {
            clear = true;
        } else {
            let texel = voxel_texel(index, u32(level), voxels.resolution);
            let info = textureLoad(voxel_meta, texel, 0).r;
            if !voxel_occupied(info) {
                clear = true;
            } else if clear {
                let facing = dot(voxel_normal(info), dir);
                // How much of the ray the voxel stops as a lit front, and as
                // the back of something. (Seen nearly edge on it is neither:
                // more likely the ray skimming a surface's own voxels.)
                var front = select(0.0, 1.0, facing < 0.0);
                var back = select(0.0, 1.0, facing > 0.25);
                if soft_facing {
                    // Gradually, so that a voxel whose normal sits on the
                    // line does not stop the ray one frame and not the next.
                    front = smoothstep(0.1, -0.1, facing);
                    back = smoothstep(0.1, 0.4, facing);
                }
                if soft_occupancy {
                    // As far as it is sure there is anything there: a voxel
                    // fades in as it is seen again, and out as it is seen
                    // through, instead of appearing and vanishing.
                    let sure = f32(voxel_confidence(info)) / f32(VOXEL_CONFIDENCE_MAX);
                    front *= sure;
                    back *= sure;
                }
                if front > 0.0 {
                    var stored = voxel_stored(texel);
                    if blend_lookup {
                        stored = voxel_blended(p, index, level, dir, stored);
                    }
                    gathered += through * front * stored * (view.exposure / VOXEL_RADIANCE_SCALE);
                }
                through *= saturate(1.0 - front - back);
                if through < 0.02 {
                    return vec4<f32>(gathered, 0.0);
                }
            }
        }

        // On to where the ray leaves it.
        let cell_size = size * f32(1u << shift);
        let cell = index >> vec3<u32>(shift);
        let exit = (vec3<f32>(cell) + vec3<f32>(dir > vec3<f32>(0.0))) * cell_size;
        // (Along an axis the ray does not move on, it never leaves: a
        // direction that lies exactly in a plane of the grid, as two in
        // every row of the cascades' do, would otherwise find its exit
        // behind it and creep forward a hundredth of a voxel a step.)
        let to_exit = select((exit - p) * inv_dir, vec3<f32>(1.0e30), abs(dir) < vec3<f32>(1e-12));
        var advance = max(min(to_exit.x, min(to_exit.y, to_exit.z)), 0.0);
        if shift != 0u && level > 0 {
            // An empty block of this level may reach into a finer one, which
            // knows better what is there.
            advance = min(advance, voxel_level_entry(p, inv_dir, level - 1));
        }
        t += advance + size * 0.01;
    }
    return vec4<f32>(gathered, through);
}
