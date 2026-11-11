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
import package::voxel::{VoxelParams, voxel_texel, voxel_mask_texel, voxel_normal, VOXEL_RADIANCE_SCALE};
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
    var t = t0;
    var level = 0;
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
            return MISS;
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
            if info == 0u {
                clear = true;
            } else if clear {
                let facing = dot(voxel_normal(info), dir);
                if facing < 0.0 {
                    let stored = rgb9e5_to_vec3_(textureLoad(voxel_radiance, texel, 0).r);
                    return vec4<f32>(stored * (view.exposure / VOXEL_RADIANCE_SCALE), 0.0);
                }
                if facing > 0.25 {
                    // The back of something. (Shallower than that, and it is more
                    // likely the ray skimming a surface's own voxels.)
                    return OCCLUDED;
                }
            }
        }

        // On to where the ray leaves it.
        let cell_size = size * f32(1u << shift);
        let cell = index >> vec3<u32>(shift);
        let exit = (vec3<f32>(cell) + vec3<f32>(dir > vec3<f32>(0.0))) * cell_size;
        let to_exit = (exit - p) * inv_dir;
        var advance = max(min(to_exit.x, min(to_exit.y, to_exit.z)), 0.0);
        if shift != 0u && level > 0 {
            // An empty block of this level may reach into a finer one, which
            // knows better what is there.
            advance = min(advance, voxel_level_entry(p, inv_dir, level - 1));
        }
        t += advance + size * 0.01;
    }
    return MISS;
}
