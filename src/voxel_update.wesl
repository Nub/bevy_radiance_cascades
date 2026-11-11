//! Brings one level of the world cache up to date with what the camera sees
//! this frame. A voxel that has a visible surface in it takes that surface's
//! lit color; one the camera sees clean through is emptied; one that is
//! hidden or off screen keeps what it remembered.
//!
//! In two steps. `mark`: every pixel of the scene's geometry marks the voxel
//! its surface lies in. `update`: every voxel that was marked has a surface in
//! it, and the rest look whether the camera sees through them. (Asking each
//! voxel instead whether a surface shows at a few points of it, as this once
//! did, misses what lies between the points and, where a surface only just
//! reaches a point, finds it one frame and not the next.)

import bevy_render::view::View;
import bevy_render::utils::octahedral_decode;
import bevy_pbr::deferred::types::unpack_24bit_normal;
import bevy_pbr::render::rgb9e5::{vec3_to_rgb9e5_, rgb9e5_to_vec3_};
import package::voxel::{
    VoxelParams, voxel_texel, pack_voxel_meta, voxel_normal, voxel_confidence, voxel_occupied,
    VOXEL_RADIANCE_SCALE, VOXEL_OCCUPIED, VOXEL_CONFIDENCE_MASK, VOXEL_CONFIDENCE_SHIFT, VOXEL_CONFIDENCE_MAX,
    VOXEL_SKIP_MOVING, VOXEL_COARSE_FROM_FINE, VOXEL_CLAMP_OUTLIERS,
};

@group(0) @binding(0) var<uniform> voxels: VoxelParams;
@group(0) @binding(1) var<uniform> view: View;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var gbuffer_tex: texture_2d<u32>;
@group(0) @binding(4) var scene_color: texture_2d<f32>;
@group(0) @binding(5) var history: texture_2d<f32>;
@group(0) @binding(6) var voxel_radiance: texture_storage_3d<r32uint, read_write>;
@group(0) @binding(7) var voxel_meta: texture_storage_3d<r32uint, read_write>;
/// The prefiltered lit scene (see `color_pyramid.wesl`).
@group(0) @binding(8) var color_pyramid: texture_2d<f32>;
@group(0) @binding(9) var linear_sampler: sampler;
/// Which blocks of voxels have anything in them (see `voxel_mask_texel`).
@group(0) @binding(10) var voxel_mask: texture_storage_3d<r32uint, write>;
/// For each texel of the level being updated: 0, or one more than the index
/// of a pixel whose surface lies in its voxel.
@group(0) @binding(11) var<storage, read_write> marks: array<atomic<u32>>;
/// How far each pixel has moved across the view since last frame, in UV.
@group(0) @binding(12) var motion_vectors: texture_2d<f32>;

/// rgb9e5 tops out here.
const MAX_VOXEL_RADIANCE: f32 = 65000.0;
/// How far, in pixels, something may move across the view on its own in a
/// frame and still be taken for part of the standing world.
const STILL: f32 = 0.75;

fn world_at(pix: vec2<i32>, depth: f32) -> vec3<f32> {
    let uv = (vec2<f32>(pix) + 0.5) / view.main_pass_viewport.zw;
    let p = view.world_from_clip * vec4<f32>((uv - 0.5) * vec2<f32>(2.0, -2.0), depth, 1.0);
    return p.xyz / p.w;
}

/// Whether what shows at a pixel, at `p`, is moving on its own: across the
/// view otherwise than the camera's own movement takes the world. What moves
/// is left out of the cache. It would only ever be remembered where it no
/// longer is, and the screen shows where it is for as long as it is in view.
fn moving(pix: vec2<i32>, p: vec3<f32>) -> bool {
    if voxels.has_motion_vectors == 0u || (voxels.options & VOXEL_SKIP_MOVING) == 0u {
        return false;
    }
    let now = view.unjittered_clip_from_world * vec4<f32>(p, 1.0);
    let before = voxels.prev_clip_from_world * vec4<f32>(p, 1.0);
    if now.w <= 0.0 || before.w <= 0.0 {
        return false;
    }
    let with_camera = (now.xy / now.w - before.xy / before.w) * vec2<f32>(0.5, -0.5);
    let motion = textureLoad(motion_vectors, pix, 0).rg;
    return length((motion - with_camera) * view.main_pass_viewport.zw) > STILL;
}

/// Every pixel that shows the scene's standing geometry marks the voxel (of
/// the level being updated) that its surface lies in.
@compute @workgroup_size(8, 8, 1)
fn mark(@builtin(global_invocation_id) id: vec3<u32>) {
    let view_size = vec2<u32>(view.main_pass_viewport.zw);
    if any(id.xy >= view_size) {
        return;
    }
    let pix = vec2<i32>(id.xy);
    let depth = textureLoad(depth_tex, pix, 0);
    // Not the background, nor what is drawn without the deferred prepass
    // (effects, overlays, the odd forward mesh).
    if depth <= 0.0 || textureLoad(gbuffer_tex, pix, 0).a == 0u {
        return;
    }
    let p = world_at(pix, depth);
    if moving(pix, p) {
        return;
    }
    let n = i32(voxels.resolution);
    let level = voxels.update_level;
    let size = voxels.voxel_size * f32(1u << level);
    let index = vec3<i32>(floor(p / size));
    let rel = index - voxels.origin[level].xyz;
    if any(rel < vec3<i32>(0)) || any(rel >= vec3<i32>(n)) {
        return;
    }
    let local = ((index % n) + n) % n;
    // (The greatest index of them all: the same pixel from frame to frame,
    // while the view is still.)
    atomicMax(&marks[u32(local.x + n * (local.y + n * local.z))], id.x + id.y * view_size.x + 1u);
}

@compute @workgroup_size(4, 4, 4)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = i32(voxels.resolution);
    if any(id >= vec3<u32>(voxels.resolution)) {
        return;
    }
    let level = voxels.update_level;
    let local = vec3<i32>(id);
    let origin = voxels.origin[level].xyz;
    let prev_origin = voxels.update_prev_origin.xyz;
    // The voxel this texel stands for now, and stood for at the last update.
    let index = origin + (((local - origin) % n) + n) % n;
    let prev_index = prev_origin + (((local - prev_origin) % n) + n) % n;
    let texel = voxel_texel(index, level, voxels.resolution);
    let marked = atomicLoad(&marks[u32(local.x + n * (local.y + n * local.z))]);

    var radiance = vec3<f32>(0.0);
    var info = 0u;
    if all(index == prev_index) {
        info = textureLoad(voxel_meta, texel).r;
        radiance = rgb9e5_to_vec3_(textureLoad(voxel_radiance, texel).r);
    } else if marked == 0u {
        // Scrolled in, and nothing shows in it.
        textureStore(voxel_meta, texel, vec4<u32>(0u));
        textureStore(voxel_radiance, texel, vec4<u32>(0u));
        return;
    }
    if marked == 0u && info == 0u {
        // Empty, and still empty.
        return;
    }

    let size = voxels.voxel_size * f32(1u << level);
    let view_size = view.main_pass_viewport.zw;
    let near = view.clip_from_view[3][2];
    let center = (vec3<f32>(index) + 0.5) * size;

    if marked != 0u {
        // A surface shows in it, at this pixel among others.
        let width = u32(view_size.x);
        let pix = vec2<i32>(vec2<u32>((marked - 1u) % width, (marked - 1u) / width));
        let distance = near / max(textureLoad(depth_tex, pix, 0), 1e-12);
        // Lit color now, over as much of the screen as a part of the voxel
        // covers: a single pixel of it is a different one every time the view
        // moves or is jittered, and the voxel would flicker between them.
        // (Level 0 of the color pyramid is two pixels a texel.)
        let across = size * 0.25 * view.clip_from_view[1][1] * view_size.y / distance;
        let lod = clamp(log2(max(across, 1.0)) - 1.0, 0.0, f32(textureNumLevels(color_pyramid)) - 1.0);
        let filtered = textureSampleLevel(color_pyramid, linear_sampler, (vec2<f32>(pix) + 0.5) * 0.5 / vec2<f32>(textureDimensions(color_pyramid, 0)), lod);
        var color = textureLoad(scene_color, pix, 0).rgb;
        if filtered.a > 0.05 {
            color = filtered.rgb / filtered.a;
        }
        // Plus the indirect light it had last frame.
        color += voxels.history_scale * textureLoad(history, pix, 0).rgb;
        let to_stored = VOXEL_RADIANCE_SCALE / max(view.exposure, 1e-12);
        var observed = clamp(color * to_stored, vec3<f32>(0.0), vec3<f32>(MAX_VOXEL_RADIANCE));
        if info != 0u && (voxels.options & VOXEL_CLAMP_OUTLIERS) != 0u {
            // No more than so much brighter or darker in one look: a glint that
            // lands on the pixel for a frame does not make the voxel pulse.
            observed = clamp(observed, radiance * 0.25, radiance * 4.0 + 1.0);
        }
        // Blend, so a voxel does not flicker between the pixels that land in it.
        radiance = select(observed, mix(radiance, observed, voxels.blend), info != 0u);

        var normal = octahedral_decode(unpack_24bit_normal(textureLoad(gbuffer_tex, pix, 0).a));
        if info != 0u {
            // Which way it faces settles like its color does, or rays would
            // take it for a front one update and a back the next.
            let settled = mix(voxel_normal(info), normal, voxels.blend);
            let settled_sq = dot(settled, settled);
            normal = select(normal, settled * inverseSqrt(settled_sq), settled_sq > 1e-6);
        }
        // Seen: surer of it, and once sure enough rays find it. The fine
        // levels, which are looked at every few frames, wait for a second
        // sighting; the coarse ones are looked at too seldom to.
        let confidence = min(voxel_confidence(info) + 2u, VOXEL_CONFIDENCE_MAX);
        let enough = select(2u, 4u, level <= 2u);
        info = pack_voxel_meta(normal, confidence, voxel_occupied(info) || confidence >= enough);
    } else {
        // Nothing shows in it. Does the camera see clean through it? Asked at
        // the centres of its eight octants, of all of which it has to be true.
        var on_screen = true;
        for (var plane = 0u; plane < 5u; plane++) {
            let half_space = view.frustum[plane];
            on_screen = on_screen && dot(half_space.xyz, center) + half_space.w > -size;
        }
        var seen_through = 0u;
        for (var k = 0u; k < select(0u, 8u, on_screen); k++) {
            let octant = vec3<f32>(vec3<u32>(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u));
            let p = (vec3<f32>(index) + 0.25 + 0.5 * octant) * size;
            let clip = view.clip_from_world * vec4<f32>(p, 1.0);
            if clip.w <= near {
                break;
            }
            let ndc = clip.xy / clip.w;
            if any(abs(ndc) >= vec2<f32>(1.0)) {
                break;
            }
            let pix = vec2<i32>((ndc * vec2<f32>(0.5, -0.5) + 0.5) * view_size);
            let depth = textureLoad(depth_tex, pix, 0);
            if depth > 0.0 {
                // Something shows behind the point, or in front of it. In
                // front, it hides the point. And what is not the scene's
                // standing geometry (see `mark`) says nothing of what is
                // behind it either way: it may be standing in the voxel.
                if clip.w >= near / depth || textureLoad(gbuffer_tex, pix, 0).a == 0u || moving(pix, world_at(pix, depth)) {
                    break;
                }
            }
            seen_through += 1u;
        }
        if seen_through == 8u {
            // Seen clean through: less sure of it, and gone once not at all.
            // (A sighting counts for two looks through, so a voxel that is
            // seen as often as not stays.) The coarser the level the fewer
            // looks it takes, so that what has moved away is forgotten about
            // as fast in all of them.
            let confidence = voxel_confidence(info);
            let drop = select(VOXEL_CONFIDENCE_MAX, level + 1u, level <= 2u);
            if confidence <= drop {
                info = 0u;
                radiance = vec3<f32>(0.0);
            } else {
                info = (info & ~VOXEL_CONFIDENCE_MASK) | ((confidence - drop) << VOXEL_CONFIDENCE_SHIFT);
            }
        }
    }

    if info != 0u && level > 0u && (voxels.options & VOXEL_COARSE_FROM_FINE) != 0u {
        // Where the next finer level covers this voxel, its color is the mean
        // of what that level holds in it, so that the two agree on it.
        let finer = level - 1u;
        var sum = vec3<f32>(0.0);
        var count = 0.0;
        for (var k = 0u; k < 8u; k++) {
            let child = index * 2 + vec3<i32>(vec3<u32>(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u));
            let rel = child - voxels.origin[finer].xyz;
            if any(rel < vec3<i32>(0)) || any(rel >= vec3<i32>(n)) {
                continue;
            }
            let child_texel = voxel_texel(child, finer, voxels.resolution);
            if voxel_occupied(textureLoad(voxel_meta, child_texel).r) {
                sum += rgb9e5_to_vec3_(textureLoad(voxel_radiance, child_texel).r);
                count += 1.0;
            }
        }
        if count > 0.0 {
            radiance = sum / count;
        }
    }

    textureStore(voxel_meta, texel, vec4<u32>(info, 0u, 0u, 0u));
    textureStore(voxel_radiance, texel, vec4<u32>(vec3_to_rgb9e5_(radiance), 0u, 0u, 0u));
}

/// Whether any voxel of the level being updated is occupied, in a block of
/// `side` voxels a side whose first texel is `base`.
fn block_occupied(base: vec3<i32>, side: i32) -> u32 {
    var any_info = 0u;
    for (var z = 0; z < side; z++) {
        for (var y = 0; y < side; y++) {
            for (var x = 0; x < side; x++) {
                any_info |= textureLoad(voxel_meta, base + vec3<i32>(x, y, z)).r;
            }
        }
    }
    return select(0u, 1u, (any_info & VOXEL_OCCUPIED) != 0u);
}

/// Brings the fine occupancy mask of the level just updated up to date: one
/// thread a block of 4 voxels a side. (Texels wrap as voxels do and the side
/// of a level is a multiple of the block's, so a block of texels is a block
/// of voxels.)
@compute @workgroup_size(4, 4, 4)
fn mask_fine(@builtin(global_invocation_id) id: vec3<u32>) {
    let side = voxels.resolution >> 2u;
    if any(id >= vec3<u32>(side)) {
        return;
    }
    let level = i32(voxels.update_level);
    let block = vec3<i32>(id);
    let occupied = block_occupied(block * 4 + vec3<i32>(0, 0, level * i32(voxels.resolution)), 4);
    textureStore(voxel_mask, block + vec3<i32>(0, 0, level * i32(side)), vec4<u32>(occupied, 0u, 0u, 0u));
}

/// The same for the coarse mask: blocks of 16 voxels a side.
@compute @workgroup_size(4, 4, 4)
fn mask_coarse(@builtin(global_invocation_id) id: vec3<u32>) {
    let side = voxels.resolution >> 4u;
    if any(id >= vec3<u32>(side)) {
        return;
    }
    let level = i32(voxels.update_level);
    let fine_side = i32(voxels.resolution >> 2u);
    let block = vec3<i32>(id);
    let occupied = block_occupied(block * 16 + vec3<i32>(0, 0, level * i32(voxels.resolution)), 16);
    textureStore(voxel_mask, block + vec3<i32>(fine_side, 0, level * fine_side), vec4<u32>(occupied, 0u, 0u, 0u));
}
