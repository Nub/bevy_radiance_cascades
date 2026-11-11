//! Brings one level of the world cache up to date with what the camera sees
//! this frame. A voxel that has a visible surface in it takes that surface's
//! lit color; one the camera sees clean through is emptied; one that is
//! hidden or off screen keeps what it remembered.

import bevy_render::view::View;
import bevy_render::utils::octahedral_decode;
import bevy_pbr::deferred::types::unpack_24bit_normal;
import bevy_pbr::render::rgb9e5::{vec3_to_rgb9e5_, rgb9e5_to_vec3_};
import package::voxel::{
    VoxelParams, voxel_texel, pack_voxel_meta, voxel_normal, voxel_strength,
    VOXEL_RADIANCE_SCALE, VOXEL_STRENGTH_MASK, VOXEL_STRENGTH_SHIFT,
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

/// rgb9e5 tops out here.
const MAX_VOXEL_RADIANCE: f32 = 65000.0;

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

    var radiance = vec3<f32>(0.0);
    var info = 0u;
    if all(index == prev_index) {
        info = textureLoad(voxel_meta, texel).r;
        radiance = rgb9e5_to_vec3_(textureLoad(voxel_radiance, texel).r);
    }

    let size = voxels.voxel_size * f32(1u << level);
    let view_size = view.main_pass_viewport.zw;
    let near = view.clip_from_view[3][2];
    // The voxel is observed at the centres of its eight octants. Each stands
    // for a ball of this radius.
    let reach = size * 0.433;

    let to_stored = VOXEL_RADIANCE_SCALE / max(view.exposure, 1e-12);
    // Nothing to observe for a voxel that is entirely off screen.
    let center = (vec3<f32>(index) + 0.5) * size;
    var on_screen = true;
    for (var plane = 0u; plane < 5u; plane++) {
        let half_space = view.frustum[plane];
        on_screen = on_screen && dot(half_space.xyz, center) + half_space.w > -size;
    }

    var found = 0.0;
    var seen_through = 0u;
    var found_radiance = vec3<f32>(0.0);
    var found_normal = vec3<f32>(0.0);
    for (var k = 0u; k < select(0u, 8u, on_screen); k++) {
        let octant = vec3<f32>(vec3<u32>(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u));
        let p = (vec3<f32>(index) + 0.25 + 0.5 * octant) * size;
        let clip = view.clip_from_world * vec4<f32>(p, 1.0);
        if clip.w <= near {
            continue;
        }
        let ndc = clip.xy / clip.w;
        if any(abs(ndc) >= vec2<f32>(1.0)) {
            continue;
        }
        let pix = vec2<i32>((ndc * vec2<f32>(0.5, -0.5) + 0.5) * view_size);
        let depth = textureLoad(depth_tex, pix, 0);
        if depth <= 0.0 {
            // Nothing but background behind it.
            seen_through += 1u;
            continue;
        }
        // How far the point is from the surface seen through its pixel. Along
        // the surface's normal where there is one: measured along the view ray,
        // a surface seen at a grazing angle would seem far from its own voxels.
        let g = textureLoad(gbuffer_tex, pix, 0);
        if g.a == 0u {
            // Not the scene's geometry (what is drawn without the deferred
            // prepass: effects, overlays, the odd forward mesh). It is no
            // voxel, and what is behind it cannot be seen.
            continue;
        }
        let uv = (vec2<f32>(pix) + 0.5) / view_size;
        let surface = view.world_from_clip * vec4<f32>((uv - 0.5) * vec2<f32>(2.0, -2.0), depth, 1.0);
        let normal = octahedral_decode(unpack_24bit_normal(g.a));
        let distance = dot(normal, p - surface.xyz / surface.w);
        // Near the surface's plane is not enough: the plane goes on where the
        // surface does not (past the edge of a table seen from the side), and
        // would leave voxels hanging in the air there. What is seen through
        // the pixel has to be about here along the view as well.
        let along_view = abs(near / depth - clip.w);
        if abs(distance) <= reach && along_view <= size * 3.0 {
            // Lit color now, over as much of the screen as this part of the
            // voxel covers: a single pixel of it is a different one every time
            // the view moves or is jittered, and the voxel would flicker
            // between them. (Level 0 of the color pyramid is two pixels a texel.)
            let width = size * 0.25 * view.clip_from_view[1][1] * view_size.y / clip.w;
            let lod = clamp(log2(max(width, 1.0)) - 1.0, 0.0, f32(textureNumLevels(color_pyramid)) - 1.0);
            let filtered = textureSampleLevel(color_pyramid, linear_sampler, (vec2<f32>(pix) + 0.5) * 0.5 / vec2<f32>(textureDimensions(color_pyramid, 0)), lod);
            var color = textureLoad(scene_color, pix, 0).rgb;
            if filtered.a > 0.05 {
                color = filtered.rgb / filtered.a;
            }
            // Plus the indirect light it had last frame.
            color += voxels.history_scale * textureLoad(history, pix, 0).rgb;
            color = clamp(color * to_stored, vec3<f32>(0.0), vec3<f32>(MAX_VOXEL_RADIANCE));
            found_radiance += color;
            found_normal += normal;
            found += 1.0;
        } else if clip.w < near / depth {
            seen_through += 1u;
        }
    }

    if found > 0.0 {
        let observed = found_radiance / found;
        // Blend, so a voxel does not flicker between the pixels that land in it.
        radiance = select(observed, mix(radiance, observed, voxels.blend), info != 0u);
        let length_sq = dot(found_normal, found_normal);
        var normal = select(normalize(view.world_position - center), found_normal * inverseSqrt(length_sq), length_sq > 1e-6);
        if info != 0u {
            // Which way it faces settles like its color does, or rays would
            // take it for a front one update and a back the next.
            let settled = mix(voxel_normal(info), normal, voxels.blend);
            let settled_sq = dot(settled, settled);
            normal = select(normal, settled * inverseSqrt(settled_sq), settled_sq > 1e-6);
        }
        // Seen again: better established, up to a point.
        info = pack_voxel_meta(normal, select(2u, voxel_strength(info) + 1u, info != 0u));
    } else if seen_through == 8u && info != 0u {
        // Seen clean through. Along the edges of things a voxel is seen and
        // then seen through from one frame to the next, as the view is
        // jittered or creeps: one that emptied at the first look through it
        // would flash. The coarse levels are looked at too seldom to wait.
        let strength = voxel_strength(info);
        let drop = 1u << min(level, 2u);
        if strength <= drop {
            info = 0u;
            radiance = vec3<f32>(0.0);
        } else {
            info = (info & ~VOXEL_STRENGTH_MASK) | ((strength - drop) << VOXEL_STRENGTH_SHIFT);
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
    return select(0u, 1u, any_info != 0u);
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
