//! The world cache: a clipmap of voxels around the camera that remembers the
//! lit surfaces the camera has seen, so that rays which leave the screen or
//! pass behind something still find them.
//!
//! Level `l` is a cube of `resolution^3` voxels of size `voxel_size * 2^l`.
//! Voxels are addressed toroidally (texel = world index mod resolution), so
//! moving the camera moves no data: voxels that scroll out are reused by the
//! ones scrolling in. All levels live in one texture, stacked along z.
//!
//! Every level is filled from what the camera sees, on its own, and the levels
//! are brought up to date at different times: an empty voxel of a coarse level
//! says nothing about the finer voxels inside it. So a ray always reads the
//! finest level that covers where it is, and what lets it cross empty space
//! quickly is that level's own occupancy masks (see `voxel_mask_texel`).

import bevy_render::utils::{octahedral_encode, octahedral_decode};

const MAX_VOXEL_LEVELS: u32 = 8u;

struct VoxelParams {
    /// World index of each level's minimum corner voxel.
    origin: array<vec4<i32>, MAX_VOXEL_LEVELS>,
    /// Where the level being updated was when it was last updated.
    update_prev_origin: vec4<i32>,
    resolution: u32,
    levels: u32,
    update_level: u32,
    max_steps: u32,
    voxel_size: f32,
    blend: f32,
    /// Last frame's exposed radiance (the indirect light history) to this frame's.
    history_scale: f32,
}

/// The texel that holds the voxel with the given world index.
fn voxel_texel(index: vec3<i32>, level: u32, resolution: u32) -> vec3<i32> {
    let n = i32(resolution);
    let wrapped = ((index % n) + n) % n;
    return wrapped + vec3<i32>(0, 0, i32(level) * n);
}

/// Where the occupancy masks keep what is known of a block of voxels of a
/// level: whether any of them is occupied. Blocks are `1 << shift` voxels a
/// side, with `shift` 2 (fine) or 4 (coarse), and are addressed toroidally as
/// the voxels are. Both kinds share a texture: the coarse ones lie beside the
/// fine ones along x. A ray crosses a block that has nothing in it in one step.
fn voxel_mask_texel(index: vec3<i32>, level: u32, resolution: u32, shift: u32) -> vec3<i32> {
    let side = i32(resolution >> shift);
    let block = index >> vec3<u32>(shift);
    let wrapped = ((block % side) + side) % side;
    let fine_side = i32(resolution >> 2u);
    return wrapped + vec3<i32>(select(0, fine_side, shift == 4u), 0, i32(level) * fine_side);
}

const VOXEL_OCCUPIED: u32 = 0x80000000u;
/// How well established a voxel is, 1 to 3: one that has been seen again and
/// again takes more than a single look through it to empty.
const VOXEL_STRENGTH_SHIFT: u32 = 24u;
const VOXEL_STRENGTH_MASK: u32 = 0x03000000u;

fn voxel_strength(info: u32) -> u32 {
    return (info & VOXEL_STRENGTH_MASK) >> VOXEL_STRENGTH_SHIFT;
}

/// Voxels keep radiance in physical units (the lit scene divided by the
/// exposure it was rendered with), since they outlive changes of exposure.
/// Scaled to keep the brightest emitters within what rgb9e5 can hold.
const VOXEL_RADIANCE_SCALE: f32 = 0.0625;

/// A voxel's metadata: occupied, and which way its surface faces.
fn pack_voxel_meta(normal: vec3<f32>, strength: u32) -> u32 {
    let oct = vec2<u32>(saturate(octahedral_encode(normal)) * 4095.0 + 0.5);
    return VOXEL_OCCUPIED | (min(strength, 3u) << VOXEL_STRENGTH_SHIFT) | oct.x | (oct.y << 12u);
}

fn voxel_normal(info: u32) -> vec3<f32> {
    let oct = vec2<f32>(vec2<u32>(info & 0xFFFu, (info >> 12u) & 0xFFFu)) / 4095.0;
    return octahedral_decode(oct);
}
