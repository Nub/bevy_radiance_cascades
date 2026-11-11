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
    /// Last frame's (unjittered) view.
    prev_clip_from_world: mat4x4<f32>,
    resolution: u32,
    levels: u32,
    update_level: u32,
    max_steps: u32,
    voxel_size: f32,
    blend: f32,
    /// Last frame's exposed radiance (the indirect light history) to this frame's.
    history_scale: f32,
    /// `VOXEL_*` flags: see `WorldCache`.
    options: u32,
    /// Whether the camera has a motion vector prepass.
    has_motion_vectors: u32,
}

const VOXEL_SKIP_MOVING: u32 = 1u;
const VOXEL_SOFT_OCCUPANCY: u32 = 2u;
const VOXEL_BLEND_LOOKUP: u32 = 4u;
const VOXEL_COARSE_FROM_FINE: u32 = 8u;
const VOXEL_CLAMP_OUTLIERS: u32 = 16u;
const VOXEL_SOFT_FACING: u32 = 32u;

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

/// Set on a voxel that rays find. (One that is not may still hold what has
/// been seen there so far: see `VOXEL_CONFIDENCE_MASK`.)
const VOXEL_OCCUPIED: u32 = 0x80000000u;
/// How sure it is that a surface is in a voxel, 0 to 7: up when the camera
/// sees one there, down when it sees clean through. Along the edges of things
/// and on surfaces seen at a slant, a voxel is seen one frame and seen through
/// the next as the view is jittered or creeps; were it to come and go with
/// each look, it would flash, and everything it lights with it. So it takes
/// more than one sighting for rays to start finding it, and more than one
/// look through it for them to stop.
const VOXEL_CONFIDENCE_SHIFT: u32 = 24u;
const VOXEL_CONFIDENCE_MASK: u32 = 0x07000000u;
const VOXEL_CONFIDENCE_MAX: u32 = 7u;

fn voxel_confidence(info: u32) -> u32 {
    return (info & VOXEL_CONFIDENCE_MASK) >> VOXEL_CONFIDENCE_SHIFT;
}

fn voxel_occupied(info: u32) -> bool {
    return (info & VOXEL_OCCUPIED) != 0u;
}

/// Voxels keep radiance in physical units (the lit scene divided by the
/// exposure it was rendered with), since they outlive changes of exposure.
/// Scaled to keep the brightest emitters within what rgb9e5 can hold.
const VOXEL_RADIANCE_SCALE: f32 = 0.0625;

/// A voxel's metadata: whether rays find it, how sure it is that a surface
/// is there (never 0: that is an empty voxel), and which way the surface faces.
fn pack_voxel_meta(normal: vec3<f32>, confidence: u32, occupied: bool) -> u32 {
    let oct = vec2<u32>(saturate(octahedral_encode(normal)) * 4095.0 + 0.5);
    let sure = clamp(confidence, 1u, VOXEL_CONFIDENCE_MAX) << VOXEL_CONFIDENCE_SHIFT;
    return select(0u, VOXEL_OCCUPIED, occupied) | sure | oct.x | (oct.y << 12u);
}

fn voxel_normal(info: u32) -> vec3<f32> {
    let oct = vec2<f32>(vec2<u32>(info & 0xFFFu, (info >> 12u) & 0xFFFu)) / 4095.0;
    return octahedral_decode(oct);
}
