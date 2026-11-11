//! Builds the prefiltered copy of the lit scene that rays read what they hit
//! from. A ray stands for a cone, so it should see the average of the pixels
//! its cone covers: sampling single pixels instead makes the lighting shimmer
//! as the camera moves, wherever the image has bright detail.
//!
//! Level 0 is half the resolution of the view. Alpha is the fraction of each
//! texel that is covered by geometry and rgb is premultiplied by it, so that
//! the background does not bleed into the surfaces in front of it.

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

/// Keeps a stray pixel from dominating everything it is averaged with, and
/// infinities and NaNs out altogether.
const MAX_RADIANCE: f32 = 60000.0;

@compute @workgroup_size(8, 8, 1)
fn first(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(dst)) {
        return;
    }
    let base = vec2<i32>(id.xy) * 2;
    var sum = vec4<f32>(0.0);
    for (var k = 0; k < 4; k++) {
        let pix = base + vec2<i32>(k & 1, k >> 1);
        if textureLoad(depth_tex, pix, 0) > 0.0 {
            let color = clamp(textureLoad(src, pix, 0).rgb, vec3<f32>(0.0), vec3<f32>(MAX_RADIANCE));
            sum += vec4<f32>(color, 1.0);
        }
    }
    textureStore(dst, vec2<i32>(id.xy), sum * 0.25);
}

@compute @workgroup_size(8, 8, 1)
fn reduce(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(dst)) {
        return;
    }
    let max_src = vec2<i32>(textureDimensions(src)) - 1;
    let base = vec2<i32>(id.xy) * 2;
    var sum = vec4<f32>(0.0);
    for (var k = 0; k < 4; k++) {
        sum += textureLoad(src, min(base + vec2<i32>(k & 1, k >> 1), max_src), 0);
    }
    textureStore(dst, vec2<i32>(id.xy), sum * 0.25);
}
