//! Builds the depth pyramid the rays are traced against. Each texel holds the
//! closest (r) and farthest (g) depth of the pixels below it, as reverse-z NDC
//! depth: closest is the largest value, and 0 is the background.

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(8, 8, 1)
fn first(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(dst)) {
        return;
    }
    let depth = textureLoad(depth_tex, vec2<i32>(id.xy), 0);
    textureStore(dst, vec2<i32>(id.xy), vec4<f32>(depth, depth, 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn reduce(@builtin(global_invocation_id) id: vec3<u32>) {
    let dst_size = textureDimensions(dst);
    if any(id.xy >= dst_size) {
        return;
    }
    let src_size = vec2<i32>(textureDimensions(src));
    let base = vec2<i32>(id.xy) * 2;
    // An odd source leaves one row or column over: the last texel takes it.
    let extent = vec2<i32>(2) + vec2<i32>((src_size & vec2<i32>(1)) * vec2<i32>(id.xy + 1u == dst_size));
    var closest = 0.0;
    var farthest = 1.0e30;
    for (var y = 0; y < extent.y; y++) {
        for (var x = 0; x < extent.x; x++) {
            let d = textureLoad(src, min(base + vec2<i32>(x, y), src_size - 1), 0).rg;
            closest = max(closest, d.r);
            farthest = min(farthest, d.g);
        }
    }
    textureStore(dst, vec2<i32>(id.xy), vec4<f32>(closest, farthest, 0.0, 0.0));
}
