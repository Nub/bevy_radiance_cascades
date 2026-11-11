//! Adds the gathered indirect light onto the lit scene (the blend state decides
//! whether it is added or, for the debug views, replaces the scene).

import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;

@group(0) @binding(0) var indirect: texture_2d<f32>;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(textureLoad(indirect, vec2<i32>(in.position.xy), 0).rgb, 1.0);
}
