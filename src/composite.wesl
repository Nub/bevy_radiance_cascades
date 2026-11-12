//! Adds the gathered indirect light onto the lit scene (the blend state decides
//! whether it is added or, for the debug views, replaces the scene).

import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;

@group(0) @binding(0) var indirect: texture_2d<f32>;
/// The emitters' highlights (black in the debug views that are not lighting).
@group(0) @binding(1) var highlight: texture_2d<f32>;
/// What each pixel mirrors of the cascades' light, and in the alpha of
/// `cover` how much of what it mirrors is in the image already (see
/// `RadianceCascadesSpecularCover`): the rest of it is added here. Without
/// one, `cover` is `specular` itself, whose alpha is 1 wherever it is not
/// black, and nothing is.
@group(0) @binding(2) var specular: texture_2d<f32>;
@group(0) @binding(3) var cover: texture_2d<f32>;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(in.position.xy);
    let covered = saturate(textureLoad(cover, min(pix, vec2<i32>(textureDimensions(cover)) - 1), 0).a);
    let mirrored = textureLoad(specular, pix, 0).rgb * (1.0 - covered);
    return vec4<f32>(textureLoad(indirect, pix, 0).rgb + textureLoad(highlight, pix, 0).rgb + mirrored, 1.0);
}
