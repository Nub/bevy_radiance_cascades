//! Radiance cascades: diffuse global illumination for Bevy without ray tracing
//! hardware and without noise.
//!
//! The technique is Alexander Sannikov's radiance cascades, in the screen-space
//! form presented in "Rendering Path of Exile 2" (ExileCon 2023). Probes are
//! placed on the depth buffer. Cascade `n` has probes every `spacing * 2^n`
//! pixels, each storing `(resolution * 2^n)^2` directions, and only traces the
//! distance interval that this angular resolution can resolve. Far light is
//! therefore sampled with many directions from few places, near light with few
//! directions from many places, and every cascade costs the same. Merging the
//! cascades top-down yields a complete, converged radiance field every frame.
//!
//! Rays are traced against the depth buffer where it can answer. Where it
//! cannot (off the screen, behind what is on it) they continue through the
//! [`WorldCache`], a clipmap of voxels that remembers the lit surfaces the
//! camera has seen.
//!
//! Add [`RadianceCascadesPlugin`] to the app and [`RadianceCascades`] to a 3D
//! camera. See the `room` example.
//!
//! # Limitations
//!
//! * The world cache only knows what has been on screen at some point. Rays
//!   that find nothing fall back to [`RadianceCascades::sky_color`].
//! * Diffuse only: specular reflections still come from whatever you already use.
//! * Only meshes rendered through the deferred prepass receive indirect light
//!   (the plugin makes deferred the default for opaque materials). Meshes drawn
//!   forward still occlude and bounce it.
//! * Perspective projections only, and the camera viewport must start at the
//!   origin of its render target.

mod node;
mod prepare;

use bevy::{
    app::{App, Plugin},
    asset::embedded_asset,
    camera::{Camera, Hdr},
    color::LinearRgba,
    core_pipeline::{
        core_3d::{main_opaque_pass_3d, main_transparent_pass_3d},
        prepass::{DeferredPrepass, DepthPrepass},
        schedule::{Core3d, Core3dSystems},
    },
    ecs::{
        component::Component,
        reflect::ReflectComponent,
        schedule::IntoScheduleConfigs,
        system::{Commands, ResMut},
    },
    pbr::DefaultOpaqueRendererMethod,
    reflect::{Reflect, std_traits::ReflectDefault},
    render::{
        ExtractSchedule, MainWorld, Render, RenderApp, RenderStartup, RenderSystems,
        sync_world::RenderEntity,
    },
    shader::load_shader_library,
};

pub use prepare::RadianceCascadesResources;

/// Most cascades a view can have. Six to eight cover any sensible resolution:
/// each starts twice as far from its probe as the one before, and once that is
/// further than the screen is wide there is nothing left on it to find. Past
/// that point every cascade also costs four times the one before (there is one
/// probe left, and its directions keep doubling).
pub const MAX_CASCADES: u32 = 12;

/// Adds radiance cascades global illumination.
///
/// Makes deferred rendering the default for opaque materials, since indirect
/// light is applied from the G-buffer.
pub struct RadianceCascadesPlugin;

impl Plugin for RadianceCascadesPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "voxel.wesl");
        load_shader_library!(app, "common.wesl");
        embedded_asset!(app, "voxel_update.wesl");
        embedded_asset!(app, "pyramid.wesl");
        embedded_asset!(app, "color_pyramid.wesl");
        embedded_asset!(app, "cascade.wesl");
        embedded_asset!(app, "gather.wesl");
        embedded_asset!(app, "composite.wesl");

        app.register_type::<RadianceCascades>()
            .insert_resource(DefaultOpaqueRendererMethod::deferred());
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(RenderStartup, node::init_radiance_cascades_pipelines)
            .add_systems(ExtractSchedule, extract_radiance_cascades)
            .add_systems(
                Render,
                prepare::prepare_radiance_cascades_resources
                    .in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Core3d,
                node::radiance_cascades
                    .after(main_opaque_pass_3d)
                    .before(main_transparent_pass_3d)
                    .in_set(Core3dSystems::MainPass),
            );
    }
}

/// What the camera shows. Anything but [`Off`](Self::Off) replaces the image.
#[derive(Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[reflect(Clone, PartialEq, Default)]
pub enum RadianceCascadesDebug {
    /// The scene with indirect light added.
    #[default]
    Off,
    /// The indirect light as it is added to the scene (tinted by surface color).
    Indirect,
    /// The incoming indirect light alone, as a white surface would reflect it.
    Irradiance,
    /// What the cascades know of the scene. Green surfaces are in the G-buffer
    /// and receive indirect light; red ones were drawn forward, so they occlude
    /// and bounce light but receive none; black is absent from the depth
    /// buffer, and so invisible to the rays. Bands mark depth.
    Surfaces,
    /// The contents of the world cache, as if it were the scene.
    WorldCache,
}

impl RadianceCascadesDebug {
    /// Whether the view shows the indirect light itself (as opposed to a
    /// diagnostic drawn in its place).
    pub(crate) fn is_lighting(self) -> bool {
        matches!(self, Self::Off | Self::Indirect)
    }
}

/// Enables radiance cascades global illumination on a 3D camera.
///
/// The indirect light replaces ambient light, so set the ambient light's
/// brightness to zero, and leave any environment map's diffuse part out.
///
/// Requires `Msaa::Off`, which Bevy enforces for deferred cameras anyway.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, Default, Clone)]
#[require(Hdr, DeferredPrepass, DepthPrepass)]
pub struct RadianceCascades {
    /// Distance in pixels between the probes of the first cascade. A power of
    /// two. Lower is sharper and proportionally more expensive: the cost of
    /// every cascade is `(base_resolution / probe_spacing)^2` rays per pixel.
    pub probe_spacing: u32,
    /// Side length of the first cascade's direction map, so it traces
    /// `base_resolution^2` directions per probe. A power of two.
    pub base_resolution: u32,
    /// Number of cascades. Each one doubles the reach of the previous; the last
    /// one always continues to the edge of the screen.
    pub cascade_count: u32,
    /// Length of the first cascade's rays, in probe spacings. Cascade `n`
    /// covers `interval_length * [2^n - 1, 2^(n+1) - 1]` spacings.
    ///
    /// Each cascade borrows its far light from probes of the next that sit up
    /// to a spacing away, as if they stood in its own place. The longer the
    /// intervals are compared to the spacing, the less that matters, and the
    /// steadier the lighting is as the camera moves; the shorter, the sharper
    /// the directions are resolved.
    pub interval_length: f32,
    /// Most depth pyramid lookups a ray may spend finding what it hits. Rays
    /// are traced exactly (no occluder is stepped over), so this only bounds
    /// the cost of rays that skim along geometry; those that run out pass on.
    pub max_steps: u32,
    /// How thick, in world units, a surface in the depth buffer is assumed to
    /// be. Rays further behind a surface than this pass behind it.
    pub thickness: f32,
    /// How far probes are lifted off their surface, in pixels at their depth.
    pub normal_bias: f32,
    /// How far behind a surface a ray has to be before it counts as a hit, as a
    /// fraction of its depth. Avoids surfaces shadowing themselves.
    pub depth_bias: f32,
    /// Scales the indirect light.
    pub intensity: f32,
    /// How much of last frame's indirect light the surfaces re-emit. 1 gives
    /// physically correct infinite bounces (converging over a few frames), 0
    /// gives a single bounce.
    pub bounce: f32,
    /// How much of each frame's incoming light the light shown takes on; the
    /// rest is what had accumulated at the same place in the world. 1 shows
    /// every frame as traced, which is steady for a view that is not jittered.
    /// Under temporal anti-aliasing or upscaling (TAA, DLSS, FSR) the view is
    /// jittered, small lights are drawn differently every frame, and what they
    /// light flickers unless it is accumulated like the image is. Lower is
    /// steadier, and slower to follow lights that change.
    pub temporal_blend: f32,
    /// How much of the camera's `ScreenSpaceAmbientOcclusion`, if it has one,
    /// darkens the indirect light: 0 none of it, 1 all. The probes are a few
    /// pixels apart, and what light a crease or the foot of a thing loses to
    /// its nearest surroundings is finer than they resolve.
    pub ambient_occlusion: f32,
    /// Radiance of everything that is not on screen.
    pub sky_color: LinearRgba,
    /// Luminance of [`Self::sky_color`] in cd/m^2, like an ambient light's
    /// brightness.
    pub sky_brightness: f32,
    /// The world cache: what rays find once they leave the screen or pass
    /// behind something. `None` makes the lighting purely screen-space.
    pub world_cache: Option<WorldCache>,
    /// Debug visualization.
    pub debug: RadianceCascadesDebug,
}

/// A clipmap of voxels around the camera that remembers the lit surfaces the
/// camera has seen, so that light and occluders persist once they are off
/// screen or hidden.
///
/// It only knows what has been on screen at some point: what the camera has
/// never looked at does not exist until it does.
#[derive(Reflect, Clone, Copy, Debug, PartialEq)]
#[reflect(Clone, PartialEq, Default)]
pub struct WorldCache {
    /// Size of the finest voxels, in world units. Walls thinner than this may
    /// leak light.
    pub voxel_size: f32,
    /// Voxels along each side of a level. A power of two.
    pub resolution: u32,
    /// Number of levels, up to 8. Each covers twice the extent of the one
    /// before it with voxels twice the size; the first covers
    /// `voxel_size * resolution`, so the defaults remember what is within
    /// 256 units of the camera, and 8 levels what is within a thousand.
    pub levels: u32,
    /// How much of a newly observed color a voxel takes on per update. Lower
    /// is steadier, and slower to follow lights that change.
    pub blend: f32,
    /// Most steps a ray takes: a voxel each where there is something, a
    /// block of voxels each where there is nothing.
    pub max_steps: u32,
}

impl Default for WorldCache {
    fn default() -> Self {
        Self {
            voxel_size: 0.125,
            resolution: 128,
            levels: 6,
            blend: 0.25,
            max_steps: 192,
        }
    }
}

impl Default for RadianceCascades {
    fn default() -> Self {
        Self {
            probe_spacing: 4,
            base_resolution: 4,
            cascade_count: 6,
            interval_length: 2.0,
            max_steps: 64,
            thickness: 0.5,
            normal_bias: 1.0,
            depth_bias: 0.002,
            intensity: 1.0,
            bounce: 1.0,
            temporal_blend: 0.1,
            ambient_occlusion: 1.0,
            sky_color: LinearRgba::WHITE,
            sky_brightness: 80.0,
            world_cache: Some(WorldCache::default()),
            debug: RadianceCascadesDebug::Off,
        }
    }
}

fn extract_radiance_cascades(mut main_world: ResMut<MainWorld>, mut commands: Commands) {
    let mut cameras = main_world.query::<(RenderEntity, &Camera, Option<&RadianceCascades>)>();

    for (entity, camera, radiance_cascades) in cameras.iter(&main_world) {
        let Ok(mut entity_commands) = commands.get_entity(entity) else {
            continue;
        };
        match radiance_cascades {
            Some(radiance_cascades) if camera.is_active => {
                entity_commands.insert(radiance_cascades.clone());
            }
            _ => {
                entity_commands.remove::<(RadianceCascades, RadianceCascadesResources)>();
            }
        }
    }
}
