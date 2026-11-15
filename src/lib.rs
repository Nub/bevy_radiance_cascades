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

mod emitters;
mod node;
mod prepare;

pub use emitters::{MAX_EMITTERS, RadianceEmitter};

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

pub use prepare::{RadianceCascadesResources, RadianceCascadesSpecularCover};

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

        app.register_type::<RadianceEmitter>()
            .register_type::<RadianceCascades>()
            .insert_resource(DefaultOpaqueRendererMethod::deferred());
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(RenderStartup, node::init_radiance_cascades_pipelines)
            .init_resource::<emitters::EmitterBuffer>()
            .add_systems(
                ExtractSchedule,
                (extract_radiance_cascades, emitters::extract_emitters),
            )
            .add_systems(
                Render,
                (
                    prepare::prepare_radiance_cascades_resources,
                    emitters::prepare_emitters,
                )
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
    /// How much of the light arriving at each surface it sends on to the eye,
    /// its diffuse and its specular side together.
    Reflectance,
    /// What the G-buffer says of each surface: red its roughness, green how
    /// metallic it is, blue its baked occlusion.
    Materials,
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
    /// How much of the light arriving at a surface its specular side reflects
    /// to the eye, on top of what its diffuse side does: 0 none, 1 all. A
    /// metal has no diffuse side, and without this the bounced light is never
    /// seen on it.
    pub specular: f32,
    /// The perceptual roughness over which [`Self::specular`] fades in. The
    /// arriving light is all the light from in front of the surface, which is
    /// what a rough surface reflects; a smooth one mirrors one direction, and
    /// is left to screen-space reflections. Where the camera has those, start
    /// this where they stop.
    pub specular_roughness: core::ops::Range<f32>,
    /// What the G-buffer's emissive is multiplied by to be in the units of
    /// the lit scene. 1 for Bevy as it is; a fork that stores it scaled (the
    /// format tops out near 65000) says by how much here.
    pub emissive_scale: f32,
    /// Whether the G-buffer's emissive is stored without the camera's
    /// exposure applied. Bevy stores it as the material has it, which the
    /// deferred lighting takes as exposed: false.
    pub emissive_unexposed: bool,
    /// Leave what surfaces mirror ([`Self::specular`]) out of the image, and
    /// write it to [`RadianceCascadesResources::specular`] instead, for every
    /// roughness: for a reflection pass that adds it where its own rays find
    /// nothing, in place of an environment map. Added here as well, a mirror
    /// of what is on screen would show both.
    pub specular_separate: bool,
    /// A uniform light arriving at every surface the cascades light, in lux:
    /// an ambient fill. It is bounced like any other light, so a closed room
    /// ends up with more than this by what its surfaces reflect. 0 for none:
    /// the cascades are the ambient light, and a fill on top is a look, as it
    /// is in a path tracer that adds one.
    pub ambient: f32,
    /// Trace the far cascades less often than every frame while the camera is
    /// still: the fourth and fifth every other frame, turn about, and those
    /// above every fourth (every one of them every frame while it moves). Far
    /// light changes little from one frame to the next at any one place on
    /// screen, and these cascades are most of the cost, as their rays are the
    /// ones that leave the screen for the world cache. A light that changes
    /// far off is a few frames late.
    pub amortize_far: bool,
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
    /// Voxels along each side of a level: a multiple of 16, up to 256. A voxel
    /// at some distance is between a quarter and a half of this many to that
    /// distance, so this is what sets how fine the cache is for how far off;
    /// memory and the cost of keeping it up to date go with its cube.
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
    /// Leave what moves on its own out of the cache (it takes the camera
    /// having a `MotionVectorPrepass` to tell). What moves would only be
    /// remembered where it no longer is.
    pub skip_moving: bool,
    /// Experimental. A voxel stops as much of a ray as the cache is sure
    /// there is a surface in it: it fades in as it is seen again and out as
    /// it is seen through, instead of appearing and vanishing.
    pub soft_occupancy: bool,
    /// Experimental. A ray takes the color of the voxels around where it
    /// hits, weighted by distance, and near the edge of a level that of the
    /// next coarser one as well: fewer blocks and steps in what the cache
    /// lights, for more texture reads per hit.
    pub blend_lookup: bool,
    /// Experimental. Where a finer level covers a voxel, its color is the mean
    /// of that level's voxels in it, so that the two agree.
    pub coarse_from_fine: bool,
    /// Experimental. A voxel's color changes by no more than a factor of four
    /// each time it is looked at: a glint does not make it pulse, and a light
    /// that comes on takes a few looks to show in full.
    pub clamp_outliers: bool,
    /// Experimental. Voxels seen nearly edge on stop a ray partly, by how
    /// they face it, instead of wholly or not at all on either side of a line.
    pub soft_facing: bool,
    /// Anchors: the middles of up to [`MAX_ANCHORS`] volumes that stand still
    /// in the world, each `resolution` voxels of `anchor_voxel_size` a side
    /// (the first `anchor_count` of these). A level follows the camera and
    /// forgets what it leaves behind; an anchor remembers what has been seen
    /// in its volume for good, coarsely. Rays do not read the anchors: a
    /// voxel of a level that comes into range with nothing showing in it
    /// takes what an anchor remembers of the place, where no finer level
    /// does. So a place that was looked at once, from anywhere (on a tour of
    /// the world behind a loading screen, see `prime`), lights what is near
    /// it from the first frame the camera stands in it, before it has turned
    /// to look.
    ///
    /// Each anchor costs the memory of a level, and one update in eight goes
    /// to the anchor nearest the camera in place of the finest level's.
    pub anchors: [bevy::math::Vec3; MAX_ANCHORS],
    pub anchor_count: u32,
    /// Size of the anchors' voxels, in world units.
    pub anchor_voxel_size: f32,
    /// Every update goes to the anchor nearest the camera, none to the
    /// levels: for filling the anchors quickly while the camera is taken
    /// round the world out of the player's sight. What a voxel takes in is
    /// the lit scene and the light bounced onto it last frame, so a view the
    /// camera has jumped to is shown for a frame with `hold` set before one
    /// with `prime`, and `RadianceCascades::temporal_blend` is 1 for as long
    /// (or what is taken in is a tenth of the bounced light).
    pub prime: bool,
    /// Nothing of the cache is brought up to date this frame; rays read it
    /// as it is.
    pub hold: bool,
}

/// The most anchors a [`WorldCache`] has.
pub const MAX_ANCHORS: usize = 4;

impl Default for WorldCache {
    fn default() -> Self {
        Self {
            voxel_size: 0.125,
            resolution: 128,
            levels: 6,
            blend: 0.25,
            max_steps: 192,
            skip_moving: true,
            soft_occupancy: false,
            blend_lookup: false,
            coarse_from_fine: false,
            clamp_outliers: false,
            soft_facing: false,
            anchors: [bevy::math::Vec3::ZERO; MAX_ANCHORS],
            anchor_count: 0,
            anchor_voxel_size: 1.0,
            prime: false,
            hold: false,
        }
    }
}

impl Default for RadianceCascades {
    fn default() -> Self {
        Self {
            probe_spacing: 4,
            base_resolution: 4,
            cascade_count: 4,
            interval_length: 2.0,
            max_steps: 64,
            thickness: 0.5,
            normal_bias: 1.0,
            depth_bias: 0.002,
            intensity: 1.0,
            bounce: 1.0,
            temporal_blend: 0.1,
            ambient_occlusion: 1.0,
            specular: 1.0,
            specular_roughness: 0.3..0.6,
            emissive_scale: 1.0,
            emissive_unexposed: false,
            specular_separate: false,
            ambient: 0.0,
            amortize_far: true,
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
