use crate::{MAX_CASCADES, RadianceCascades, RadianceCascadesDebug};
use bevy::{
    camera::MainPassResolutionOverride,
    core_pipeline::prepass::MotionVectorPrepass,
    ecs::{
        component::Component,
        entity::Entity,
        query::Has,
        system::{Commands, Query, Res},
    },
    math::{IVec3, IVec4, Mat4, UVec2, Vec2, Vec3},
    pbr::ScreenSpaceAmbientOcclusion,
    render::{
        camera::ExtractedCamera,
        render_resource::{
            Buffer, BufferDescriptor, BufferUsages, Extent3d, ShaderType, Texture, TextureAspect,
            TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
            TextureViewDescriptor, encase,
        },
        renderer::{RenderDevice, RenderQueue},
        view::ExtractedView,
    },
};

pub const CASCADE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
pub const PYRAMID_FORMAT: TextureFormat = TextureFormat::Rg32Float;
/// Coarsest pyramid level is 2^(n-1) pixels wide.
const MAX_PYRAMID_LEVELS: u32 = 8;
/// The coarsest level of the color pyramid is 2^n pixels a texel.
const MAX_COLOR_LEVELS: u32 = 5;

/// Must match `RcParams` in `common.wesl`.
#[derive(Clone, Copy, ShaderType)]
pub struct CascadeUniform {
    prev_clip_from_world: Mat4,
    sky_radiance: Vec3,
    bounce: f32,
    probe_count: UVec2,
    upper_probe_count: UVec2,
    texture_size: UVec2,
    upper_texture_size: UVec2,
    probe_spacing: u32,
    dir_res: u32,
    cascade_index: u32,
    cascade_count: u32,
    interval_start: f32,
    interval_end: f32,
    thickness: f32,
    max_steps: u32,
    intensity: f32,
    debug: u32,
    normal_bias: f32,
    depth_bias: f32,
    pyramid_levels: u32,
    color_size: Vec2,
    color_levels: f32,
    temporal_blend: f32,
    has_motion_vectors: u32,
    ambient_occlusion: f32,
    specular: f32,
    specular_roughness: Vec2,
    emissive_scale: f32,
    emissive_unexposed: u32,
}

pub const VOXEL_FORMAT: TextureFormat = TextureFormat::R32Uint;
const MAX_VOXEL_LEVELS: usize = 8;

/// Must match `VoxelParams` in `voxel.wesl`.
#[derive(Clone, Copy, ShaderType)]
pub struct VoxelUniform {
    origin: [IVec4; MAX_VOXEL_LEVELS],
    update_prev_origin: IVec4,
    prev_clip_from_world: Mat4,
    resolution: u32,
    levels: u32,
    update_level: u32,
    max_steps: u32,
    voxel_size: f32,
    blend: f32,
    history_scale: f32,
    options: u32,
    has_motion_vectors: u32,
}

/// The world cache's textures and where each of its levels currently sits.
pub struct VoxelCache {
    pub radiance: TextureView,
    pub meta: TextureView,
    /// Which blocks of voxels have anything in them: see `voxel_mask_texel`
    /// in `voxel.wesl`.
    pub mask: TextureView,
    /// For each texel of the level being updated, a pixel whose surface lies
    /// in its voxel (see `voxel_update.wesl`). Cleared every frame.
    pub marks: Buffer,
    pub uniform: Buffer,
    pub resolution: u32,
    /// 0 when there is no world cache (and the textures are placeholders).
    pub levels: u32,
    voxel_size: f32,
    /// World index of each level's minimum corner, as of its last update.
    origin: [IVec3; MAX_VOXEL_LEVELS],
    frame: u32,
    /// The exposure the previous frame was rendered with.
    exposure: f32,
}

/// Everything that decides the size of a view's textures.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Layout {
    /// Size of the camera's depth texture (the whole render target).
    target_size: UVec2,
    view_size: UVec2,
    probe_spacing: u32,
    base_resolution: u32,
    cascade_count: u32,
    /// Resolution, levels and voxel size (as bits) of the world cache.
    voxels: Option<(u32, u32, u32)>,
}

impl Layout {
    fn new(
        settings: &RadianceCascades,
        target_size: UVec2,
        view_size: UVec2,
        max_texture_size: u32,
    ) -> Self {
        let probe_spacing = settings.probe_spacing.clamp(1, 64).next_power_of_two();
        let base_resolution = settings.base_resolution.clamp(2, 16).next_power_of_two();
        let mut layout = Self {
            target_size,
            view_size,
            probe_spacing,
            base_resolution,
            cascade_count: settings.cascade_count.clamp(1, MAX_CASCADES),
            voxels: settings.world_cache.map(|cache| {
                (
                    // (A whole number of the blocks of 16 voxels the occupancy
                    // masks are made of.)
                    cache.resolution.clamp(16, 256).next_multiple_of(16),
                    cache.levels.clamp(1, MAX_VOXEL_LEVELS as u32),
                    cache.voxel_size.max(1e-3).to_bits(),
                )
            }),
        };
        // Every cascade's texture is about `base_resolution / probe_spacing`
        // times the view: where that is more than a texture can be, space the
        // probes further apart.
        while layout.probe_spacing < 64 && layout.texture_size(0).max_element() > max_texture_size {
            layout.probe_spacing *= 2;
        }
        // Every cascade starts twice as far from its probes as the one
        // before. One that starts further off than the screen is across finds
        // nothing on it; more than one of those (the last cascade goes on
        // forever whichever it is) is rays traced for nothing, and four times
        // as many with each, as by then there is one probe left to trace them.
        let interval = settings.interval_length.max(0.25) * layout.probe_spacing as f32;
        let diagonal = view_size.as_vec2().length();
        while layout.cascade_count > 2
            && interval * ((1u32 << (layout.cascade_count - 2)) - 1) as f32 > diagonal
        {
            layout.cascade_count -= 1;
        }
        // Drop cascades whose direction maps no longer fit in a texture.
        while layout.cascade_count > 1
            && layout.texture_size(layout.cascade_count - 1).max_element() > max_texture_size
        {
            layout.cascade_count -= 1;
        }
        layout
    }

    fn probe_spacing(&self, cascade: u32) -> u32 {
        self.probe_spacing << cascade
    }

    fn dir_res(&self, cascade: u32) -> u32 {
        self.base_resolution << cascade
    }

    fn probe_count(&self, cascade: u32) -> UVec2 {
        let spacing = self.probe_spacing(cascade);
        UVec2::new(
            self.view_size.x.div_ceil(spacing),
            self.view_size.y.div_ceil(spacing),
        )
    }

    fn texture_size(&self, cascade: u32) -> UVec2 {
        self.probe_count(cascade) * self.dir_res(cascade)
    }

    /// Level 0 of the color pyramid: half the view.
    fn color_size(&self) -> UVec2 {
        (self.view_size / 2).max(UVec2::ONE)
    }

    fn color_levels(&self) -> u32 {
        (self.color_size().min_element().ilog2() + 1).min(MAX_COLOR_LEVELS)
    }

    fn pyramid_levels(&self) -> u32 {
        (self.view_size.min_element().ilog2() + 1).min(MAX_PYRAMID_LEVELS)
    }
}

pub struct CascadeLevel {
    pub view: TextureView,
    pub uniform: Buffer,
    pub size: UVec2,
}

/// Per-view GPU resources of [`RadianceCascades`].
#[derive(Component)]
pub struct RadianceCascadesResources {
    layout: Layout,
    pub(crate) cascades: Vec<CascadeLevel>,
    /// Indirect light leaving each pixel: this frame's and last frame's.
    indirect: [TextureView; 2],
    /// The incoming light as traced this frame, before it is steadied.
    pub(crate) raw_irradiance: TextureView,
    /// The accumulated incoming light: this frame's and last frame's.
    irradiance: [TextureView; 2],
    /// The same over only the last few frames (see `accumulate` in
    /// `gather.wesl`): this frame's and last frame's.
    recent: [TextureView; 2],
    /// Which of `indirect`, `irradiance` and `recent` is written this frame.
    current: usize,
    /// Frames since these were made.
    frame: u32,
    /// The cascades that keep last frame's result this frame, a bit each
    /// (see `RadianceCascades::amortize_far`).
    pub(crate) kept_cascades: u32,
    clip_from_world: Mat4,
    /// Copy of the depth buffer as the opaque pass left it. (The depth
    /// prepass's own copy lacks everything that is drawn without a prepass.)
    pub(crate) depth: Texture,
    pub(crate) depth_view: TextureView,
    /// The whole depth pyramid, and each of its levels on its own.
    pub(crate) pyramid: TextureView,
    pub(crate) pyramid_levels: Vec<TextureView>,
    /// The prefiltered lit scene, whole and level by level.
    pub(crate) color_pyramid: TextureView,
    pub(crate) color_levels: Vec<TextureView>,
    pub(crate) color_size: UVec2,
    pub(crate) voxels: VoxelCache,
    pub(crate) view_size: UVec2,
    /// What the debug views that are not lighting are drawn into.
    pub(crate) debug_view: Option<TextureView>,
    /// Show the indirect light (or `debug_view`) instead of adding it.
    pub(crate) replace_scene: bool,
    // Keep the textures alive alongside their views.
    _textures: Vec<Texture>,
}

impl RadianceCascadesResources {
    /// This frame's indirect light, as it is added to the view target.
    pub fn indirect(&self) -> &TextureView {
        &self.indirect[self.current]
    }

    pub(crate) fn previous_indirect(&self) -> &TextureView {
        &self.indirect[1 - self.current]
    }

    pub(crate) fn irradiance(&self) -> &TextureView {
        &self.irradiance[self.current]
    }

    pub(crate) fn previous_irradiance(&self) -> &TextureView {
        &self.irradiance[1 - self.current]
    }

    pub(crate) fn recent(&self) -> &TextureView {
        &self.recent[self.current]
    }

    pub(crate) fn previous_recent(&self) -> &TextureView {
        &self.recent[1 - self.current]
    }
}

#[allow(clippy::type_complexity)]
pub fn prepare_radiance_cascades_resources(
    mut views: Query<(
        Entity,
        &ExtractedCamera,
        &ExtractedView,
        &RadianceCascades,
        Option<&mut RadianceCascadesResources>,
        Option<&MainPassResolutionOverride>,
        Has<MotionVectorPrepass>,
        Has<ScreenSpaceAmbientOcclusion>,
    )>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut commands: Commands,
) {
    let max_texture_size = render_device.limits().max_texture_dimension_2d;

    for (
        entity,
        camera,
        view,
        settings,
        resources,
        resolution_override,
        motion_vectors,
        ambient_occlusion,
    ) in &mut views
    {
        let Some(mut view_size) = camera.physical_viewport_size else {
            continue;
        };
        if let Some(MainPassResolutionOverride(resolution_override)) = resolution_override {
            view_size = *resolution_override;
        }
        if view_size.x == 0 || view_size.y == 0 {
            continue;
        }

        let Some(target_size) = camera.physical_target_size else {
            continue;
        };
        let layout = Layout::new(settings, target_size, view_size, max_texture_size);
        let clip_from_world = view
            .clip_from_world
            .unwrap_or_else(|| view.clip_from_view * view.world_from_view.to_matrix().inverse());
        let replace_scene = settings.debug != RadianceCascadesDebug::Off;

        match resources {
            Some(mut resources) if resources.layout == layout => {
                let prev_clip_from_world = resources.clip_from_world;
                resources.clip_from_world = clip_from_world;
                resources.current = 1 - resources.current;
                resources.replace_scene = replace_scene;
                resources.frame = resources.frame.wrapping_add(1);
                resources.kept_cascades = kept_cascades(settings, resources.frame);
                if !settings.debug.is_lighting() && resources.debug_view.is_none() {
                    resources.debug_view = Some(create_view_texture(
                        &render_device,
                        "radiance_cascades_debug_view",
                        view_size,
                    ));
                } else if settings.debug.is_lighting() {
                    resources.debug_view = None;
                }
                write_voxel_uniform(
                    &render_queue,
                    &mut resources.voxels,
                    settings,
                    view,
                    camera.exposure,
                    prev_clip_from_world,
                    motion_vectors,
                );
                write_uniforms(
                    &render_queue,
                    &resources.cascades,
                    &layout,
                    settings,
                    prev_clip_from_world,
                    motion_vectors,
                    ambient_occlusion,
                );
            }
            _ => {
                let mut resources =
                    create_resources(&render_device, layout, clip_from_world, replace_scene);
                write_voxel_uniform(
                    &render_queue,
                    &mut resources.voxels,
                    settings,
                    view,
                    camera.exposure,
                    clip_from_world,
                    motion_vectors,
                );
                write_uniforms(
                    &render_queue,
                    &resources.cascades,
                    &layout,
                    settings,
                    clip_from_world,
                    motion_vectors,
                    ambient_occlusion,
                );
                commands.entity(entity).insert(resources);
            }
        }
    }
}

fn create_view_texture(
    render_device: &RenderDevice,
    label: &'static str,
    size: UVec2,
) -> TextureView {
    render_device
        .create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: CASCADE_FORMAT,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        })
        .create_view(&TextureViewDescriptor::default())
}

fn create_resources(
    render_device: &RenderDevice,
    layout: Layout,
    clip_from_world: Mat4,
    replace_scene: bool,
) -> RadianceCascadesResources {
    let mut textures = Vec::new();
    let mut create_texture = |label: &'static str, size: UVec2| {
        let texture = render_device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: CASCADE_FORMAT,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        textures.push(texture);
        view
    };

    let cascades = (0..layout.cascade_count)
        .map(|cascade| {
            let size = layout.texture_size(cascade);
            CascadeLevel {
                view: create_texture("radiance_cascade", size),
                uniform: render_device.create_buffer(&BufferDescriptor {
                    label: Some("radiance_cascade_uniform"),
                    size: CascadeUniform::min_size().get(),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                size,
            }
        })
        .collect();

    // wgpu zero-initializes textures, so there is no stale history to worry about.
    let indirect = [
        create_texture("radiance_cascades_indirect_a", layout.view_size),
        create_texture("radiance_cascades_indirect_b", layout.view_size),
    ];
    let raw_irradiance = create_texture("radiance_cascades_raw_irradiance", layout.view_size);
    let irradiance = [
        create_texture("radiance_cascades_irradiance_a", layout.view_size),
        create_texture("radiance_cascades_irradiance_b", layout.view_size),
    ];
    let recent = [
        create_texture("radiance_cascades_recent_a", layout.view_size),
        create_texture("radiance_cascades_recent_b", layout.view_size),
    ];

    let depth = render_device.create_texture(&TextureDescriptor {
        label: Some("radiance_cascades_depth"),
        size: Extent3d {
            width: layout.target_size.x,
            height: layout.target_size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Depth32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&TextureViewDescriptor {
        aspect: TextureAspect::DepthOnly,
        ..Default::default()
    });

    let pyramid_texture = render_device.create_texture(&TextureDescriptor {
        label: Some("radiance_cascades_depth_pyramid"),
        size: Extent3d {
            width: layout.view_size.x,
            height: layout.view_size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: layout.pyramid_levels(),
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: PYRAMID_FORMAT,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let pyramid = pyramid_texture.create_view(&TextureViewDescriptor::default());
    let pyramid_levels = (0..layout.pyramid_levels())
        .map(|level| {
            pyramid_texture.create_view(&TextureViewDescriptor {
                label: Some("radiance_cascades_depth_pyramid_level"),
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            })
        })
        .collect();
    textures.push(pyramid_texture);

    let color_size = layout.color_size();
    let color_texture = render_device.create_texture(&TextureDescriptor {
        label: Some("radiance_cascades_color_pyramid"),
        size: Extent3d {
            width: color_size.x,
            height: color_size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: layout.color_levels(),
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: CASCADE_FORMAT,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let color_pyramid = color_texture.create_view(&TextureViewDescriptor::default());
    let color_levels = (0..layout.color_levels())
        .map(|level| {
            color_texture.create_view(&TextureViewDescriptor {
                label: Some("radiance_cascades_color_pyramid_level"),
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            })
        })
        .collect();
    textures.push(color_texture);

    // Without a world cache the bindings still need something to point at.
    let (resolution, levels, voxel_size) = layout
        .voxels
        .map_or((4, 0, 1.0), |(resolution, levels, size)| {
            (resolution, levels, f32::from_bits(size))
        });
    let mut create_voxel_texture = |label: &'static str| {
        let texture = render_device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: resolution * levels.max(1),
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: VOXEL_FORMAT,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        textures.push(texture);
        view
    };
    // The occupancy masks: blocks of 4 voxels a side and, beside them along
    // x, blocks of 16.
    let fine_side = (resolution / 4).max(1);
    let mask_texture = render_device.create_texture(&TextureDescriptor {
        label: Some("radiance_cascades_voxel_mask"),
        size: Extent3d {
            width: fine_side + (resolution / 16).max(1),
            height: fine_side,
            depth_or_array_layers: fine_side * levels.max(1),
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D3,
        format: VOXEL_FORMAT,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let mask = mask_texture.create_view(&TextureViewDescriptor::default());
    let voxels = VoxelCache {
        radiance: create_voxel_texture("radiance_cascades_voxel_radiance"),
        meta: create_voxel_texture("radiance_cascades_voxel_meta"),
        mask,
        marks: render_device.create_buffer(&BufferDescriptor {
            label: Some("radiance_cascades_voxel_marks"),
            size: (resolution as u64).pow(3) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        uniform: render_device.create_buffer(&BufferDescriptor {
            label: Some("radiance_cascades_voxel_uniform"),
            size: VoxelUniform::min_size().get(),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        resolution,
        levels,
        voxel_size,
        // Nowhere near anything, so that the first update of each level
        // starts it from empty.
        origin: [IVec3::splat(i32::MIN / 2); MAX_VOXEL_LEVELS],
        frame: 0,
        exposure: 0.0,
    };

    RadianceCascadesResources {
        layout,
        voxels,
        cascades,
        indirect,
        raw_irradiance,
        irradiance,
        recent,
        depth,
        depth_view,
        pyramid,
        pyramid_levels,
        color_pyramid,
        color_levels,
        color_size,
        current: 0,
        frame: 0,
        kept_cascades: 0,
        clip_from_world,
        view_size: layout.view_size,
        debug_view: None,
        replace_scene,
        _textures: {
            textures.push(mask_texture);
            textures
        },
    }
}

/// Moves on to the next level of the world cache (one is updated per frame),
/// recentring it on the camera.
/// Which cascades are not traced on a frame (see
/// `RadianceCascades::amortize_far`): none for the first frames, while there
/// is nothing yet to keep.
fn kept_cascades(settings: &RadianceCascades, frame: u32) -> u32 {
    if !settings.amortize_far || frame < 8 {
        return 0;
    }
    (0..MAX_CASCADES).fold(0, |kept, cascade| {
        let traced = match cascade {
            0..=2 => true,
            3 | 4 => frame % 2 == cascade % 2,
            _ => frame % 4 == cascade % 4,
        };
        kept | (!traced as u32) << cascade
    })
}

fn write_voxel_uniform(
    render_queue: &RenderQueue,
    voxels: &mut VoxelCache,
    settings: &RadianceCascades,
    view: &ExtractedView,
    exposure: f32,
    prev_clip_from_world: Mat4,
    motion_vectors: bool,
) {
    let cache = settings.world_cache.unwrap_or_default();
    // Must match the `VOXEL_*` flags in `voxel.wesl`.
    let options = [
        cache.skip_moving,
        cache.soft_occupancy,
        cache.blend_lookup,
        cache.coarse_from_fine,
        cache.clamp_outliers,
        cache.soft_facing,
    ]
    .iter()
    .enumerate()
    .fold(0u32, |flags, (bit, on)| flags | (*on as u32) << bit);
    // The fine levels are what is near, where the view changes fastest: the
    // first is updated every other frame, the next every fourth, and so on.
    voxels.frame = voxels.frame.wrapping_add(1);
    let update_level = voxels
        .frame
        .max(1)
        .trailing_zeros()
        .min(voxels.levels.max(1) - 1);

    let size = voxels.voxel_size * (1u32 << update_level) as f32;
    let camera = view.world_from_view.translation();
    let prev_origin = voxels.origin[update_level as usize];
    voxels.origin[update_level as usize] =
        (camera / size).floor().as_ivec3() - IVec3::splat(voxels.resolution as i32 / 2);

    let uniform = VoxelUniform {
        origin: voxels.origin.map(|origin| origin.extend(0)),
        update_prev_origin: prev_origin.extend(0),
        prev_clip_from_world,
        resolution: voxels.resolution,
        levels: voxels.levels,
        update_level,
        max_steps: cache.max_steps.clamp(1, 1024),
        voxel_size: voxels.voxel_size,
        // A level that is brought up to date half as often takes on twice
        // as much each time, so that all of them settle at the pace of the
        // first. (Up to a point: the coarse ones are looked at so seldom that
        // they would take whatever a single look showed.)
        blend: 1.0 - (1.0 - cache.blend.clamp(0.01, 1.0)).powi(1 << update_level.min(2)),
        history_scale: if voxels.exposure > 0.0 {
            exposure / voxels.exposure
        } else {
            1.0
        },
        options,
        has_motion_vectors: motion_vectors as u32,
    };
    voxels.exposure = exposure;
    let mut buffer = encase::UniformBuffer::new(Vec::new());
    buffer.write(&uniform).unwrap();
    render_queue.write_buffer(&voxels.uniform, 0, buffer.as_ref());
}

fn write_uniforms(
    render_queue: &RenderQueue,
    cascades: &[CascadeLevel],
    layout: &Layout,
    settings: &RadianceCascades,
    prev_clip_from_world: Mat4,
    motion_vectors: bool,
    ambient_occlusion: bool,
) {
    let sky = settings.sky_color;
    let sky_radiance = Vec3::new(sky.red, sky.green, sky.blue) * settings.sky_brightness;
    let interval_length = settings.interval_length.max(0.25) * layout.probe_spacing as f32;
    let pyramid_levels = layout.pyramid_levels();

    for (cascade, level) in cascades.iter().enumerate() {
        let cascade = cascade as u32;
        let last = cascade + 1 == layout.cascade_count;
        let upper = if last { cascade } else { cascade + 1 };

        let uniform = CascadeUniform {
            prev_clip_from_world,
            sky_radiance,
            bounce: settings.bounce.max(0.0),
            probe_count: layout.probe_count(cascade),
            upper_probe_count: layout.probe_count(upper),
            texture_size: layout.texture_size(cascade),
            upper_texture_size: layout.texture_size(upper),
            probe_spacing: layout.probe_spacing(cascade),
            dir_res: layout.dir_res(cascade),
            cascade_index: cascade,
            cascade_count: layout.cascade_count,
            interval_start: interval_length * ((1u32 << cascade) - 1) as f32,
            interval_end: if last {
                // Far enough to leave any screen.
                1.0e6
            } else {
                interval_length * ((2u32 << cascade) - 1) as f32
            },
            thickness: settings.thickness.max(0.0),
            max_steps: settings.max_steps.clamp(1, 1024),
            intensity: settings.intensity.max(0.0),
            debug: settings.debug as u32,
            normal_bias: settings.normal_bias,
            depth_bias: settings.depth_bias,
            pyramid_levels,
            color_size: layout.color_size().as_vec2(),
            color_levels: layout.color_levels() as f32,
            temporal_blend: settings.temporal_blend.clamp(0.01, 1.0),
            has_motion_vectors: motion_vectors as u32,
            ambient_occlusion: if ambient_occlusion {
                settings.ambient_occlusion.clamp(0.0, 1.0)
            } else {
                0.0
            },
            specular: settings.specular.clamp(0.0, 1.0),
            specular_roughness: Vec2::new(
                settings.specular_roughness.start,
                settings.specular_roughness.end,
            ),
            emissive_scale: settings.emissive_scale.max(0.0),
            emissive_unexposed: settings.emissive_unexposed as u32,
        };

        let mut buffer = encase::UniformBuffer::new(Vec::new());
        buffer.write(&uniform).unwrap();
        render_queue.write_buffer(&level.uniform, 0, buffer.as_ref());
    }
}
