use crate::prepare::{
    CASCADE_FORMAT, CascadeUniform, PYRAMID_FORMAT, RadianceCascadesResources, VOXEL_FORMAT,
    VoxelUniform,
};
use bevy::{
    asset::{AssetServer, load_embedded_asset},
    core_pipeline::{FullscreenShader, prepass::ViewPrepassTextures},
    ecs::{
        resource::Resource,
        system::{Commands, Res},
    },
    math::UVec2,
    pbr::ScreenSpaceAmbientOcclusionResources,
    render::{
        diagnostic::RecordDiagnostics as _,
        render_resource::{
            AddressMode, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            BlendComponent, BlendFactor, BlendOperation, BlendState, CachedComputePipelineId,
            CachedRenderPipelineId, ColorTargetState, ColorWrites, CommandEncoder, ComputePass,
            ComputePassDescriptor, ComputePipelineDescriptor, FilterMode, FragmentState, LoadOp,
            MipmapFilterMode, Operations, PipelineCache, RenderPassColorAttachment,
            RenderPassDescriptor, RenderPipelineDescriptor, Sampler, SamplerBindingType,
            SamplerDescriptor, ShaderStages, StorageTextureAccess, StoreOp, TextureFormat,
            TextureSampleType, TextureUsages, TextureView,
            binding_types::{
                sampler, storage_buffer_sized, texture_2d, texture_3d, texture_depth_2d,
                texture_storage_2d, texture_storage_3d, uniform_buffer,
            },
        },
        renderer::{RenderContext, RenderDevice, ViewQuery},
        view::{ViewDepthStencilTexture, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
    },
    utils::default,
};

#[derive(Resource)]
pub struct RadianceCascadesPipelines {
    layout: BindGroupLayoutDescriptor,
    pyramid_layout: BindGroupLayoutDescriptor,
    voxel_layout: BindGroupLayoutDescriptor,
    voxel_mark: CachedComputePipelineId,
    voxel_update: CachedComputePipelineId,
    voxel_mask_fine: CachedComputePipelineId,
    voxel_mask_coarse: CachedComputePipelineId,
    color_layout: BindGroupLayoutDescriptor,
    color_first: CachedComputePipelineId,
    color_reduce: CachedComputePipelineId,
    composite_layout: BindGroupLayoutDescriptor,
    pyramid_first: CachedComputePipelineId,
    pyramid_reduce: CachedComputePipelineId,
    cascade: CachedComputePipelineId,
    gather: CachedComputePipelineId,
    resolve: CachedComputePipelineId,
    debug_view: CachedComputePipelineId,
    composite_add: CachedRenderPipelineId,
    composite_replace: CachedRenderPipelineId,
    sampler: Sampler,
}

pub fn init_radiance_cascades_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
) {
    // Must match the bindings in `common.wesl`.
    let layout = BindGroupLayoutDescriptor::new(
        "radiance_cascades_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<CascadeUniform>(false),
                uniform_buffer::<ViewUniform>(true),
                texture_depth_2d(),
                texture_2d(TextureSampleType::Uint),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_storage_2d(CASCADE_FORMAT, StorageTextureAccess::WriteOnly),
                texture_2d(TextureSampleType::Float { filterable: false }),
                uniform_buffer::<VoxelUniform>(false),
                texture_3d(TextureSampleType::Uint),
                texture_3d(TextureSampleType::Uint),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_storage_2d(CASCADE_FORMAT, StorageTextureAccess::WriteOnly),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_3d(TextureSampleType::Uint),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_storage_2d(CASCADE_FORMAT, StorageTextureAccess::WriteOnly),
                uniform_buffer::<crate::emitters::EmitterUniform>(false),
                texture_storage_2d(CASCADE_FORMAT, StorageTextureAccess::WriteOnly),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_storage_2d(CASCADE_FORMAT, StorageTextureAccess::WriteOnly),
            ),
        ),
    );

    // Must match the bindings in `color_pyramid.wesl`.
    let color_layout = BindGroupLayoutDescriptor::new(
        "radiance_cascades_color_pyramid_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_depth_2d(),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_storage_2d(CASCADE_FORMAT, StorageTextureAccess::WriteOnly),
            ),
        ),
    );

    // Must match the bindings in `voxel_update.wesl`.
    let voxel_layout = BindGroupLayoutDescriptor::new(
        "radiance_cascades_voxel_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<VoxelUniform>(false),
                uniform_buffer::<ViewUniform>(true),
                texture_depth_2d(),
                texture_2d(TextureSampleType::Uint),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_storage_3d(VOXEL_FORMAT, StorageTextureAccess::ReadWrite),
                texture_storage_3d(VOXEL_FORMAT, StorageTextureAccess::ReadWrite),
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_storage_3d(VOXEL_FORMAT, StorageTextureAccess::WriteOnly),
                storage_buffer_sized(false, None),
                texture_2d(TextureSampleType::Float { filterable: true }),
            ),
        ),
    );

    // Must match the bindings in `pyramid.wesl`.
    let pyramid_layout = BindGroupLayoutDescriptor::new(
        "radiance_cascades_pyramid_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_depth_2d(),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_storage_2d(PYRAMID_FORMAT, StorageTextureAccess::WriteOnly),
            ),
        ),
    );

    let composite_layout = BindGroupLayoutDescriptor::new(
        "radiance_cascades_composite_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Float { filterable: true }),
            ),
        ),
    );

    let compute_pipeline = |label: &'static str,
                            entry_point: &'static str,
                            layout: &BindGroupLayoutDescriptor,
                            shader| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: vec![layout.clone()],
            shader,
            entry_point: Some(entry_point.into()),
            ..default()
        })
    };
    let cascade = compute_pipeline(
        "radiance_cascades_cascade_pipeline",
        "cascade",
        &layout,
        load_embedded_asset!(asset_server.as_ref(), "cascade.wesl"),
    );
    let gather = compute_pipeline(
        "radiance_cascades_gather_pipeline",
        "gather",
        &layout,
        load_embedded_asset!(asset_server.as_ref(), "gather.wesl"),
    );
    let resolve = compute_pipeline(
        "radiance_cascades_resolve_pipeline",
        "resolve",
        &layout,
        load_embedded_asset!(asset_server.as_ref(), "gather.wesl"),
    );

    let debug_view = compute_pipeline(
        "radiance_cascades_debug_view_pipeline",
        "debug_view",
        &layout,
        load_embedded_asset!(asset_server.as_ref(), "gather.wesl"),
    );
    let pyramid_first = compute_pipeline(
        "radiance_cascades_pyramid_first_pipeline",
        "first",
        &pyramid_layout,
        load_embedded_asset!(asset_server.as_ref(), "pyramid.wesl"),
    );
    let pyramid_reduce = compute_pipeline(
        "radiance_cascades_pyramid_reduce_pipeline",
        "reduce",
        &pyramid_layout,
        load_embedded_asset!(asset_server.as_ref(), "pyramid.wesl"),
    );

    let color_first = compute_pipeline(
        "radiance_cascades_color_first_pipeline",
        "first",
        &color_layout,
        load_embedded_asset!(asset_server.as_ref(), "color_pyramid.wesl"),
    );
    let color_reduce = compute_pipeline(
        "radiance_cascades_color_reduce_pipeline",
        "reduce",
        &color_layout,
        load_embedded_asset!(asset_server.as_ref(), "color_pyramid.wesl"),
    );
    let voxel_update = compute_pipeline(
        "radiance_cascades_voxel_update_pipeline",
        "update",
        &voxel_layout,
        load_embedded_asset!(asset_server.as_ref(), "voxel_update.wesl"),
    );
    let voxel_mark = compute_pipeline(
        "radiance_cascades_voxel_mark_pipeline",
        "mark",
        &voxel_layout,
        load_embedded_asset!(asset_server.as_ref(), "voxel_update.wesl"),
    );
    let voxel_mask_fine = compute_pipeline(
        "radiance_cascades_voxel_mask_fine_pipeline",
        "mask_fine",
        &voxel_layout,
        load_embedded_asset!(asset_server.as_ref(), "voxel_update.wesl"),
    );
    let voxel_mask_coarse = compute_pipeline(
        "radiance_cascades_voxel_mask_coarse_pipeline",
        "mask_coarse",
        &voxel_layout,
        load_embedded_asset!(asset_server.as_ref(), "voxel_update.wesl"),
    );

    let composite_pipeline = |label: &'static str, blend: Option<BlendState>| {
        pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some(label.into()),
            layout: vec![composite_layout.clone()],
            vertex: fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: load_embedded_asset!(asset_server.as_ref(), "composite.wesl"),
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    // `RadianceCascades` requires `Hdr`.
                    format: TextureFormat::Rgba16Float,
                    blend,
                    write_mask: ColorWrites::COLOR,
                })],
                ..default()
            }),
            ..default()
        })
    };
    let additive = BlendComponent {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::One,
        operation: BlendOperation::Add,
    };
    let composite_add = composite_pipeline(
        "radiance_cascades_composite_pipeline",
        Some(BlendState {
            color: additive,
            alpha: additive,
        }),
    );
    let composite_replace = composite_pipeline("radiance_cascades_composite_debug_pipeline", None);

    let sampler = render_device.create_sampler(&SamplerDescriptor {
        label: Some("radiance_cascades_sampler"),
        min_filter: FilterMode::Linear,
        mag_filter: FilterMode::Linear,
        mipmap_filter: MipmapFilterMode::Linear,
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        ..default()
    });

    commands.insert_resource(RadianceCascadesPipelines {
        layout,
        pyramid_layout,
        voxel_layout,
        voxel_mark,
        voxel_update,
        voxel_mask_fine,
        voxel_mask_coarse,
        color_layout,
        color_first,
        color_reduce,
        composite_layout,
        pyramid_first,
        pyramid_reduce,
        cascade,
        gather,
        resolve,
        debug_view,
        composite_add,
        composite_replace,
        sampler,
    });
}

pub fn radiance_cascades(
    view: ViewQuery<(
        &RadianceCascadesResources,
        &ViewTarget,
        &ViewPrepassTextures,
        &ViewDepthStencilTexture,
        &ViewUniformOffset,
        Option<&ScreenSpaceAmbientOcclusionResources>,
        Option<&crate::prepare::RadianceCascadesSpecularCover>,
    )>,
    pipelines: Option<Res<RadianceCascadesPipelines>>,
    pipeline_cache: Res<PipelineCache>,
    view_uniforms: Res<ViewUniforms>,
    emitters: Res<crate::emitters::EmitterBuffer>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    let (
        resources,
        view_target,
        prepass_textures,
        view_depth,
        view_uniform_offset,
        occlusion,
        specular_cover,
    ) = view.into_inner();

    let Some(pipelines) = pipelines else {
        return;
    };
    let Some(emitters) = emitters.0.binding() else {
        return;
    };
    if view_target.main_texture_format() != TextureFormat::Rgba16Float {
        return;
    }
    // Depth can only be copied whole, and only if `DepthPrepass` (which
    // `RadianceCascades` requires) has made the depth buffer a copy source.
    let scene_depth = view_depth.texture();
    if scene_depth.size() != resources.depth.size()
        || scene_depth.sample_count() != 1
        || !scene_depth.usage().contains(TextureUsages::COPY_SRC)
    {
        return;
    }
    let depth = &resources.depth_view;
    let composite_pipeline_id = if resources.replace_scene {
        pipelines.composite_replace
    } else {
        pipelines.composite_add
    };
    let (
        Some(voxel_mark_pipeline),
        Some(voxel_update_pipeline),
        Some(voxel_mask_fine_pipeline),
        Some(voxel_mask_coarse_pipeline),
        Some(color_first_pipeline),
        Some(color_reduce_pipeline),
        Some(pyramid_first_pipeline),
        Some(pyramid_reduce_pipeline),
        Some(cascade_pipeline),
        Some(gather_pipeline),
        Some(resolve_pipeline),
        Some(debug_view_pipeline),
        Some(composite_pipeline),
        Some(gbuffer),
        Some(view_uniforms),
    ) = (
        pipeline_cache.get_compute_pipeline(pipelines.voxel_mark),
        pipeline_cache.get_compute_pipeline(pipelines.voxel_update),
        pipeline_cache.get_compute_pipeline(pipelines.voxel_mask_fine),
        pipeline_cache.get_compute_pipeline(pipelines.voxel_mask_coarse),
        pipeline_cache.get_compute_pipeline(pipelines.color_first),
        pipeline_cache.get_compute_pipeline(pipelines.color_reduce),
        pipeline_cache.get_compute_pipeline(pipelines.pyramid_first),
        pipeline_cache.get_compute_pipeline(pipelines.pyramid_reduce),
        pipeline_cache.get_compute_pipeline(pipelines.cascade),
        pipeline_cache.get_compute_pipeline(pipelines.gather),
        pipeline_cache.get_compute_pipeline(pipelines.resolve),
        pipeline_cache.get_compute_pipeline(pipelines.debug_view),
        pipeline_cache.get_render_pipeline(composite_pipeline_id),
        prepass_textures.deferred_view(),
        view_uniforms.uniforms.binding(),
    )
    else {
        return;
    };

    let layout = pipeline_cache.get_bind_group_layout(&pipelines.layout);
    let scene_color = view_target.main_texture_view();
    let history = resources.previous_indirect();
    // The motion vector prepass, where the camera has one. (Where it has not,
    // the uniform says so and anything will do for the binding.)
    let motion_vectors = prepass_textures.motion_vectors_view().unwrap_or(history);
    // The same goes for screen-space ambient occlusion.
    let occlusion = occlusion.map_or(history, |occlusion| {
        &occlusion
            .screen_space_ambient_occlusion_texture
            .default_view
    });
    let cascades = &resources.cascades;
    let voxels = &resources.voxels;

    // Cascade `n` reads the merged cascade `n + 1`. The top one reads nothing,
    // but still needs something bound.
    let cascade_bind_groups: Vec<_> = cascades
        .iter()
        .enumerate()
        .map(|(index, level)| {
            let upper = cascades.get(index + 1).map_or(history, |upper| &upper.view);
            render_device.create_bind_group(
                "radiance_cascades_cascade_bind_group",
                &layout,
                &BindGroupEntries::sequential((
                    level.uniform.as_entire_binding(),
                    view_uniforms.clone(),
                    depth,
                    gbuffer,
                    scene_color,
                    history,
                    upper,
                    &pipelines.sampler,
                    &level.view,
                    &resources.pyramid,
                    voxels.uniform.as_entire_binding(),
                    &voxels.radiance,
                    &voxels.meta,
                    &resources.color_pyramid,
                    // (Only the resolve uses these three.)
                    resources.previous_irradiance(),
                    resources.irradiance(),
                    motion_vectors,
                    &voxels.mask,
                    occlusion,
                    resources.previous_recent(),
                    resources.recent(),
                    emitters.clone(),
                    // What was mirrored last frame, to bounce again (the
                    // resolve writes this frame's, into a binding the
                    // cascades leave alone).
                    &resources.raw_irradiance,
                    history,
                    resources.previous_specular(),
                    &resources.highlight,
                )),
            )
        })
        .collect();

    // The gather and the debug views read cascade 0; the resolve reads what
    // the gather traced in its place, and accumulates last frame's incoming
    // light into this frame's. The debug views, drawn after it, read the
    // result (and write nothing to the other).
    let gather_bind_group = |source: &TextureView,
                             output: &TextureView,
                             read: &TextureView,
                             write: &TextureView,
                             swap: bool| {
        let (read_recent, write_recent) = if swap {
            (resources.recent(), resources.previous_recent())
        } else {
            (resources.previous_recent(), resources.recent())
        };
        render_device.create_bind_group(
            "radiance_cascades_gather_bind_group",
            &layout,
            &BindGroupEntries::sequential((
                cascades[0].uniform.as_entire_binding(),
                view_uniforms.clone(),
                depth,
                gbuffer,
                scene_color,
                history,
                source,
                &pipelines.sampler,
                output,
                &resources.pyramid,
                voxels.uniform.as_entire_binding(),
                &voxels.radiance,
                &voxels.meta,
                &resources.color_pyramid,
                read,
                write,
                motion_vectors,
                &voxels.mask,
                occlusion,
                read_recent,
                write_recent,
                emitters.clone(),
                resources.specular(),
                &cascades[resources.specular_level].view,
                resources.previous_specular(),
                &resources.highlight,
            )),
        )
    };
    let debug_bind_group = resources.debug_view.as_ref().map(|output| {
        gather_bind_group(
            &cascades[0].view,
            output,
            resources.irradiance(),
            resources.previous_irradiance(),
            true,
        )
    });
    let resolve_bind_group = gather_bind_group(
        &resources.raw_irradiance,
        resources.indirect(),
        resources.previous_irradiance(),
        resources.irradiance(),
        false,
    );
    let gather_bind_group = gather_bind_group(
        &cascades[0].view,
        &resources.raw_irradiance,
        resources.previous_irradiance(),
        resources.irradiance(),
        false,
    );

    // Level 0 of the color pyramid is reduced from the lit scene, level `n`
    // from level `n - 1`.
    let color_layout = pipeline_cache.get_bind_group_layout(&pipelines.color_layout);
    let color_bind_groups: Vec<_> = resources
        .color_levels
        .iter()
        .enumerate()
        .map(|(level, view)| {
            let source = match level {
                0 => scene_color,
                _ => &resources.color_levels[level - 1],
            };
            render_device.create_bind_group(
                "radiance_cascades_color_pyramid_bind_group",
                &color_layout,
                &BindGroupEntries::sequential((depth, source, view)),
            )
        })
        .collect();

    let voxel_bind_group = render_device.create_bind_group(
        "radiance_cascades_voxel_bind_group",
        &pipeline_cache.get_bind_group_layout(&pipelines.voxel_layout),
        &BindGroupEntries::sequential((
            voxels.uniform.as_entire_binding(),
            view_uniforms.clone(),
            depth,
            gbuffer,
            scene_color,
            history,
            &voxels.radiance,
            &voxels.meta,
            &resources.color_pyramid,
            &pipelines.sampler,
            &voxels.mask,
            voxels.marks.as_entire_binding(),
            motion_vectors,
        )),
    );

    // Level `n` of the pyramid is reduced from level `n - 1`; level 0 from the
    // depth buffer (and then the source binding is just a placeholder).
    let pyramid_layout = pipeline_cache.get_bind_group_layout(&pipelines.pyramid_layout);
    let pyramid_bind_groups: Vec<_> = resources
        .pyramid_levels
        .iter()
        .enumerate()
        .map(|(level, view)| {
            let source = match level {
                0 => history,
                _ => &resources.pyramid_levels[level - 1],
            };
            render_device.create_bind_group(
                "radiance_cascades_pyramid_bind_group",
                &pyramid_layout,
                &BindGroupEntries::sequential((depth, source, view)),
            )
        })
        .collect();

    let composite_bind_group = render_device.create_bind_group(
        "radiance_cascades_composite_bind_group",
        &pipeline_cache.get_bind_group_layout(&pipelines.composite_layout),
        &BindGroupEntries::sequential((
            resources
                .debug_view
                .as_ref()
                .unwrap_or(resources.indirect()),
            &resources.highlight,
            resources.specular(),
            // (Not in the debug views, nor without one: see `composite.wesl`.)
            specular_cover
                .filter(|_| resources.debug_view.is_none())
                .map_or(resources.specular(), |cover| &cover.0),
        )),
    );

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let command_encoder = ctx.command_encoder();
    let time_span = diagnostics.time_span(command_encoder, "radiance_cascades");

    command_encoder.copy_texture_to_texture(
        scene_depth.as_image_copy(),
        resources.depth.as_image_copy(),
        scene_depth.size(),
    );

    // A pass for each stage, so that each is timed on its own.
    let view_groups = resources.view_size.map(|size| size.div_ceil(8));
    fn compute_pass<'a>(encoder: &'a mut CommandEncoder, label: &'static str) -> ComputePass<'a> {
        encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        })
    }

    {
        let span = diagnostics.time_span(command_encoder, "rc_pyramids");
        let mut pass = compute_pass(command_encoder, "radiance_cascades_pyramids");
        for (level, bind_group) in pyramid_bind_groups.iter().enumerate() {
            let size = (resources.view_size >> level as u32).max(UVec2::ONE);
            pass.set_pipeline(if level == 0 {
                pyramid_first_pipeline
            } else {
                pyramid_reduce_pipeline
            });
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(size.x.div_ceil(8), size.y.div_ceil(8), 1);
        }

        for (level, bind_group) in color_bind_groups.iter().enumerate() {
            let size = (resources.color_size >> level as u32).max(UVec2::ONE);
            pass.set_pipeline(if level == 0 {
                color_first_pipeline
            } else {
                color_reduce_pipeline
            });
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(size.x.div_ceil(8), size.y.div_ceil(8), 1);
        }
        drop(pass);
        span.end(command_encoder);
    }

    // The world cache before the cascades, so their rays already find this
    // frame in it. (It reads the color pyramid.)
    if voxels.levels > 0 && !voxels.held {
        let span = diagnostics.time_span(command_encoder, "rc_voxels");
        command_encoder.clear_buffer(&voxels.marks, 0, None);
        let mut pass = compute_pass(command_encoder, "radiance_cascades_voxels");
        // Every pixel marks the voxel its surface lies in; then the voxels
        // are brought up to date with that.
        pass.set_pipeline(voxel_mark_pipeline);
        pass.set_bind_group(0, &voxel_bind_group, &[view_uniform_offset.offset]);
        pass.dispatch_workgroups(view_groups.x, view_groups.y, 1);
        let groups = voxels.resolution.div_ceil(4);
        pass.set_pipeline(voxel_update_pipeline);
        pass.dispatch_workgroups(groups, groups, groups);
        // And what that leaves of the level's blocks of voxels. (An anchor
        // has none: no ray crosses it.)
        if !voxels.updating_anchor {
            let fine_groups = (voxels.resolution / 4).div_ceil(4);
            pass.set_pipeline(voxel_mask_fine_pipeline);
            pass.dispatch_workgroups(fine_groups, fine_groups, fine_groups);
            let coarse_groups = (voxels.resolution / 16).div_ceil(4);
            pass.set_pipeline(voxel_mask_coarse_pipeline);
            pass.dispatch_workgroups(coarse_groups, coarse_groups, coarse_groups);
        }
        drop(pass);
        span.end(command_encoder);
    }

    const CASCADE_SPANS: [&str; 12] = [
        "rc_cascade_00",
        "rc_cascade_01",
        "rc_cascade_02",
        "rc_cascade_03",
        "rc_cascade_04",
        "rc_cascade_05",
        "rc_cascade_06",
        "rc_cascade_07",
        "rc_cascade_08",
        "rc_cascade_09",
        "rc_cascade_10",
        "rc_cascade_11",
    ];
    for (index, (level, bind_group)) in cascades.iter().zip(&cascade_bind_groups).enumerate().rev()
    {
        if resources.kept_cascades & (1 << index) != 0 {
            continue;
        }
        let span = diagnostics.time_span(
            command_encoder,
            CASCADE_SPANS[index.min(CASCADE_SPANS.len() - 1)],
        );
        let mut pass = compute_pass(command_encoder, "radiance_cascades_cascade");
        pass.set_pipeline(cascade_pipeline);
        pass.set_bind_group(0, bind_group, &[view_uniform_offset.offset]);
        pass.dispatch_workgroups(level.size.x.div_ceil(8), level.size.y.div_ceil(8), 1);
        drop(pass);
        span.end(command_encoder);
    }

    {
        let span = diagnostics.time_span(command_encoder, "rc_gather");
        let mut pass = compute_pass(command_encoder, "radiance_cascades_gather");
        pass.set_pipeline(gather_pipeline);
        pass.set_bind_group(0, &gather_bind_group, &[view_uniform_offset.offset]);
        pass.dispatch_workgroups(view_groups.x, view_groups.y, 1);
        drop(pass);
        span.end(command_encoder);
    }

    {
        let span = diagnostics.time_span(command_encoder, "rc_resolve");
        let mut pass = compute_pass(command_encoder, "radiance_cascades_resolve");
        pass.set_pipeline(resolve_pipeline);
        pass.set_bind_group(0, &resolve_bind_group, &[view_uniform_offset.offset]);
        pass.dispatch_workgroups(view_groups.x, view_groups.y, 1);

        if let Some(debug_bind_group) = &debug_bind_group {
            pass.set_pipeline(debug_view_pipeline);
            pass.set_bind_group(0, debug_bind_group, &[view_uniform_offset.offset]);
            pass.dispatch_workgroups(view_groups.x, view_groups.y, 1);
        }
        drop(pass);
        span.end(command_encoder);
    }

    {
        let mut pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("radiance_cascades_composite"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: view_target.main_texture_view(),
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_viewport(
            0.0,
            0.0,
            resources.view_size.x as f32,
            resources.view_size.y as f32,
            0.0,
            1.0,
        );
        pass.set_pipeline(composite_pipeline);
        pass.set_bind_group(0, &composite_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    time_span.end(command_encoder);
}
