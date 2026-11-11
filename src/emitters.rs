//! Lights the cascades are told of, rather than left to find on screen.
//!
//! The cascades light the scene with what the camera sees and with what the
//! world cache remembers of it: a lamp that has never been in view lights
//! nothing until it is, and one that is in view lights only as bright as it is
//! drawn. A [`RadianceEmitter`] is a lamp's lit face given outright, for the
//! rays to find wherever it is.

use crate::RadianceCascades;
use bevy::{
    color::LinearRgba,
    ecs::{
        component::Component,
        query::With,
        resource::Resource,
        system::{Commands, Query, Res, ResMut},
    },
    math::{Dir3, Vec3},
    prelude::{GlobalTransform, InheritedVisibility, ReflectComponent, ReflectDefault},
    reflect::Reflect,
    render::{
        Extract,
        render_resource::{ShaderType, UniformBuffer},
        renderer::{RenderDevice, RenderQueue},
        view::ExtractedView,
    },
};

/// The most emitters a view's cascades take: the nearest to the camera.
pub const MAX_EMITTERS: usize = 64;

/// A flat glowing surface that lights the scene through the cascades wherever
/// it is, in view or not: a lamp's face, a light panel, a screen.
///
/// It is not drawn: put it on (or beside) the mesh that is, whose own glow
/// can then be whatever looks right on screen.
#[derive(Component, Reflect, Clone, Copy, Debug)]
#[reflect(Component, Default, Clone)]
pub struct RadianceEmitter {
    /// Its luminance in cd/m^2, in linear colour: what an emissive material
    /// that lit the scene as it should would have.
    pub radiance: LinearRgba,
    /// Its area in world units squared.
    pub area: f32,
    /// The side it shines from, in the entity's own space.
    pub normal: Dir3,
    /// How far its light reaches, in world units, fading out over the last
    /// quarter. What stands between it and a surface is only known where the
    /// camera has seen it: the shorter this is, the less can leak.
    pub range: f32,
    /// The box its light stays in, as two corners relative to where it is,
    /// along the world's axes: the room it hangs in. The cascades know what
    /// stands between a light and a surface only where the camera has seen
    /// it, and a lamp otherwise lights the next room through a wall nobody
    /// has looked at yet. `None` lets it light whatever is in range.
    pub bounds: Option<(Vec3, Vec3)>,
}

impl Default for RadianceEmitter {
    fn default() -> Self {
        Self {
            radiance: LinearRgba::WHITE,
            area: 1.0,
            normal: Dir3::NEG_Y,
            range: 30.0,
            bounds: None,
        }
    }
}

/// An emitter as the render world has it.
#[derive(Clone, Copy)]
pub struct ExtractedEmitter {
    position: Vec3,
    normal: Vec3,
    radiance: Vec3,
    area: f32,
    range: f32,
    bounds: (Vec3, Vec3),
}

#[derive(Resource, Default)]
pub struct ExtractedEmitters(Vec<ExtractedEmitter>);

/// Must match `Emitter` in `common.wesl`.
#[derive(Clone, Copy, Default, ShaderType)]
struct EmitterData {
    position: Vec3,
    range: f32,
    normal: Vec3,
    area: f32,
    radiance: Vec3,
    bounds_min: Vec3,
    bounds_max: Vec3,
}

/// Must match `Emitters` in `common.wesl`.
#[derive(Clone, ShaderType)]
pub struct EmitterUniform {
    count: u32,
    items: [EmitterData; MAX_EMITTERS],
}

impl Default for EmitterUniform {
    fn default() -> Self {
        Self {
            count: 0,
            items: [EmitterData::default(); MAX_EMITTERS],
        }
    }
}

#[derive(Resource, Default)]
pub struct EmitterBuffer(pub UniformBuffer<EmitterUniform>);

pub fn extract_emitters(
    mut commands: Commands,
    emitters: Extract<Query<(&RadianceEmitter, &GlobalTransform, &InheritedVisibility)>>,
) {
    let list = emitters
        .iter()
        .filter(|(emitter, _, visible)| visible.get() && emitter.area > 0.0 && emitter.range > 0.0)
        .map(|(emitter, transform, _)| ExtractedEmitter {
            position: transform.translation(),
            normal: (transform.rotation() * *emitter.normal).normalize_or_zero(),
            radiance: Vec3::new(
                emitter.radiance.red,
                emitter.radiance.green,
                emitter.radiance.blue,
            )
            .max(Vec3::ZERO),
            area: emitter.area,
            range: emitter.range,
            bounds: emitter.bounds.map_or(
                (Vec3::splat(-1.0e30), Vec3::splat(1.0e30)),
                |(low, high)| {
                    let at = transform.translation();
                    (at + low.min(high), at + low.max(high))
                },
            ),
        })
        .collect();
    commands.insert_resource(ExtractedEmitters(list));
}

pub fn prepare_emitters(
    extracted: Option<Res<ExtractedEmitters>>,
    views: Query<&ExtractedView, With<RadianceCascades>>,
    mut buffer: ResMut<EmitterBuffer>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let mut uniform = EmitterUniform::default();
    if let (Some(extracted), Some(view)) = (extracted, views.iter().next()) {
        let camera = view.world_from_view.translation();
        // Those whose light reaches nearest the camera.
        let mut list: Vec<_> = extracted
            .0
            .iter()
            .filter(|e| e.radiance.max_element() > 0.0)
            .map(|e| (e.position.distance(camera) - e.range, e))
            .collect();
        list.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (slot, (_, e)) in uniform.items.iter_mut().zip(&list) {
            *slot = EmitterData {
                position: e.position,
                range: e.range,
                normal: e.normal,
                area: e.area,
                radiance: e.radiance,
                bounds_min: e.bounds.0,
                bounds_max: e.bounds.1,
            };
        }
        uniform.count = list.len().min(MAX_EMITTERS) as u32;
    }
    buffer.0.set(uniform);
    buffer.0.write_buffer(&render_device, &render_queue);
}
