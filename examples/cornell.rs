//! A Cornell box lit by nothing but the glowing panel in its ceiling, to
//! compare the cascades with a path-traced reference of the same scene.
//!
//! No light source in it is one the engine knows of: without the cascades the
//! box is black but for the panel.
//!
//! Controls: `Space` toggles the cascades, `D` cycles the debug views.
//!
//! `RC_MODE=reference` path traces the scene instead (Bevy Solari's reference
//! path tracer; needs ray tracing hardware), accumulating for as long as it
//! runs. `RC_MODE=off|on|indirect|irradiance|surfaces|world` as in `room`.
//!
//! For automated captures set `RC_SHOT=<file.png>` (renders a hidden window,
//! saves the frame `RC_FRAME` and exits). `RC_VIEW=inside` looks at the boxes
//! from within, with the panel out of view; `RC_VIEW=floor` looks down at the
//! foot of the tall box. `RC_EMITTER=1` hands the panel to the cascades as a
//! `RadianceEmitter` instead (and draws it dark). `RC_GLOSS=1` makes the tall box a rough mirror.

use bevy::{
    camera::{CameraMainTextureUsages, Hdr},
    core_pipeline::prepass::{DeferredPrepass, DepthPrepass},
    mesh::Indices,
    prelude::*,
    render::{
        render_resource::TextureUsages,
        view::screenshot::{Screenshot, save_to_disk},
    },
    solari::{
        pathtracer::{Pathtracer, PathtracingPlugin},
        prelude::RaytracingMesh3d,
        scene::RaytracingScenePlugin,
    },
    window::{PresentMode, WindowResolution},
};
use bevy_radiance_cascades::{
    RadianceCascades, RadianceCascadesDebug, RadianceCascadesPlugin, RadianceEmitter,
};

/// The panel's luminance, in cd/m^2.
const PANEL: Vec3 = Vec3::new(34_000.0, 28_000.0, 19_000.0);
const PANEL_SIDE: f32 = 1.2;

#[derive(Resource)]
struct Shot {
    path: String,
    frame: u32,
}

fn main() {
    let shot = std::env::var("RC_SHOT").ok();
    let reference = mode() == "reference";

    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "cornell box".into(),
                resolution: WindowResolution::new(
                    env_or("RC_WIDTH", 960.0) as u32,
                    env_or("RC_HEIGHT", 960.0) as u32,
                )
                .with_scale_factor_override(1.0),
                // Captures must not pop up over whatever else is going on,
                // nor wait for a display nobody is looking at.
                visible: shot.is_none(),
                focused: shot.is_none(),
                present_mode: if shot.is_some() {
                    PresentMode::AutoNoVsync
                } else {
                    PresentMode::AutoVsync
                },
                ..default()
            }),
            ..default()
        }),
        RadianceCascadesPlugin,
    ))
    .insert_resource(GlobalAmbientLight {
        brightness: 0.0,
        ..default()
    })
    .add_systems(Startup, setup)
    .add_systems(Update, controls);

    if reference {
        app.add_plugins((RaytracingScenePlugin, PathtracingPlugin));
    }
    if let Some(path) = shot {
        app.insert_resource(Shot { path, frame: 0 })
            .add_systems(Update, capture);
    }
    app.run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let reference = mode() == "reference";
    // (`RC_PANEL` scales the panel's brightness: with `RC_LINEAR=1`, which
    // leaves the image untonemapped, a dim one keeps everything in range to be
    // measured.)
    let panel = PANEL * env_or("RC_PANEL", 1.0);
    let mut matte = |color: Color| {
        materials.add(StandardMaterial {
            base_color: color,
            perceptual_roughness: 1.0,
            reflectance: 0.0,
            ..default()
        })
    };
    let white = matte(Color::linear_rgb(0.73, 0.73, 0.73));
    let red = matte(Color::linear_rgb(0.65, 0.05, 0.05));
    let green = matte(Color::linear_rgb(0.12, 0.45, 0.15));
    let tall = if std::env::var_os("RC_GLOSS").is_some() {
        materials.add(StandardMaterial {
            base_color: Color::linear_rgb(0.9, 0.9, 0.9),
            perceptual_roughness: 0.25,
            metallic: 1.0,
            ..default()
        })
    } else {
        white.clone()
    };
    // The path tracer takes a glow to be in cd/m^2 and exposes it as the
    // camera does everything else. Bevy's deferred renderer draws it as given,
    // whatever the exposure (`emissive_exposure_weight` does not survive the
    // G-buffer), so there it is exposed here.
    let glow = if reference {
        panel
    } else if std::env::var_os("RC_EMITTER").is_some() {
        // An emitter's light is its own, on top of whatever its mesh glows
        // with: to compare it with the reference, the mesh is left dark.
        Vec3::ZERO
    } else {
        panel * bevy::camera::Exposure::default().exposure()
    };
    let glowing = materials.add(StandardMaterial {
        base_color: Color::BLACK,
        emissive: LinearRgba::rgb(glow.x, glow.y, glow.z),
        reflectance: 0.0,
        ..default()
    });

    // (What the path tracer wants of a mesh: tangents and 32-bit indices.)
    let mut traceable = |mesh: Mesh| {
        let mut mesh = mesh.with_generated_tangents().unwrap();
        if let Some(Indices::U16(indices)) = mesh.indices() {
            let wide = indices.iter().map(|&i| i as u32).collect();
            mesh.insert_indices(Indices::U32(wide));
        }
        meshes.add(mesh)
    };
    let mut spawn =
        |mesh: Handle<Mesh>, material: &Handle<StandardMaterial>, transform: Transform| -> Entity {
            let mut entity = commands.spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(material.clone()),
                transform,
            ));
            if reference {
                entity.insert(RaytracingMesh3d(mesh));
            }
            entity.id()
        };

    // The box: 4 m a side, its floor at 0, open towards +z.
    let mut slab = |size: Vec3, at: Vec3, turn: f32, material: &Handle<StandardMaterial>| {
        let mesh = traceable(Cuboid::from_size(size).into());
        spawn(
            mesh,
            material,
            Transform::from_translation(at).with_rotation(Quat::from_rotation_y(turn)),
        );
    };
    slab(
        Vec3::new(4.4, 0.2, 4.4),
        Vec3::new(0.0, -0.1, 0.0),
        0.0,
        &white,
    );
    slab(
        Vec3::new(4.4, 0.2, 4.4),
        Vec3::new(0.0, 4.1, 0.0),
        0.0,
        &white,
    );
    slab(
        Vec3::new(4.4, 4.4, 0.2),
        Vec3::new(0.0, 2.0, -2.1),
        0.0,
        &white,
    );
    slab(
        Vec3::new(0.2, 4.4, 4.4),
        Vec3::new(-2.1, 2.0, 0.0),
        0.0,
        &red,
    );
    slab(
        Vec3::new(0.2, 4.4, 4.4),
        Vec3::new(2.1, 2.0, 0.0),
        0.0,
        &green,
    );
    // The two boxes.
    slab(
        Vec3::new(1.2, 2.4, 1.2),
        Vec3::new(-0.65, 1.2, -0.6),
        0.3,
        &tall,
    );
    slab(
        Vec3::new(1.2, 1.2, 1.2),
        Vec3::new(0.7, 0.6, 0.6),
        -0.3,
        &white,
    );

    // The panel, just under the ceiling and facing down.
    let face = traceable(Rectangle::new(PANEL_SIDE, PANEL_SIDE).into());
    let face = spawn(
        face,
        &glowing,
        Transform::from_xyz(0.0, 3.99, 0.0)
            .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
    );
    if std::env::var_os("RC_EMITTER").is_some() {
        commands.entity(face).insert(RadianceEmitter {
            radiance: LinearRgba::rgb(panel.x, panel.y, panel.z),
            area: PANEL_SIDE * PANEL_SIDE,
            // (Of the mesh, which is turned to face down.)
            normal: Some(Dir3::Z),
            range: 12.0,
            clearance: 0.1,
            bounds: None,
        });
    }

    // (The cascades look in from outside first, as the default view does, and
    // only then go to another: what is off screen there they must remember.)
    let view = match std::env::var("RC_VIEW")
        .as_deref()
        .ok()
        .filter(|_| reference)
    {
        Some("up") => look_up(),
        Some("inside") => {
            Transform::from_xyz(1.5, 1.5, 1.8).looking_at(Vec3::new(-0.6, 0.9, -0.8), Vec3::Y)
        }
        Some("floor") => {
            Transform::from_xyz(0.9, 2.6, 1.6).looking_at(Vec3::new(-0.4, 0.0, -0.2), Vec3::Y)
        }
        // (`RC_EYE`: from how high. From 2 m the top of the tall box is never
        // seen, and what it sends up to the ceiling is not known.)
        _ => Transform::from_xyz(0.0, env_or("RC_EYE", 2.0), 7.3)
            .looking_at(Vec3::new(0.0, 2.0, 0.0), Vec3::Y),
    };
    let mut camera = commands.spawn((
        Camera3d::default(),
        (Hdr, DeferredPrepass, DepthPrepass),
        Msaa::Off,
        view,
    ));
    if std::env::var_os("RC_LINEAR").is_some() {
        camera.insert(bevy::core_pipeline::tonemapping::Tonemapping::None);
    }
    if reference {
        camera.insert((
            Pathtracer::default(),
            CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        ));
    } else if mode() != "off" {
        camera.insert(cascades());
    }
    if std::env::var_os("RC_SSAO").is_some() && !reference {
        camera.insert((
            bevy::pbr::ScreenSpaceAmbientOcclusion::default(),
            bevy::core_pipeline::prepass::NormalPrepass,
        ));
    }
}

fn cascades() -> RadianceCascades {
    RadianceCascades {
        // Nothing but black beyond the box.
        sky_brightness: 0.0,
        world_cache: (std::env::var("RC_WORLD").as_deref() != Ok("0")).then(default),
        cascade_count: env_or("RC_CASCADES", 4.0) as u32,
        probe_spacing: env_or("RC_SPACING", 4.0) as u32,
        base_resolution: env_or("RC_RES", 4.0) as u32,
        interval_length: env_or("RC_INTERVAL", 2.0),
        bounce: env_or("RC_BOUNCE", 1.0),
        thickness: env_or("RC_THICKNESS", 0.5),
        normal_bias: env_or("RC_NORMAL_BIAS", 1.0),
        depth_bias: env_or("RC_DEPTH_BIAS", 0.002),
        max_steps: env_or("RC_STEPS", 64.0) as u32,
        debug: match mode().as_str() {
            "indirect" => RadianceCascadesDebug::Indirect,
            "irradiance" => RadianceCascadesDebug::Irradiance,
            "surfaces" => RadianceCascadesDebug::Surfaces,
            "world" => RadianceCascadesDebug::WorldCache,
            _ => RadianceCascadesDebug::Off,
        },
        ..default()
    }
}

fn controls(
    keys: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    mut cameras: Query<(Entity, Option<&mut RadianceCascades>), With<Camera3d>>,
) {
    if mode() == "reference" {
        return;
    }
    for (entity, radiance_cascades) in &mut cameras {
        if keys.just_pressed(KeyCode::Space) {
            if radiance_cascades.is_some() {
                commands.entity(entity).remove::<RadianceCascades>();
            } else {
                commands.entity(entity).insert(cascades());
            }
        } else if let Some(mut radiance_cascades) = radiance_cascades
            && keys.just_pressed(KeyCode::KeyD)
        {
            radiance_cascades.debug = match radiance_cascades.debug {
                RadianceCascadesDebug::Off => RadianceCascadesDebug::Indirect,
                RadianceCascadesDebug::Indirect => RadianceCascadesDebug::Irradiance,
                RadianceCascadesDebug::Irradiance => RadianceCascadesDebug::Surfaces,
                RadianceCascadesDebug::Surfaces => RadianceCascadesDebug::WorldCache,
                _ => RadianceCascadesDebug::Off,
            };
        }
    }
}

/// From the floor, up at the ceiling: neither the floor nor the boxes, which
/// are what lights it, are in view.
fn look_up() -> Transform {
    Transform::from_xyz(0.0, 0.5, 1.7).looking_at(Vec3::new(0.0, 4.0, -0.9), Vec3::Y)
}

fn mode() -> String {
    std::env::var("RC_MODE").unwrap_or_default()
}

fn env_or(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn capture(
    mut commands: Commands,
    mut shot: ResMut<Shot>,
    mut exit: MessageWriter<AppExit>,
    mut cameras: Query<&mut Transform, With<Camera3d>>,
) {
    shot.frame += 1;
    // (Glided to over a hundred frames, as a player would get there, unless
    // `RC_JUMP` is set: what the cascades remember has to survive either.)
    let glide = if std::env::var_os("RC_JUMP").is_some() {
        1
    } else {
        100
    };
    if (120..120 + glide).contains(&shot.frame) && mode() != "reference" {
        let view = match std::env::var("RC_VIEW").as_deref() {
            Ok("up") => Some(look_up()),
            Ok("inside") => Some(
                Transform::from_xyz(1.5, 1.5, 1.8).looking_at(Vec3::new(-0.6, 0.9, -0.8), Vec3::Y),
            ),
            Ok("floor") => Some(
                Transform::from_xyz(0.9, 2.6, 1.6).looking_at(Vec3::new(-0.4, 0.0, -0.2), Vec3::Y),
            ),
            _ => None,
        };
        if let Some(view) = view {
            let left = (120 + glide - shot.frame) as f32;
            for mut transform in &mut cameras {
                transform.translation = transform.translation.lerp(view.translation, 1.0 / left);
                transform.rotation = transform.rotation.slerp(view.rotation, 1.0 / left);
            }
        }
    }
    // Time for the pipelines to compile and the bounces to converge (or, path
    // tracing, for the noise to: a sample a pixel a frame).
    let at = env_or(
        "RC_FRAME",
        if mode() == "reference" { 4000.0 } else { 420.0 },
    ) as u32;
    if shot.frame == at {
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(shot.path.clone()));
    }
    if shot.frame == at + 30 {
        exit.write(AppExit::Success);
    }
}
