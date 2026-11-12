//! A closed room lit by one spot light and one emissive block. Everything
//! outside the light's cone is lit by radiance cascades alone.
//!
//! Controls: `Space` toggles GI, `D` cycles the debug views, `M` toggles the
//! camera sway.
//!
//! For automated captures set `RC_SHOT=<file.png>` (renders a hidden window,
//! saves one frame and exits) and optionally `RC_MODE=off|on|indirect|irradiance|surfaces`,
//! `RC_WORLD=0` (no world cache) and `RC_VIEW=away` (turn away from the lights first).

use bevy::{
    camera::Hdr,
    core_pipeline::prepass::{DeferredPrepass, DepthPrepass},
    prelude::*,
    render::view::screenshot::{Screenshot, save_to_disk},
    window::WindowResolution,
};
use bevy_radiance_cascades::{RadianceCascades, RadianceCascadesDebug, RadianceCascadesPlugin};

#[derive(Resource)]
struct Shot {
    path: String,
    frame: u32,
}

#[derive(Resource)]
struct Sway(bool);

fn main() {
    let shot = std::env::var("RC_SHOT").ok();

    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "radiance cascades".into(),
                // (`RC_WIDTH`, `RC_HEIGHT`: a small view, for a software renderer.)
                resolution: WindowResolution::new(
                    env_or("RC_WIDTH", 1280.0) as u32,
                    env_or("RC_HEIGHT", 720.0) as u32,
                )
                .with_scale_factor_override(1.0),
                // Captures must not pop up over whatever else is going on.
                visible: shot.is_none(),
                focused: shot.is_none(),
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
    .insert_resource(Sway(
        shot.is_none() || std::env::var_os("RC_SWAY").is_some(),
    ))
    .add_systems(Startup, setup)
    .add_systems(Update, (controls, sway));

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
    let mut matte = |color: Color| {
        materials.add(StandardMaterial {
            base_color: color,
            perceptual_roughness: 0.9,
            ..default()
        })
    };
    let white = matte(Color::srgb(0.8, 0.8, 0.8));
    let red = matte(Color::srgb(0.8, 0.1, 0.08));
    let green = matte(Color::srgb(0.12, 0.7, 0.1));
    let blue = matte(Color::srgb(0.15, 0.25, 0.8));

    let mut slab = |size: Vec3, at: Vec3, material: &Handle<StandardMaterial>| {
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::from_size(size))),
            MeshMaterial3d(material.clone()),
            Transform::from_translation(at),
        ));
    };

    // A 6 x 4 x 6 room, open towards the camera.
    slab(Vec3::new(6.4, 0.2, 6.4), Vec3::new(0.0, -0.1, 0.0), &white);
    slab(Vec3::new(6.4, 0.2, 6.4), Vec3::new(0.0, 4.1, 0.0), &white);
    slab(Vec3::new(6.4, 4.4, 0.2), Vec3::new(0.0, 2.0, -3.1), &white);
    slab(Vec3::new(0.2, 4.4, 6.4), Vec3::new(-3.1, 2.0, 0.0), &red);
    slab(Vec3::new(0.2, 4.4, 6.4), Vec3::new(3.1, 2.0, 0.0), &green);

    // Things to cast and receive indirect shadows.
    slab(Vec3::new(1.2, 2.4, 1.2), Vec3::new(-1.3, 1.2, -1.2), &white);
    slab(Vec3::new(1.2, 1.2, 1.2), Vec3::new(1.2, 0.6, 0.4), &blue);
    slab(Vec3::new(2.4, 0.1, 1.0), Vec3::new(1.6, 2.2, -2.4), &white);

    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.45).mesh().uv(48, 24))),
        MeshMaterial3d(white.clone()),
        Transform::from_xyz(-0.4, 0.45, 1.3),
    ));

    // An emissive bar tucked under the shelf: light with no light source.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.8, 0.08, 0.2))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::BLACK,
            emissive: LinearRgba::new(30.0, 14.0, 3.0, 1.0),
            ..default()
        })),
        // (`RC_BAR=wall` mounts it flat on the back wall instead, `RC_BAR=none` hides it.)
        match std::env::var("RC_BAR").as_deref() {
            Ok("wall") => Transform::from_xyz(-0.2, 3.0, -2.96),
            Ok("none") => Transform::from_xyz(0.0, -50.0, 0.0),
            _ => Transform::from_xyz(1.6, 2.1, -2.5),
        },
    ));

    commands.spawn((
        SpotLight {
            intensity: 3_000_000.0,
            range: 20.0,
            inner_angle: 0.35,
            outer_angle: 0.5,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(-0.5, 3.9, 0.6).looking_at(Vec3::new(0.2, 0.0, 0.2), Vec3::Z),
    ));

    let mode = std::env::var("RC_MODE").unwrap_or_default();
    let mut camera = commands.spawn((
        Camera3d::default(),
        // What `RadianceCascades` requires, kept on so toggling it compares like with like.
        (Hdr, DeferredPrepass, DepthPrepass),
        Msaa::Off,
        Transform::from_xyz(0.0, 2.1, 8.2).looking_at(Vec3::new(0.0, 1.8, 0.0), Vec3::Y),
    ));
    // `RC_SSAO=1`: screen-space ambient occlusion, which the cascades then use.
    if std::env::var_os("RC_SSAO").is_some() {
        camera.insert((
            bevy::pbr::ScreenSpaceAmbientOcclusion::default(),
            bevy::core_pipeline::prepass::NormalPrepass,
        ));
    }
    if mode != "off" {
        camera.insert(RadianceCascades {
            // The room is closed: nothing but black beyond the screen.
            sky_brightness: 0.0,
            world_cache: (std::env::var("RC_WORLD").as_deref() != Ok("0")).then(default),
            // Overrides for experiments.
            cascade_count: env_or("RC_CASCADES", 4.0) as u32,
            probe_spacing: env_or("RC_SPACING", 4.0) as u32,
            base_resolution: env_or("RC_RES", 4.0) as u32,
            interval_length: env_or("RC_INTERVAL", 2.0),
            bounce: env_or("RC_BOUNCE", 1.0),
            thickness: env_or("RC_THICKNESS", 0.5),
            debug: match mode.as_str() {
                "indirect" => RadianceCascadesDebug::Indirect,
                "irradiance" => RadianceCascadesDebug::Irradiance,
                "surfaces" => RadianceCascadesDebug::Surfaces,
                "world" => RadianceCascadesDebug::WorldCache,
                _ => RadianceCascadesDebug::Off,
            },
            ..default()
        });
    }
}

fn controls(
    keys: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    mut sway: ResMut<Sway>,
    mut cameras: Query<(Entity, Option<&mut RadianceCascades>), With<Camera3d>>,
) {
    for (entity, radiance_cascades) in &mut cameras {
        if keys.just_pressed(KeyCode::Space) {
            if radiance_cascades.is_some() {
                commands.entity(entity).remove::<RadianceCascades>();
            } else {
                commands.entity(entity).insert(RadianceCascades {
                    sky_brightness: 0.0,
                    ..default()
                });
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
    if keys.just_pressed(KeyCode::KeyM) {
        sway.0 = !sway.0;
    }
}

fn sway(time: Res<Time>, sway: Res<Sway>, mut cameras: Query<&mut Transform, With<Camera3d>>) {
    if !sway.0 {
        return;
    }
    let t = time.elapsed_secs() * 0.4;
    for mut transform in &mut cameras {
        *transform = Transform::from_xyz(2.5 * t.sin(), 2.1 + 0.6 * (t * 0.7).cos(), 8.2)
            .looking_at(Vec3::new(0.0, 1.8, 0.0), Vec3::Y);
    }
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
    // `RC_VIEW=away`: after a look at the room, step in and face a wall, with
    // every light source behind the camera.
    if shot.frame == 40 && std::env::var("RC_VIEW").as_deref() == Ok("away") {
        for mut transform in &mut cameras {
            *transform =
                Transform::from_xyz(1.8, 1.9, 1.6).looking_at(Vec3::new(-3.0, 1.4, 0.4), Vec3::Y);
        }
    }
    // `RC_YAW=<degrees>` turns the camera a little just before the capture, to
    // compare against a capture without: the lighting should not care.
    let at = env_or("RC_FRAME", 90.0) as u32;
    if shot.frame == at - env_or("RC_SETTLE", 3.0) as u32 {
        for mut transform in &mut cameras {
            transform.rotate_y(env_or("RC_YAW", 0.0).to_radians());
        }
    }
    // Give pipelines time to compile and the bounces time to converge.
    if shot.frame == at {
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(shot.path.clone()));
    }
    if shot.frame == at + 30 {
        exit.write(AppExit::Success);
    }
}
