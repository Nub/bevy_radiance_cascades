# bevy_radiance_cascades

Diffuse global illumination for Bevy 0.20 without ray tracing hardware and
without noise.

It implements Alexander Sannikov's radiance cascades in the screen-space form
shown in [Rendering Path of Exile 2](https://www.youtube.com/watch?v=TrHHTQqmAaM),
plus a world-space cache so that light and occluders do not vanish when they
leave the screen.

## In a game

Two rooms of `h4ck`, a Bevy game this was written for and is tested in. Both
are lit by a few lamps and many small glowing strips and panels. Left: the
game's rasterized lighting with a flat fill light standing in for bounced
light. Right: the same frame with the fill replaced by radiance cascades
(screen space and world cache, default settings, DLSS quality at 1080p).

| Off | On |
| --- | --- |
| ![Room A, cascades off](docs/room_a_off.jpg) | ![Room A, cascades on](docs/room_a_on.jpg) |
| ![Room B, cascades off](docs/room_b_off.jpg) | ![Room B, cascades on](docs/room_b_on.jpg) |

In room A the glowing ring under the machine lights the floor around it and
the cyan strips light the cabinets beside them. In room B the lamp's light
bounces off the cabinets and the floor into the parts of the room it does not
reach. None of these glows is a light source as far as the engine knows: they
are emissive materials, and the cascades are what makes them cast light.

Where little light reaches, the result is darker than a flat fill, which lights
everything alike.

```rust
use bevy::prelude::*;
use bevy_radiance_cascades::{RadianceCascades, RadianceCascadesPlugin};

App::new()
    .add_plugins((DefaultPlugins, RadianceCascadesPlugin))
    // The cascades are the ambient light now.
    .insert_resource(GlobalAmbientLight { brightness: 0.0, ..default() });

commands.spawn((Camera3d::default(), Msaa::Off, RadianceCascades::default()));
```

`cargo run --example room` shows a closed room lit by one spot light and one
glowing bar. `Space` toggles the cascades, `D` cycles the debug views.

## How it works

Every frame, after the opaque pass:

1. **Probes on the depth buffer.** Cascade `n` has a probe every
   `probe_spacing * 2^n` pixels, and each probe has `(base_resolution * 2^n)^2`
   directions. So every cascade has the same number of rays: near light is
   sampled from many places in few directions, far light from few places in
   many directions.
2. **Each cascade traces one interval.** Cascade `n` only covers the distances
   its angular resolution can resolve. Rays are traced exactly against a depth
   pyramid (no stepping over thin occluders), and read what they hit from a
   prefiltered copy of the lit scene, sized to the cone each ray stands for.
3. **Rays the screen cannot answer go to the world cache.** A ray that leaves
   the screen, or passes behind something, continues through a clipmap of voxels
   around the camera that remembers every lit surface the camera has seen.
4. **Merge, top down.** A ray that found nothing in its interval continues with
   the radiance of the cascade above, interpolated between the surrounding
   probes. Cascade 0 ends up with the complete radiance field.
5. **Gather and add.** Each pixel integrates cascade 0 over its hemisphere and
   the result, times the surface's diffuse color, is added to the scene. It is
   also what surfaces re-emit next frame, so bounces accumulate.

Nothing is traced at random, so there is no sampling noise to denoise. What
does change from frame to frame is the view itself, when it is jittered for TAA
or DLSS: small lights are drawn differently every frame. So the light each
pixel receives is averaged with its neighbours on the same surface and blended
over a few frames (`temporal_blend`, reprojected with motion vectors where the
camera has them); set it to 1 to show every frame as traced. What has
accumulated is held within reach of the light of the last few frames, so a
light that goes out is gone in about five frames, and a frame that is far
brighter than what came before it is taken for a stray sample and held back.

## Settings

All on the `RadianceCascades` component; see its docs for detail.

| Field | Default | |
| --- | --- | --- |
| `probe_spacing` | 4 | Pixels between the finest probes. Cost per cascade is `(base_resolution / probe_spacing)^2` rays per pixel. |
| `base_resolution` | 4 | Directions across the finest probes' maps (16 directions). |
| `cascade_count` | 6 | The last cascade always reaches to the edge of the screen. |
| `interval_length` | 2 | Length of the finest rays in probe spacings. Longer is steadier under camera motion. |
| `thickness` | 0.5 | How thick, in world units, on-screen surfaces are assumed to be. |
| `bounce` | 1 | 1 for physically correct multiple bounces, 0 for one. |
| `temporal_blend` | 0.1 | How much of each frame's light shows. Lower is steadier under TAA/DLSS jitter, slower to follow changing lights; 1 is no accumulation. |
| `ambient_occlusion` | 1 | How much of the camera's `ScreenSpaceAmbientOcclusion`, if it has one, darkens the indirect light in creases and contacts the probes are too coarse for. |
| `specular` | 1 | How much of the arriving light a surface's specular side reflects to the eye. Metals have no diffuse side: without it the bounced light does not show on them. |
| `specular_roughness` | 0.3..0.6 | The perceptual roughness over which `specular` fades in. Smoother surfaces are left to screen-space reflections. |
| `emissive_scale`, `emissive_unexposed` | 1, off | For a Bevy fork that stores the G-buffer's emissive scaled or without the exposure applied. |
| `amortize_far` | on | Traces the far cascades every second or fourth frame instead of every one. Far light is a few frames late; the cost falls by about a third. |
| `sky_color`, `sky_brightness` | white, 80 | What rays that escape everything see. Set the brightness to 0 indoors. |
| `world_cache` | on | `None` for purely screen-space lighting. |
| `debug` | off | `Indirect`, `Irradiance`, `Surfaces` (what the cascades know of the scene), `WorldCache`. |

## Lamps the camera has not seen

The cascades light the scene with what is on screen and what the world cache
remembers. A lamp that has never been in view lights nothing until it is, and
one in view lights only as bright as it is drawn. Put a `RadianceEmitter` on a
lamp's lit face to hand it to the cascades outright:

```rust
commands.spawn((
    Mesh3d(face),
    MeshMaterial3d(glowing),
    RadianceEmitter {
        radiance: LinearRgba::rgb(40_000.0, 40_000.0, 36_000.0), // cd/m^2
        area: 0.36,
        normal: Dir3::NEG_Y,
        range: 30.0,
        // The room it hangs in, relative to it: its light stays inside.
        bounds: Some((Vec3::new(-4.0, -3.0, -4.0), Vec3::new(4.0, 0.5, 4.0))),
    },
));
```

It lights like a flat emissive surface of that luminance and area, from
wherever it is, and the mesh's own glow can be whatever looks right on screen.
The 64 nearest the camera count. What stands between an emitter and a surface
is known only where the camera has looked, so give each its room as `bounds`,
or it lights the next room through a wall nobody has seen yet.

## Cost

Measured in a real game scene (an interior of `h4ck`) on an RTX 5090, rendering
1280x720 upscaled to 2560x1440 with DLSS, world cache on, 8 levels of 128
voxels: 0.9 to 1.0 ms at the default `probe_spacing` of 4, 1.7 to 2.1 ms at 2.
Other hardware has not been measured. The cost does not depend on the number
of lights or emissive surfaces. With `RenderDiagnosticsPlugin`, each stage
reports its own GPU time (`rc_pyramids`, `rc_voxels`, `rc_cascade_NN`,
`rc_gather`, `rc_resolve`).

## Requirements and limitations

* The camera needs `Hdr`, `DeferredPrepass`, `DepthPrepass` (all inserted
  automatically) and `Msaa::Off`. The plugin makes deferred the default opaque
  render method.
* **Diffuse only.** Specular reflections still come from whatever you use for
  them. An `EnvironmentMapLight` adds its own diffuse light on top.
* **Only deferred materials receive indirect light.** Meshes drawn forward
  still occlude and bounce light, but there is no albedo to light them with. The
  `Surfaces` debug view shows which is which. Transparent meshes are ignored.
* **The world cache only knows what has been on screen.** Light sources the
  camera has never looked at contribute nothing until it does, and what it
  remembers of places it no longer sees is as they were when last seen.
* Voxels of the world cache are 12.5 cm by default: thinner walls can leak
  remembered light.
* Perspective projections only, and the camera's viewport must start at the
  origin of its render target.
* Exposure changes are handled, but the cache is not shared between cameras.
* Work in progress: stability under camera motion and the cost of the gather
  are being worked on.
