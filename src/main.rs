//! This is the main file of the project. It contains structures used by all other parts of the
//! engine and the main method

/*#![deny(
    rust_2018_compatibility,
    rust_2018_idioms,
    future_incompatible,
    nonstandard_style,
    unused,
    missing_copy_implementations,
    clippy::all
)]*/

mod config;
mod galaxygen;
mod render;

use std::{env, f32::consts::PI, fs::File};

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use config::{Config, Construction};
use glam::{Mat4, Vec3};
use ron::de::from_reader;

#[derive(Clone, Copy, Debug, Zeroable, Pod)]
#[repr(C)]
/// An object with a position, velocity and mass that can be sent to the GPU.
pub struct Particle {
    /// Position
    pos: Vec3, // 4, 8, 12

    /// The radius of the particle (currently unused)
    radius: f32, // 16

    /// Velocity
    vel: Vec3, // 4, 8, 12
    _p: f32, // 16

    /// Mass
    mass: f64, // 4, 8
    _p2: [f32; 2], // 12, 16
}

#[derive(Clone, Copy, Debug, Zeroable, Pod)]
#[repr(C)]
/// All variables that define the state of the program. Will be sent to the GPU.
pub struct Globals {
    /// The camera matrix (projection x view matrix)
    matrix: Mat4, // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    /// The current camera position (used for particle size)
    camera_pos: Vec3, // 16, 17, 18
    /// The number of particles
    particles: u32, // 19
    /// Newton's law of gravitation has problems with 1D particles, this value works against
    /// gravitation in close ranges.
    safety: f64, // 20, 21
    /// How much time passes each frame
    delta: f32, // 22

    _p: f32, // 23
}

impl Particle {
    fn new(pos: Vec3, vel: Vec3, mass: f64, density: f64) -> Self {
        Self {
            pos,
            // V = 4/3*pi*r^3
            // V = m/ d
            // 4/3*pi*r^3 = m / d
            // r^3 = 3*m / (4*d*pi)
            // r = cbrt(3*m / (4*d*pi))
            radius: (3.0 * mass / (4.0 * density * PI as f64)).cbrt() as f32,
            vel,
            mass,
            _p: 0.0,
            _p2: [0.0; 2],
        }
    }
}

fn main() -> Result<()> {
    let config = read_config().unwrap_or_else(|| {
        println!("Using default config.");
        default_config()
    });

    // Construct particles from config
    let particles = config.construct_particles();

    let globals = Globals {
        matrix: Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)),
        camera_pos: config.camera_pos.into(),
        particles: particles.len() as u32,
        safety: config.safety,
        delta: 0.0,
        _p: 0.0,
    };

    pollster::block_on(render::run(globals, particles))
}

/// Read configuration file
fn read_config() -> Option<Config> {
    let input_path = env::args().nth(1)?;
    let f = File::open(&input_path).expect("Failed opening file!");
    let config = from_reader(f).expect("Failed to parse config!");

    Some(config)
}

fn default_config() -> Config {
    Config {
        camera_pos: [0.0, 0.0, 1e10],
        safety: 1e20,
        constructions: vec![
            Construction::Galaxy {
                center_pos: [-1e11, -1e11, 0.0],
                center_vel: [10e6, 0.0, 0.0],
                center_mass: 1e35,
                amount: 100000,
                normal: [1.0, 0.0, 0.0],
            },
            Construction::Galaxy {
                center_pos: [1e11, 1e11, 0.0],
                center_vel: [0.0, 0.0, 0.0],
                center_mass: 3e35,
                amount: 100000,
                normal: [1.0, 1.0, 0.0],
            },
        ],
    }
}
