//! This module handles everything that has to do with the window. That includes opening a window,
//! parsing events and rendering. See shader.comp for the physics simulation algorithm.

use std::{collections::HashSet, f32::consts::PI, time::Instant};

use crate::{Globals, Particle};

use anyhow::Result;
use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::error::EventLoopError;
use winit::event::WindowEvent;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowBuilder};
use winit::{event, event_loop::EventLoop};

const TICKS_PER_FRAME: u32 = 3; // Number of simulation steps per redraw
const PARTICLES_PER_GROUP: u32 = 256; // REMEMBER TO CHANGE SHADER.COMP

fn build_matrix(pos: Vec3, dir: Vec3, aspect: f32) -> Mat4 {
    Mat4::perspective_rh(PI / 2.0, aspect, 1E8, 1E14)
        * Mat4::look_to_rh(pos, dir, Vec3::new(0.0, 1.0, 0.0))
}

pub async fn run(globals: Globals, particles: Vec<Particle>) -> Result<()> {
    let (state, event_loop) = init(globals, particles).await?;
    state.run(event_loop)?;
    Ok(())
}

async fn init(mut globals: Globals, particles: Vec<Particle>) -> Result<(State, EventLoop<()>)> {
    // How many bytes do the particles need
    let particles_size = (particles.len() * std::mem::size_of::<Particle>()) as u64;

    let work_group_count = ((particles.len() as f32) / (PARTICLES_PER_GROUP as f32)).ceil() as u32;

    let event_loop = EventLoop::new().unwrap();

    let window = WindowBuilder::new().build(&event_loop)?;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let surface = unsafe { instance.create_surface(&window) }?;

    let size = window.inner_size();
    // Try to grab mouse
    window
        .set_cursor_grab(winit::window::CursorGrabMode::Confined)
        .ok();
    window.set_cursor_visible(false);
    window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(
        window.primary_monitor(),
    )));

    // Pick a GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    println!("Graphics adapter: {:?}", adapter.get_info());

    // Request access to that GPU
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::SHADER_F64 | wgpu::Features::VERTEX_WRITABLE_STORAGE,
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .unwrap();

    // Load compute shader for the simulation
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::SpirV(wgpu::include_spirv_raw!("shader.comp.spv").source),
    });

    // Load vertex shader to set calculate perspective, size and position of particles
    let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::SpirV(wgpu::include_spirv_raw!("shader.vert.spv").source),
    });

    // Load fragment shader
    let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::SpirV(wgpu::include_spirv_raw!("shader.frag.spv").source),
    });

    // Create globals buffer to give global information to the shader
    let globals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        contents: bytemuck::bytes_of(&globals),
    });

    // Create buffer for the previous state of the particles
    let old_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        size: particles_size,
        mapped_at_creation: false,
    });

    let current_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        contents: bytemuck::cast_slice(&particles),
    });

    let surface_caps = surface.get_capabilities(&adapter);

    let surface_format = surface_caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(surface_caps.formats[0]);

    // Create swap chain to render images to
    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
    };
    surface.configure(&device, &surface_config);

    // Texture to keep track of which particle is in front (for the camera)
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: surface_config.width,
            height: surface_config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Describe the buffers that will be available to the GPU
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            // Globals
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Old Particle data
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Current Particle data
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // Create the resources described by the bind_group_layout
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            // Globals
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(globals_buffer.as_entire_buffer_binding()),
            },
            // Old Particle data
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(old_buffer.as_entire_buffer_binding()),
            },
            // Current Particle data
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(current_buffer.as_entire_buffer_binding()),
            },
        ],
    });

    // Combine all bind_group_layouts
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create compute pipeline
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "main",
    });

    // Create render pipeline
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &vs_module,
            entry_point: "main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &fs_module,
            entry_point: "main",
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent::REPLACE,
                    alpha: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::PointList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Front),
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState {
                front: wgpu::StencilFaceState::IGNORE,
                back: wgpu::StencilFaceState::IGNORE,
                read_mask: 0,
                write_mask: 0,
            },
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 1.0,
                clamp: 0.0,
            },
        }),
        multiview: None,
    });

    // Where is the camera looking at?
    let mut camera_dir = -globals.camera_pos;
    camera_dir = camera_dir.normalize();
    globals.matrix = build_matrix(
        globals.camera_pos,
        camera_dir,
        size.width as f32 / size.height as f32,
    );

    // Speed of the camera
    let fly_speed = 1E10;

    // Which keys are currently held down?
    let pressed_keys = HashSet::new();

    // Vector that points to the right of the camera
    let right = camera_dir.cross(Vec3::new(0.0, 1.0, 0.0)).normalize();

    // Time of the last tick
    let last_tick = Instant::now();

    let state = State {
        globals,
        particles,
        particles_size,

        window,
        device,
        queue,
        surface,

        work_group_count,
        size,
        surface_config,
        globals_buffer,
        old_buffer,
        current_buffer,

        depth_view,
        bind_group,
        compute_pipeline,
        render_pipeline,

        fly_speed,
        pressed_keys,
        camera_dir,
        right,
        last_tick,
    };

    return Ok((state, event_loop));
}

struct State {
    globals: Globals,
    particles: Vec<Particle>,
    particles_size: u64,

    window: Window,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface,

    work_group_count: u32,
    size: PhysicalSize<u32>,
    surface_config: wgpu::SurfaceConfiguration,

    globals_buffer: wgpu::Buffer,
    old_buffer: wgpu::Buffer,
    current_buffer: wgpu::Buffer,

    depth_view: wgpu::TextureView,

    bind_group: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,

    fly_speed: f32,
    pressed_keys: HashSet<KeyCode>,
    camera_dir: Vec3,
    right: Vec3,
    last_tick: Instant,
}

impl State {
    fn run(mut self, event_loop: EventLoop<()>) -> Result<(), EventLoopError> {
        event_loop.run(move |event, event_loop| {
            match event {
                // Move mouse
                event::Event::DeviceEvent {
                    event: event::DeviceEvent::MouseMotion { delta },
                    ..
                } => {
                    let camera_dir =
                        Quat::from_rotation_y(-delta.0 as f32 / 300.0) * self.camera_dir;
                    self.camera_dir =
                        Quat::from_axis_angle(self.right, delta.1 as f32 / 300.0) * camera_dir;
                }

                event::Event::WindowEvent { event, .. } => match event {
                    // Close window
                    event::WindowEvent::CloseRequested => {
                        event_loop.exit();
                    }

                    // Keyboard input
                    event::WindowEvent::KeyboardInput {
                        event:
                            event::KeyEvent {
                                physical_key: PhysicalKey::Code(keycode),
                                state: event::ElementState::Pressed,
                                ..
                            },
                        ..
                    } => {
                        match keycode {
                            // Exit
                            KeyCode::Escape => {
                                event_loop.exit();
                            }
                            KeyCode::Digit0 => {
                                self.globals.delta = 0.0;
                            }
                            KeyCode::Digit1 => {
                                self.globals.delta = 1E0;
                            }
                            KeyCode::Digit2 => {
                                self.globals.delta = 2E0;
                            }
                            KeyCode::Digit3 => {
                                self.globals.delta = 4E0;
                            }
                            KeyCode::Digit4 => {
                                self.globals.delta = 8E0;
                            }
                            KeyCode::Digit5 => {
                                self.globals.delta = 16E0;
                            }
                            KeyCode::Digit6 => {
                                self.globals.delta = 32E0;
                            }
                            KeyCode::KeyF => {
                                let delta = self.last_tick.elapsed();
                                println!(
                                    "delta: {:?}, fps: {:.2}",
                                    delta,
                                    1.0 / delta.as_secs_f32()
                                );
                            }
                            KeyCode::F11 => {
                                if self.window.fullscreen().is_some() {
                                    self.window.set_fullscreen(None);
                                } else {
                                    self.window.set_fullscreen(Some(
                                        winit::window::Fullscreen::Borderless(
                                            self.window.primary_monitor(),
                                        ),
                                    ));
                                }
                            }
                            _ => {}
                        }
                        self.pressed_keys.insert(keycode);
                    }

                    // Release key
                    event::WindowEvent::KeyboardInput {
                        event:
                            event::KeyEvent {
                                physical_key: PhysicalKey::Code(keycode),
                                state: event::ElementState::Released,
                                ..
                            },
                        ..
                    } => {
                        self.pressed_keys.remove(&keycode);
                    }

                    // Mouse scroll
                    event::WindowEvent::MouseWheel { delta, .. } => {
                        self.fly_speed *= (1.0
                            + (match delta {
                                event::MouseScrollDelta::LineDelta(_, c) => c as f32 / 8.0,
                                event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 64.0,
                            }))
                        .min(4.0)
                        .max(0.25);

                        self.fly_speed = self.fly_speed.min(1E13).max(1E9);
                    }

                    // Resize window
                    event::WindowEvent::Resized(new_size) => {
                        self.resize(new_size);
                    }
                    WindowEvent::RedrawRequested => {
                        match self.render() {
                            Ok(_) => {}
                            // Reconfigure the surface if lost
                            Err(wgpu::SurfaceError::Lost) => self.resize(self.size),
                            // The system is out of memory, we should probably quit
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                event_loop.exit();
                            }
                            // All other errors (Outdated, Timeout) should be resolved by the next frame
                            Err(e) => eprintln!("{:?}", e),
                        }
                    }
                    _ => {}
                },

                // No more events in queue
                event::Event::AboutToWait => {
                    self.window.request_redraw();
                }
                _ => {}
            }
        })
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.size = new_size;

        // Reconfigure surface
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);

        // Reset depth texture
        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: self.size.width,
                height: self.size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let delta = self.last_tick.elapsed();
        let dt = delta.as_secs_f32();
        self.last_tick = Instant::now();

        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.camera_dir = self.camera_dir.normalize();
        self.right = self.camera_dir.cross(Vec3::new(0.0, 1.0, 0.0));
        self.right = self.right.normalize();

        if self.pressed_keys.contains(&KeyCode::KeyA) {
            self.globals.camera_pos += -self.right * self.fly_speed * dt;
        }

        if self.pressed_keys.contains(&KeyCode::KeyD) {
            self.globals.camera_pos += self.right * self.fly_speed * dt;
        }

        if self.pressed_keys.contains(&KeyCode::KeyW) {
            self.globals.camera_pos += self.camera_dir * self.fly_speed * dt;
        }

        if self.pressed_keys.contains(&KeyCode::KeyS) {
            self.globals.camera_pos += -self.camera_dir * self.fly_speed * dt;
        }

        if self.pressed_keys.contains(&KeyCode::Space) {
            self.globals.camera_pos.y -= self.fly_speed * dt;
        }

        if self.pressed_keys.contains(&KeyCode::ShiftLeft) {
            self.globals.camera_pos.y += self.fly_speed * dt;
        }

        self.globals.matrix = build_matrix(
            self.globals.camera_pos,
            self.camera_dir,
            self.size.width as f32 / self.size.height as f32,
        );

        // Create new globals buffer
        let new_globals_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC,
                    contents: bytemuck::bytes_of(&self.globals),
                });

        // Upload the new globals buffer to the GPU
        encoder.copy_buffer_to_buffer(
            &new_globals_buffer,
            0,
            &self.globals_buffer,
            0,
            std::mem::size_of::<Globals>() as u64,
        );

        // Compute the simulation a few times
        for _ in 0..TICKS_PER_FRAME {
            encoder.copy_buffer_to_buffer(
                &self.current_buffer,
                0,
                &self.old_buffer,
                0,
                self.particles_size,
            );
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(self.work_group_count, 1, 1);
        }

        {
            // Render the current state
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.03,
                            g: 0.03,
                            b: 0.03,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: true,
                    }),
                }),
            });
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.draw(0..self.particles.len() as u32, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        Ok(())
    }
}
