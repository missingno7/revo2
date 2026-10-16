use super::{Config, Engine, Stats};
use crate::{
    program::{Node, Program},
    target::Sample,
};
use anyhow::{Context, Result};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use std::sync::Arc;

pub struct CudaEngine {
    config: Config,
    stream: Arc<CudaStream>,
    current: CudaSlice<Node>,
    next: CudaSlice<Node>,
    scores: CudaSlice<f32>,
    next_scores: CudaSlice<f32>,
    samples: CudaSlice<Sample>,
    sample_count: u32,
    evaluate: CudaFunction,
    breed: CudaFunction,
    replace: CudaFunction,
    summarize: CudaFunction,
    render: CudaFunction,
    generation: u32,
    name: String,
}

impl CudaEngine {
    pub(crate) fn new(config: Config, samples: Vec<Sample>) -> Result<Self> {
        // These presence probes load system CUDA libraries without invoking APIs.
        anyhow::ensure!(
            unsafe { cudarc::driver::sys::is_culib_present() },
            "NVIDIA CUDA driver not found. Install an NVIDIA driver, or select --backend cpu."
        );
        anyhow::ensure!(unsafe {cudarc::nvrtc::sys::is_culib_present()},"CUDA runtime compiler (NVRTC 12.x) not found. Install CUDA 12.8 or add its bin directory to PATH. See README.md.");
        let ctx = CudaContext::new(0).context("opening NVIDIA device 0")?;
        let name = format!("CUDA · {}", ctx.name()?);
        let (major, minor) = ctx.compute_capability()?;
        let source = format!(
            "#define NODES {}\n{}",
            config.nodes,
            include_str!("../../kernels/image.cu")
        );
        let ptx = compile_ptx_with_opts(
            source,
            CompileOptions {
                fmad: Some(false),
                options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
                name: Some("revo_image.cu".into()),
                ..Default::default()
            },
        )
        .context("compiling image evolution kernels")?;
        let module = ctx.load_module(ptx).context("loading CUDA kernels")?;
        let initialize = module.load_function("initialize")?;
        let stream = ctx.default_stream();
        let count = config.population();
        // Node/Sample have padding-free repr(C) layouts with DeviceRepr guarantees.
        let mut current = unsafe { stream.alloc::<Node>(count * config.nodes) }?;
        let next = unsafe { stream.alloc::<Node>(count * config.nodes) }?;
        let scores = stream.alloc_zeros::<f32>(count)?;
        let next_scores = stream.alloc_zeros::<f32>(count)?;
        let sample_count = samples.len() as u32;
        let device_samples = stream.clone_htod(&samples)?;
        let count_arg = count as u32;
        // SAFETY: Kernel signatures match the arguments; initialize writes every
        // Node before evaluation, and its bound is the allocated population size.
        unsafe {
            stream
                .launch_builder(&initialize)
                .arg(&mut current)
                .arg(&count_arg)
                .arg(&config.seed)
                .launch(LaunchConfig::for_num_elems(count_arg))
        }?;
        let mut engine = Self {
            evaluate: module.load_function("evaluate_population")?,
            breed: module.load_function("breed")?,
            replace: module.load_function("replace")?,
            summarize: module.load_function("summarize")?,
            render: module.load_function("render_image")?,
            config,
            stream,
            current,
            next,
            scores,
            next_scores,
            samples: device_samples,
            sample_count,
            generation: 0,
            name,
        };
        engine.evaluate_initial()?;
        Ok(engine)
    }

    fn evaluation_launch(&self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (self.config.population() as u32, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn evaluate_initial(&mut self) -> Result<()> {
        let launch = self.evaluation_launch();
        // SAFETY: Exactly one 64-thread block per allocated candidate; samples
        // and program lengths match the constants passed to the compiled kernel.
        unsafe {
            self.stream
                .launch_builder(&self.evaluate)
                .arg(&self.current)
                .arg(&self.samples)
                .arg(&self.sample_count)
                .arg(&self.config.complexity_weight)
                .arg(&mut self.scores)
                .launch(launch)
        }?;
        self.stream.synchronize()?;
        Ok(())
    }
}

impl Engine for CudaEngine {
    fn backend(&self) -> &str {
        &self.name
    }

    fn step(&mut self, generations: u32) -> Result<()> {
        anyhow::ensure!(
            self.generation.checked_add(generations).is_some(),
            "generation counter exhausted"
        );
        let count = self.config.population() as u32;
        let launch = LaunchConfig::for_num_elems(count);
        let evaluation = self.evaluation_launch();
        for _ in 0..generations {
            let generation = self.generation + 1;
            // SAFETY: Disjoint source/destination buffers each contain count
            // complete programs. All kernels execute in order on the same stream.
            unsafe {
                self.stream
                    .launch_builder(&self.breed)
                    .arg(&self.current)
                    .arg(&self.scores)
                    .arg(&mut self.next)
                    .arg(&self.config.width)
                    .arg(&self.config.height)
                    .arg(&self.config.seed)
                    .arg(&generation)
                    .arg(&self.config.mutation_rate)
                    .arg(&self.config.mutation_scale)
                    .arg(&self.config.crossover_rate)
                    .launch(launch)?;
                self.stream
                    .launch_builder(&self.evaluate)
                    .arg(&self.next)
                    .arg(&self.samples)
                    .arg(&self.sample_count)
                    .arg(&self.config.complexity_weight)
                    .arg(&mut self.next_scores)
                    .launch(evaluation)?;
                self.stream
                    .launch_builder(&self.replace)
                    .arg(&self.current)
                    .arg(&self.scores)
                    .arg(&mut self.next)
                    .arg(&mut self.next_scores)
                    .arg(&count)
                    .launch(launch)?;
            }
            std::mem::swap(&mut self.current, &mut self.next);
            std::mem::swap(&mut self.scores, &mut self.next_scores);
            self.generation = generation;
        }
        self.stream
            .synchronize()
            .context("executing evolution kernels")?;
        Ok(())
    }

    fn stats(&self) -> Result<Stats> {
        let count = self.config.population() as u32;
        let mut result = self.stream.alloc_zeros::<f32>(4)?;
        // SAFETY: summarize expects 256 lanes and writes exactly four floats.
        unsafe {
            self.stream
                .launch_builder(&self.summarize)
                .arg(&self.scores)
                .arg(&count)
                .arg(&mut result)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }?;
        let values = self.stream.clone_dtoh(&result)?;
        Ok(Stats {
            generation: self.generation,
            best_index: values[0] as usize,
            best_score: values[1],
            mean_score: values[2],
            worst_score: values[3],
        })
    }

    fn fitness(&self) -> Result<Vec<f32>> {
        Ok(self.stream.clone_dtoh(&self.scores)?)
    }

    fn program(&self, index: usize) -> Result<Program> {
        anyhow::ensure!(index < self.config.population(), "cell index out of range");
        let start = index * self.config.nodes;
        Ok(Program {
            nodes: self
                .stream
                .clone_dtoh(&self.current.slice(start..start + self.config.nodes))?,
        })
    }

    fn render(&self, index: usize, width: u32, height: u32) -> Result<image::RgbImage> {
        anyhow::ensure!(index < self.config.population(), "cell index out of range");
        anyhow::ensure!(
            (1..=4096).contains(&width) && (1..=4096).contains(&height),
            "render dimensions must be 1–4096"
        );
        let index = index as u32;
        let mut pixels = self
            .stream
            .alloc_zeros::<u8>((width * height * 3) as usize)?;
        // SAFETY: Each thread writes one RGB pixel within width*height*3 bytes.
        unsafe {
            self.stream
                .launch_builder(&self.render)
                .arg(&self.current)
                .arg(&index)
                .arg(&width)
                .arg(&height)
                .arg(&mut pixels)
                .launch(LaunchConfig::for_num_elems(width * height))
        }?;
        let pixels = self.stream.clone_dtoh(&pixels)?;
        Ok(image::RgbImage::from_raw(width, height, pixels).expect("exact pixel buffer length"))
    }
}
