use super::{Config, Engine, Problem, Stats};
use anyhow::{Context, Result};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use std::sync::Arc;
pub struct CudaEngine {
    config: Config,
    stream: Arc<CudaStream>,
    current: CudaSlice<u32>,
    next: CudaSlice<u32>,
    scores: CudaSlice<f32>,
    next_scores: CudaSlice<f32>,
    data: CudaSlice<u32>,
    words: usize,
    lanes: u32,
    evaluate: CudaFunction,
    breed: CudaFunction,
    replace: CudaFunction,
    summarize: CudaFunction,
    generation: u32,
    name: String,
}
impl CudaEngine {
    pub(crate) fn new(config: Config, problem: Arc<dyn Problem>) -> Result<Self> {
        let definition = problem
            .cuda()
            .context("problem has no CUDA implementation")?;
        anyhow::ensure!(
            (1..=256).contains(&definition.block_size) && definition.block_size.is_power_of_two(),
            "CUDA evaluation block size must be a power of two from 1 to 256"
        );
        anyhow::ensure!(
            unsafe { cudarc::driver::sys::is_culib_present() },
            "NVIDIA driver not found; install one or select --backend cpu"
        );
        anyhow::ensure!(
            unsafe { cudarc::nvrtc::sys::is_culib_present() },
            "NVRTC 12.x not found; install CUDA 12.8 or add its bin directory to PATH"
        );
        let ctx = CudaContext::new(0)?;
        let name = format!("CUDA · {}", ctx.name()?);
        let (major, minor) = ctx.compute_capability()?;
        let words = problem.genome_len();
        let lanes = definition.block_size;
        let source = format!(
            "#define GENOME_WORDS {words}\n#define EVALUATION_LANES {lanes}\n{}\n{}\n{}",
            include_str!("../kernels/common.cuh"),
            definition.source,
            include_str!("../kernels/evolution.cu")
        );
        let ptx = compile_ptx_with_opts(
            source,
            CompileOptions {
                fmad: Some(false),
                options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
                name: Some(format!("revo_{}.cu", problem.name())),
                ..Default::default()
            },
        )
        .context("compiling problem callbacks and evolution kernels")?;
        let module = ctx.load_module(ptx)?;
        let stream = ctx.default_stream();
        let count = config.population() as u32;
        let mut current = stream.alloc_zeros::<u32>(count as usize * words)?;
        let next = stream.alloc_zeros::<u32>(count as usize * words)?;
        let scores = stream.alloc_zeros::<f32>(count as usize)?;
        let next_scores = stream.alloc_zeros::<f32>(count as usize)?;
        let data_words: &[u32] = if definition.data.is_empty() {
            &[0]
        } else {
            &definition.data
        };
        let data = stream.clone_htod(data_words)?;
        let initialize = module.load_function("initialize")?;
        // SAFETY: Validated sizes and trusted problem callbacks obey the u32 ABI.
        unsafe {
            stream
                .launch_builder(&initialize)
                .arg(&mut current)
                .arg(&data)
                .arg(&count)
                .arg(&config.seed)
                .launch(LaunchConfig::for_num_elems(count))
        }?;
        let mut engine = Self {
            evaluate: module.load_function("evaluate_population")?,
            breed: module.load_function("breed")?,
            replace: module.load_function("replace")?,
            summarize: module.load_function("summarize")?,
            config,
            stream,
            current,
            next,
            scores,
            next_scores,
            data,
            words,
            lanes,
            generation: 0,
            name,
        };
        let launch = engine.evaluation_launch();
        // SAFETY: One block per candidate, with the validated lane count.
        unsafe {
            engine
                .stream
                .launch_builder(&engine.evaluate)
                .arg(&engine.current)
                .arg(&engine.data)
                .arg(&mut engine.scores)
                .launch(launch)
        }?;
        engine.stream.synchronize()?;
        Ok(engine)
    }
    fn evaluation_launch(&self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (self.config.population() as u32, 1, 1),
            block_dim: (self.lanes, 1, 1),
            shared_mem_bytes: 0,
        }
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
            // SAFETY: Disjoint, correctly sized buffers; kernels run in stream
            // order and trusted problem callbacks preserve their genome validity.
            unsafe {
                self.stream
                    .launch_builder(&self.breed)
                    .arg(&self.current)
                    .arg(&self.scores)
                    .arg(&mut self.next)
                    .arg(&self.data)
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
                    .arg(&self.data)
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
        // SAFETY: Exactly 256 lanes and four result words.
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
        let r = self.stream.clone_dtoh(&result)?;
        Ok(Stats {
            generation: self.generation,
            best_index: r[0] as usize,
            best_score: r[1],
            mean_score: r[2],
            worst_score: r[3],
        })
    }
    fn fitness(&self) -> Result<Vec<f32>> {
        Ok(self.stream.clone_dtoh(&self.scores)?)
    }
    fn genome(&self, index: usize) -> Result<Vec<u32>> {
        anyhow::ensure!(index < self.config.population(), "cell index out of range");
        let start = index * self.words;
        Ok(self
            .stream
            .clone_dtoh(&self.current.slice(start..start + self.words))?)
    }
}
