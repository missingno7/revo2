pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod problem;
pub mod rng;
use anyhow::{bail, Result};
pub use problem::{CudaProblem, Problem};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub width: u32,
    pub height: u32,
    pub seed: u64,
    pub mutation_rate: f32,
    pub mutation_scale: f32,
    pub crossover_rate: f32,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            width: 32,
            height: 32,
            seed: 7,
            mutation_rate: 0.06,
            mutation_scale: 0.35,
            crossover_rate: 0.25,
        }
    }
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            (1..=256).contains(&self.width) && (1..=256).contains(&self.height),
            "population dimensions must be 1–256"
        );
        for (name, v) in [
            ("mutation rate", self.mutation_rate),
            ("crossover rate", self.crossover_rate),
        ] {
            anyhow::ensure!(
                v.is_finite() && (0.0..=1.0).contains(&v),
                "{name} must be between 0 and 1"
            );
        }
        anyhow::ensure!(
            self.mutation_scale.is_finite() && (0.0..=4.0).contains(&self.mutation_scale),
            "mutation scale must be 0–4"
        );
        Ok(())
    }
    pub fn population(&self) -> usize {
        (self.width * self.height) as usize
    }
}
#[derive(Clone, Serialize)]
pub struct Stats {
    pub generation: u32,
    pub best_index: usize,
    pub best_score: f32,
    pub mean_score: f32,
    pub worst_score: f32,
}
/// Rendering, file formats and interpretation belong to the application.
pub trait Engine: Send {
    fn backend(&self) -> &str;
    fn step(&mut self, generations: u32) -> Result<()>;
    fn stats(&self) -> Result<Stats>;
    fn fitness(&self) -> Result<Vec<f32>>;
    fn genome(&self, index: usize) -> Result<Vec<u32>>;
}
pub fn create(
    backend: &str,
    config: &Config,
    problem: Arc<dyn Problem>,
) -> Result<Box<dyn Engine>> {
    config.validate()?;
    problem.validate()?;
    anyhow::ensure!(
        (1..=4096).contains(&problem.genome_len()),
        "genomes must contain 1–4096 words"
    );
    match backend {
        "cpu" => Ok(Box::new(cpu::CpuEngine::new(config.clone(), problem))),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(Box::new(cuda::CudaEngine::new(config.clone(), problem)?)),
        _ => bail!("backend '{backend}' is unavailable; use cpu or build with the cuda feature"),
    }
}
pub fn neighbours(index: usize, width: usize, height: usize) -> [usize; 5] {
    let x = index % width;
    let y = index / width;
    [
        index,
        y * width + (x + width - 1) % width,
        y * width + (x + 1) % width,
        ((y + height - 1) % height) * width + x,
        ((y + 1) % height) * width + x,
    ]
}
