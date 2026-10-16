pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;

use crate::{program::Program, target::Sample};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub width: u32,
    pub height: u32,
    pub nodes: usize,
    pub seed: u64,
    pub mutation_rate: f32,
    pub mutation_scale: f32,
    pub crossover_rate: f32,
    pub complexity_weight: f32,
    pub sample_resolution: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            width: 32,
            height: 32,
            nodes: 24,
            seed: 7,
            mutation_rate: 0.06,
            mutation_scale: 0.35,
            crossover_rate: 0.25,
            complexity_weight: 0.0005,
            sample_resolution: 24,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            (2..=256).contains(&self.width) && (2..=256).contains(&self.height),
            "population dimensions must be 2–256"
        );
        anyhow::ensure!(
            (3..=64).contains(&self.nodes),
            "program length must be 3–64 nodes"
        );
        anyhow::ensure!(
            (4..=128).contains(&self.sample_resolution),
            "sample resolution must be 4–128"
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
        anyhow::ensure!(
            self.complexity_weight.is_finite() && (0.0..=1.0).contains(&self.complexity_weight),
            "complexity weight must be 0–1"
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

pub trait Engine: Send {
    fn backend(&self) -> &str;
    fn step(&mut self, generations: u32) -> Result<()>;
    fn stats(&self) -> Result<Stats>;
    fn fitness(&self) -> Result<Vec<f32>>;
    fn program(&self, index: usize) -> Result<Program>;
    fn render(&self, index: usize, width: u32, height: u32) -> Result<image::RgbImage> {
        anyhow::ensure!(
            (1..=4096).contains(&width) && (1..=4096).contains(&height),
            "render dimensions must be 1–4096"
        );
        Ok(crate::target::render(&self.program(index)?, width, height))
    }
}

pub fn create(backend: &str, config: &Config, samples: Vec<Sample>) -> Result<Box<dyn Engine>> {
    config.validate()?;
    anyhow::ensure!(
        !samples.is_empty()
            && samples.iter().all(|s| s.x.is_finite()
                && s.y.is_finite()
                && (-1.0..=1.0).contains(&s.x)
                && (-1.0..=1.0).contains(&s.y)
                && s.color
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v))),
        "fitness samples must be nonempty, with finite coordinates in [-1,1] and colors in [0,1]"
    );
    match backend {
        "cpu" => Ok(Box::new(cpu::CpuEngine::new(config.clone(), samples))),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(Box::new(cuda::CudaEngine::new(config.clone(), samples)?)),
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
