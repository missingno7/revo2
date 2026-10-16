pub mod program;
#[cfg(feature = "cuda")]
pub mod renderer;
pub mod target;
use crate::evolution::{rng::Rng, Config, CudaProblem, Problem};
use anyhow::Result;
pub use program::{Node, Program};
use serde::{Deserialize, Serialize};
pub use target::{Sample, Target};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageConfig {
    pub nodes: usize,
    pub complexity_weight: f32,
    pub sample_resolution: u32,
}
impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            nodes: 24,
            complexity_weight: 0.0005,
            sample_resolution: 24,
        }
    }
}
impl ImageConfig {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            (3..=64).contains(&self.nodes),
            "program length must be 3–64 nodes"
        );
        anyhow::ensure!(
            (4..=128).contains(&self.sample_resolution),
            "sample resolution must be 4–128"
        );
        anyhow::ensure!(
            self.complexity_weight.is_finite() && (0.0..=1.0).contains(&self.complexity_weight),
            "complexity weight must be 0–1"
        );
        Ok(())
    }
}
pub struct ImageProblem {
    pub config: ImageConfig,
    samples: Vec<Sample>,
}
impl ImageProblem {
    pub fn new(config: ImageConfig, samples: Vec<Sample>) -> Result<Self> {
        let p = Self { config, samples };
        p.validate()?;
        Ok(p)
    }
    pub fn from_target(config: ImageConfig, target: &Target) -> Result<Self> {
        config.validate()?;
        let samples = target.samples(config.sample_resolution);
        Self::new(config, samples)
    }
    pub fn decode(genome: &[u32]) -> Result<Program> {
        anyhow::ensure!(
            genome.len().is_multiple_of(4),
            "image genome must contain whole nodes"
        );
        let p = Program {
            nodes: genome
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| Node {
                    op: w[0],
                    a: w[1],
                    b: w[2],
                    parameter: f32::from_bits(w[3]),
                })
                .collect(),
        };
        p.validate()?;
        Ok(p)
    }
}
impl Problem for ImageProblem {
    fn name(&self) -> &str {
        "image"
    }
    fn genome_len(&self) -> usize {
        self.config.nodes * 4
    }
    fn validate(&self) -> Result<()> {
        self.config.validate()?;
        anyhow::ensure!(
            !self.samples.is_empty()
                && self.samples.len() <= 128 * 128
                && self.samples.iter().all(|s| s.x.is_finite()
                    && s.y.is_finite()
                    && (-1.0..=1.0).contains(&s.x)
                    && (-1.0..=1.0).contains(&s.y)
                    && s.color
                        .iter()
                        .all(|v| v.is_finite() && (0.0..=1.0).contains(v))),
            "invalid image fitness samples"
        );
        Ok(())
    }
    fn initialize(&self, g: &mut [u32], rng: &mut Rng) {
        for (i, w) in g.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let n = program::random_node(rng, i);
            w.copy_from_slice(&[n.op, n.a, n.b, n.parameter.to_bits()]);
        }
    }
    fn recombine(&self, a: &[u32], b: &[u32], out: &mut [u32], rng: &mut Rng, cross: bool) {
        let split = rng.index(self.config.nodes) * 4;
        out.copy_from_slice(a);
        if cross {
            out[split..].copy_from_slice(&b[split..]);
        }
    }
    fn mutate(&self, g: &mut [u32], rng: &mut Rng, c: &Config) {
        for (i, n) in g.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            if rng.unit() < c.mutation_rate {
                match rng.index(4) {
                    0 => n[0] = rng.next_u32() % program::OPS,
                    1 => n[1] = rng.index(program::INPUTS + i) as u32,
                    2 => n[2] = rng.index(program::INPUTS + i) as u32,
                    _ => {
                        n[3] = (f32::from_bits(n[3]) + (rng.unit() * 2.0 - 1.0) * c.mutation_scale)
                            .clamp(-4.0, 4.0)
                            .to_bits()
                    }
                }
            }
        }
    }
    fn score(&self, g: &[u32]) -> f32 {
        let p = Self::decode(g).expect("valid image genome");
        target::mse(&p, &self.samples)
            + self.config.complexity_weight * p.active_mask().count_ones() as f32
                / self.config.nodes as f32
    }
    fn cuda(&self) -> Option<CudaProblem> {
        let mut data = vec![
            self.samples.len() as u32,
            self.config.complexity_weight.to_bits(),
        ];
        for s in &self.samples {
            data.extend([
                s.x.to_bits(),
                s.y.to_bits(),
                s.color[0].to_bits(),
                s.color[1].to_bits(),
                s.color[2].to_bits(),
            ]);
        }
        // SAFETY: These callbacks match the validated node/sample encodings.
        Some(unsafe {
            CudaProblem::new(
                format!(
                    "#define NODES {}\n{}",
                    self.config.nodes,
                    include_str!("cuda.cu")
                ),
                data,
                64,
            )
        })
    }
}
