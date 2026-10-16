use super::{neighbours, Config, Engine, Stats};
use crate::{
    program::{Program, Rng, INPUTS},
    target::{mse, Sample},
};
use anyhow::Result;
use rayon::prelude::*;

pub struct CpuEngine {
    config: Config,
    samples: Vec<Sample>,
    programs: Vec<Program>,
    scores: Vec<f32>,
    generation: u32,
}

pub fn score(program: &Program, samples: &[Sample], weight: f32) -> f32 {
    mse(program, samples)
        + weight * program.active_mask().count_ones() as f32 / program.nodes.len() as f32
}

impl CpuEngine {
    pub(crate) fn new(config: Config, samples: Vec<Sample>) -> Self {
        let programs: Vec<_> = (0..config.population())
            .map(|i| Program::random(config.nodes, config.seed, i as u32))
            .collect();
        let scores = programs
            .par_iter()
            .map(|p| score(p, &samples, config.complexity_weight))
            .collect();
        Self {
            config,
            samples,
            programs,
            scores,
            generation: 0,
        }
    }

    fn offspring(&self, i: usize) -> Program {
        let c = &self.config;
        let mut rng = Rng::new(c.seed, i as u32, self.generation + 1);
        let neighbourhood = neighbours(i, c.width as usize, c.height as usize);
        let tournament = |rng: &mut Rng| {
            let a = neighbourhood[rng.index(5)];
            let b = neighbourhood[rng.index(5)];
            if self.scores[a] <= self.scores[b] {
                a
            } else {
                b
            }
        };
        let first = tournament(&mut rng);
        let second = tournament(&mut rng);
        let cross = rng.unit() < c.crossover_rate;
        let split = rng.index(c.nodes);
        let mut program = self.programs[first].clone();
        for j in 0..c.nodes {
            if cross && j >= split {
                program.nodes[j] = self.programs[second].nodes[j];
            }
            if rng.unit() < c.mutation_rate {
                let choice = rng.index(4);
                let node = &mut program.nodes[j];
                match choice {
                    0 => node.op = rng.next() % crate::program::OPS,
                    1 => node.a = rng.index(INPUTS + j) as u32,
                    2 => node.b = rng.index(INPUTS + j) as u32,
                    _ => {
                        node.parameter = (node.parameter
                            + (rng.unit() * 2.0 - 1.0) * c.mutation_scale)
                            .clamp(-4.0, 4.0)
                    }
                }
            }
        }
        program
    }
}

impl Engine for CpuEngine {
    fn backend(&self) -> &str {
        "CPU · Rayon"
    }
    fn step(&mut self, generations: u32) -> Result<()> {
        anyhow::ensure!(
            self.generation.checked_add(generations).is_some(),
            "generation counter exhausted"
        );
        for _ in 0..generations {
            let next: Vec<_> = (0..self.programs.len())
                .into_par_iter()
                .map(|i| {
                    let p = self.offspring(i);
                    let fitness = score(&p, &self.samples, self.config.complexity_weight);
                    if fitness <= self.scores[i] {
                        (p, fitness)
                    } else {
                        (self.programs[i].clone(), self.scores[i])
                    }
                })
                .collect();
            for (i, (p, s)) in next.into_iter().enumerate() {
                self.programs[i] = p;
                self.scores[i] = s;
            }
            self.generation += 1;
        }
        Ok(())
    }
    fn stats(&self) -> Result<Stats> {
        let best_index = (0..self.scores.len())
            .min_by(|&a, &b| self.scores[a].total_cmp(&self.scores[b]).then(a.cmp(&b)))
            .unwrap();
        Ok(Stats {
            generation: self.generation,
            best_index,
            best_score: self.scores[best_index],
            mean_score: self.scores.iter().sum::<f32>() / self.scores.len() as f32,
            worst_score: self.scores.iter().copied().fold(0.0, f32::max),
        })
    }
    fn fitness(&self) -> Result<Vec<f32>> {
        Ok(self.scores.clone())
    }
    fn program(&self, index: usize) -> Result<Program> {
        self.programs
            .get(index)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("cell index out of range"))
    }
}
