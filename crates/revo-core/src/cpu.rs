use super::{neighbours, problem::finite_score, rng::Rng, Config, Engine, Problem, Stats};
use anyhow::Result;
use rayon::prelude::*;
use std::sync::Arc;
pub struct CpuEngine {
    config: Config,
    problem: Arc<dyn Problem>,
    current: Vec<Vec<u32>>,
    next: Vec<Vec<u32>>,
    scores: Vec<f32>,
    next_scores: Vec<f32>,
    generation: u32,
}
impl CpuEngine {
    pub(crate) fn new(config: Config, problem: Arc<dyn Problem>) -> Self {
        let current: Vec<_> = (0..config.population())
            .map(|i| {
                let mut g = vec![0; problem.genome_len()];
                problem.initialize(&mut g, &mut Rng::new(config.seed, i as u32, 0));
                g
            })
            .collect();
        let scores = current
            .par_iter()
            .map(|g| finite_score(problem.score(g)))
            .collect();
        let next = vec![vec![0; problem.genome_len()]; config.population()];
        let next_scores = vec![0.0; config.population()];
        Self {
            config,
            problem,
            current,
            next,
            scores,
            next_scores,
            generation: 0,
        }
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
            let c = &self.config;
            let scores = &self.scores;
            let current = &self.current;
            let problem = &self.problem;
            let generation = self.generation + 1;
            self.next
                .par_iter_mut()
                .zip(self.next_scores.par_iter_mut())
                .enumerate()
                .for_each(|(i, (child, score))| {
                    let mut rng = Rng::new(c.seed, i as u32, generation);
                    let hood = neighbours(i, c.width as usize, c.height as usize);
                    let select = |rng: &mut Rng| {
                        let a = hood[rng.index(5)];
                        let b = hood[rng.index(5)];
                        if scores[a] <= scores[b] {
                            a
                        } else {
                            b
                        }
                    };
                    let first = select(&mut rng);
                    let second = select(&mut rng);
                    let cross = rng.unit() < c.crossover_rate;
                    problem.recombine(&current[first], &current[second], child, &mut rng, cross);
                    problem.mutate(child, &mut rng, c);
                    *score = finite_score(problem.score(child));
                    if *score > scores[i] {
                        child.copy_from_slice(&current[i]);
                        *score = scores[i];
                    }
                });
            std::mem::swap(&mut self.current, &mut self.next);
            std::mem::swap(&mut self.scores, &mut self.next_scores);
            self.generation = generation;
        }
        Ok(())
    }
    fn stats(&self) -> Result<Stats> {
        let best_index = (0..self.scores.len())
            .min_by(|&a, &b| {
                self.scores[a]
                    .partial_cmp(&self.scores[b])
                    .expect("scores are sanitized")
                    .then(a.cmp(&b))
            })
            .unwrap();
        Ok(Stats {
            generation: self.generation,
            best_index,
            best_score: self.scores[best_index],
            mean_score: (self.scores.iter().map(|&x| x as f64).sum::<f64>()
                / self.scores.len() as f64) as f32,
            worst_score: self
                .scores
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max),
        })
    }
    fn fitness(&self) -> Result<Vec<f32>> {
        Ok(self.scores.clone())
    }
    fn genome(&self, index: usize) -> Result<Vec<u32>> {
        self.current
            .get(index)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("cell index out of range"))
    }
}
