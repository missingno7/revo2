use super::{rng::Rng, Config};
use anyhow::Result;
/// A fixed-length genome encoded as 32-bit words. Each problem owns its
/// encoding, feasible variation, objective, and immutable shared data.
/// The engine minimizes scores; nonfinite scores receive the worst penalty.
pub trait Problem: Send + Sync {
    fn name(&self) -> &str;
    fn genome_len(&self) -> usize;
    fn validate(&self) -> Result<()>;
    fn initialize(&self, genome: &mut [u32], rng: &mut Rng);
    /// Always write every word. Preserve validity even with crossover disabled.
    fn recombine(
        &self,
        first: &[u32],
        second: &[u32],
        child: &mut [u32],
        rng: &mut Rng,
        crossover: bool,
    );
    fn mutate(&self, genome: &mut [u32], rng: &mut Rng, config: &Config);
    fn score(&self, genome: &[u32]) -> f32;
    fn cuda(&self) -> Option<CudaProblem> {
        None
    }
}
/// Device callbacks compile into the shared kernels. See docs/problems.md.
pub struct CudaProblem {
    pub(crate) source: String,
    pub(crate) data: Vec<u32>,
    /// Power of two, 1–256. Each lane returns a contribution to the total score.
    pub(crate) block_size: u32,
}
impl CudaProblem {
    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn data(&self) -> &[u32] {
        &self.data
    }
    pub fn block_size(&self) -> u32 {
        self.block_size
    }
    /// Supply trusted CUDA callbacks for this problem.
    ///
    /// # Safety
    /// Callbacks must obey the ABI in docs/problems.md, stay within genome/data
    /// bounds, preserve valid genomes, and use barriers uniformly across lanes.
    /// They must not retain pointers or race with other candidate blocks.
    pub unsafe fn new(source: String, data: Vec<u32>, block_size: u32) -> Self {
        Self {
            source,
            data,
            block_size,
        }
    }
}
pub(crate) fn finite_score(score: f32) -> f32 {
    if score.is_finite() {
        score
    } else {
        f32::MAX
    }
}
