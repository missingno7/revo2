/// Scheduling-independent integer RNG shared by CPU and CUDA problems.
pub struct Rng(pub u32);
impl Rng {
    pub fn new(seed: u64, cell: u32, generation: u32) -> Self {
        let mut x = seed as u32 ^ (seed >> 32) as u32;
        x ^= cell.wrapping_mul(0x9e3779b9);
        x ^= generation.wrapping_mul(0x85ebca6b);
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846ca68b);
        x ^= x >> 16;
        Self(x.max(1))
    }
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16777216.0)
    }
    pub fn index(&mut self, n: usize) -> usize {
        (self.next_u32() % n as u32) as usize
    }
}
