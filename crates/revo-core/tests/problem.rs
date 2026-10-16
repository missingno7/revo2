use revo_core::{self as core, rng::Rng, Config, CudaProblem, Problem};
use std::sync::Arc;
struct Sphere(f32);
impl Problem for Sphere {
    fn name(&self) -> &str {
        "external_sphere"
    }
    fn genome_len(&self) -> usize {
        1
    }
    fn validate(&self) -> anyhow::Result<()> {
        Ok(())
    }
    fn initialize(&self, g: &mut [u32], rng: &mut Rng) {
        g[0] = rng.next_u32() % 21;
    }
    fn recombine(&self, a: &[u32], _b: &[u32], g: &mut [u32], _rng: &mut Rng, _cross: bool) {
        g.copy_from_slice(a);
    }
    fn mutate(&self, g: &mut [u32], rng: &mut Rng, c: &Config) {
        if rng.unit() < c.mutation_rate {
            g[0] = rng.next_u32() % 21;
        }
    }
    fn score(&self, g: &[u32]) -> f32 {
        let x = g[0] as f32 - 10.0;
        x * x + self.0
    }
    fn cuda(&self) -> Option<CudaProblem> {
        Some(unsafe {
            CudaProblem::new(r#"
        __device__ void problem_initialize(unsigned*g,const unsigned*d,unsigned&r){g[0]=next(r)%21;}
        __device__ void problem_recombine(const unsigned*a,const unsigned*b,unsigned*g,const unsigned*d,unsigned&r,bool c){g[0]=a[0];}
        __device__ void problem_mutate(unsigned*g,const unsigned*d,unsigned&r,float rate,float scale){if(unit(r)<rate)g[0]=next(r)%21;}
        __device__ float problem_score(const unsigned*g,const unsigned*d,unsigned lane,unsigned lanes){float x=float(g[0])-10.0f;return lane==0?x*x+__uint_as_float(d[0]):0.0f;}
    "#.into(),vec![self.0.to_bits()],16)
        })
    }
}
#[test]
fn external_problem_works_without_any_demo_dependency() {
    let c = Config {
        width: 4,
        height: 3,
        ..Default::default()
    };
    let mut e = core::create("cpu", &c, Arc::new(Sphere(-200.0))).unwrap();
    e.step(10).unwrap();
    let scores = e.fitness().unwrap();
    let s = e.stats().unwrap();
    assert!(s.worst_score < 0.0);
    assert_eq!(
        s.best_score,
        scores.iter().copied().fold(f32::INFINITY, f32::min)
    );
    assert!(e.genome(c.population()).is_err());
}

fn rectangular_populations(backend: &str) {
    assert_eq!(core::neighbours(0, 1, 3), [0, 0, 0, 2, 1]);
    assert_eq!(core::neighbours(0, 3, 1), [0, 2, 1, 0, 0]);
    for (width, height) in [(1, 1), (1, 7), (7, 1), (37, 13)] {
        let c = Config {
            width,
            height,
            ..Default::default()
        };
        let mut e = core::create(backend, &c, Arc::new(Sphere(-200.0))).unwrap();
        let before = e.fitness().unwrap();
        assert_eq!(before.len(), (width * height) as usize);
        e.step(4).unwrap();
        let after = e.fitness().unwrap();
        assert_eq!(e.stats().unwrap().generation, 4);
        assert!(after.iter().zip(before).all(|(a, b)| *a <= b));
        assert!(e.genome((width * height - 1) as usize).is_ok());
    }
    for (width, height) in [(0, 1), (1, 0), (257, 1), (1, 257)] {
        assert!(Config {
            width,
            height,
            ..Default::default()
        }
        .validate()
        .is_err());
    }
}
#[test]
fn arbitrary_rectangular_and_single_axis_populations() {
    rectangular_populations("cpu");
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an NVIDIA GPU and NVRTC"]
fn cuda_rectangular_and_single_axis_populations() {
    rectangular_populations("cuda");
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an NVIDIA GPU and NVRTC"]
fn generic_cuda_statistics_support_negative_and_large_scores() {
    for offset in [-200.0, 50.0, f32::NAN] {
        let c = Config {
            width: 3,
            height: 2,
            mutation_rate: 0.5,
            ..Default::default()
        };
        let p = Arc::new(Sphere(offset));
        let mut a = core::create("cpu", &c, p.clone()).unwrap();
        let mut b = core::create("cuda", &c, p).unwrap();
        for _ in 0..3 {
            assert_eq!(a.fitness().unwrap(), b.fitness().unwrap());
            let sa = a.stats().unwrap();
            let sb = b.stats().unwrap();
            assert_eq!(sa.best_index, sb.best_index);
            assert_eq!(sa.best_score, sb.best_score);
            assert_eq!(sa.worst_score, sb.worst_score);
            assert!((sa.mean_score - sb.mean_score).abs() < 0.001);
            a.step(3).unwrap();
            b.step(3).unwrap();
        }
    }
}
