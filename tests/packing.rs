use revo2::{
    evolution::{self, rng::Rng, Config, Problem},
    problems::packing::{Layout, PackingConfig, PackingProblem, Rectangle},
};
use std::sync::Arc;

fn feasible(problem: &PackingProblem, layout: &Layout) {
    assert_eq!(layout.placements.len(), problem.rectangles().len());
    let mut seen = vec![false; problem.rectangles().len()];
    let mut sum = 0;
    for (a, p) in layout.placements.iter().enumerate() {
        assert!(!seen[p.id]);
        seen[p.id] = true;
        let r = &problem.rectangles()[p.id];
        assert_eq!(
            (p.width, p.height),
            if p.rotated {
                (r.height, r.width)
            } else {
                (r.width, r.height)
            }
        );
        assert!(p.x + p.width <= layout.width && p.y + p.height <= layout.height);
        sum += p.width * p.height;
        for q in &layout.placements[..a] {
            assert!(
                p.x + p.width <= q.x
                    || q.x + q.width <= p.x
                    || p.y + p.height <= q.y
                    || q.y + q.height <= p.y,
                "overlap between {} and {}",
                p.id,
                q.id
            );
        }
    }
    assert_eq!(sum, layout.rectangle_area);
    assert!(layout.area >= sum);
    assert!((0.0..=1.0).contains(&layout.utilization));
}

#[test]
fn variation_and_decoder_always_preserve_feasibility() {
    for n in [2, 24, 64] {
        let p = PackingProblem::preset(
            "mixed",
            &PackingConfig {
                rectangle_count: n,
                ..Default::default()
            },
        )
        .unwrap();
        let mut a = vec![0; p.genome_len()];
        let mut b = a.clone();
        let mut child = a.clone();
        let c = Config {
            mutation_rate: 1.0,
            mutation_scale: 4.0,
            ..Default::default()
        };
        for seed in 0..100 {
            let mut rng = Rng::new(seed, 0, 0);
            p.initialize(&mut a, &mut rng);
            p.initialize(&mut b, &mut rng);
            p.recombine(&a, &b, &mut child, &mut rng, true);
            p.mutate(&mut child, &mut rng, &c);
            feasible(&p, &p.decode(&child).unwrap());
        }
    }
}
#[test]
fn exact_tiling_scores_zero_and_invalid_genomes_are_rejected() {
    let p = PackingProblem::new(vec![
        Rectangle {
            width: 10,
            height: 20
        };
        2
    ])
    .unwrap();
    let g = vec![0, 1, 0, 0, 20];
    let l = p.decode(&g).unwrap();
    feasible(&p, &l);
    assert_eq!(l.area, 400);
    assert_eq!(p.score(&g), 0.0);
    assert!(p.decode(&[0, 0, 0, 0, 20]).is_err());
    assert!(PackingProblem::new(vec![Rectangle {
        width: 0,
        height: 20
    }])
    .is_err());
}
#[test]
fn cpu_packing_improves_without_overlap() {
    let p = Arc::new(
        PackingProblem::preset(
            "mixed",
            &PackingConfig {
                rectangle_count: 12,
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let config = Config {
        width: 4,
        height: 5,
        ..Default::default()
    };
    let mut e = evolution::create("cpu", &config, p.clone()).unwrap();
    let first = e.stats().unwrap().best_score;
    e.step(30).unwrap();
    assert!(e.stats().unwrap().best_score <= first);
    for i in 0..config.population() {
        feasible(&p, &p.decode(&e.genome(i).unwrap()).unwrap());
    }
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an NVIDIA GPU and NVRTC"]
fn packing_uses_the_same_cuda_engine_and_matches_cpu() {
    for n in [2, 24, 64] {
        let p = Arc::new(
            PackingProblem::preset(
                "mixed",
                &PackingConfig {
                    rectangle_count: n,
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let c = Config {
            width: 4,
            height: 3,
            ..Default::default()
        };
        let mut a = evolution::create("cpu", &c, p.clone()).unwrap();
        let mut b = evolution::create("cuda", &c, p.clone()).unwrap();
        assert!(b.backend().starts_with("CUDA"));
        for _ in 0..5 {
            let af = a.fitness().unwrap();
            let bf = b.fitness().unwrap();
            for i in 0..c.population() {
                assert_eq!(
                    a.genome(i).unwrap(),
                    b.genome(i).unwrap(),
                    "{n} rectangle genome {i}"
                );
                assert!(
                    (af[i] - bf[i]).abs() < 1e-6,
                    "packing score CPU {} GPU {}",
                    af[i],
                    bf[i]
                );
                feasible(&p, &p.decode(&b.genome(i).unwrap()).unwrap());
            }
            a.step(1).unwrap();
            b.step(1).unwrap();
        }
    }
}
