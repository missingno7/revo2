use revo2::{
    evolution::{self, Config},
    problems::image::{
        program::{Node, Program, INPUTS},
        target::{self, Sample, Target},
        ImageConfig, ImageProblem,
    },
};
use std::sync::Arc;

fn image_problem(nodes: usize, samples: Vec<Sample>) -> Arc<dyn evolution::Problem> {
    Arc::new(
        ImageProblem::new(
            ImageConfig {
                nodes,
                ..Default::default()
            },
            samples,
        )
        .unwrap(),
    )
}
trait ImageEngine {
    fn program(&self, index: usize) -> anyhow::Result<Program>;
}
impl ImageEngine for Box<dyn evolution::Engine> {
    fn program(&self, index: usize) -> anyhow::Result<Program> {
        ImageProblem::decode(&self.genome(index)?)
    }
}

#[test]
fn wrapped_neighbourhood_is_correct_at_edges() {
    assert_eq!(evolution::neighbours(0, 4, 3), [0, 3, 1, 8, 4]);
    assert_eq!(evolution::neighbours(11, 4, 3), [11, 10, 8, 7, 3]);
}

#[test]
fn program_is_finite_bounded_and_roundtrips() {
    for cell in 0..100 {
        let p = Program::random(64, 123, cell);
        p.validate().unwrap();
        for x in [-1.0, -0.01, 0.0, 0.5, 1.0] {
            for y in [-1.0, 0.0, 1.0] {
                assert!(p
                    .evaluate(x, y)
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
            }
        }
        let serialized = serde_json::to_string(&p).unwrap();
        assert_eq!(serde_json::from_str::<Program>(&serialized).unwrap(), p);
    }
}

#[test]
fn dead_code_does_not_contribute_to_complexity_or_shader() {
    let p = Program {
        nodes: vec![
            Node {
                op: 3,
                a: 0,
                b: 1,
                parameter: 0.0,
            },
            Node {
                op: 0,
                a: INPUTS as u32,
                b: 0,
                parameter: 0.0,
            },
            Node {
                op: 0,
                a: 0,
                b: 0,
                parameter: -1.0,
            },
            Node {
                op: 0,
                a: 0,
                b: 0,
                parameter: 1.0,
            },
        ],
    };
    p.validate().unwrap();
    assert_eq!(p.active_mask().count_ones(), 3);
    assert!(!p.glsl().contains("float v6="));
    assert_eq!(p.evaluate(0.3, 0.4), [0.5, 0.0, 1.0]);
}

#[test]
fn invalid_programs_and_config_are_rejected() {
    let mut p = Program::random(12, 1, 0);
    p.nodes[0].a = 99;
    assert!(p.validate().is_err());
    let invalid = Config {
        width: 0,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());
    let invalid = Config {
        mutation_rate: f32::NAN,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());
    assert!(ImageProblem::new(ImageConfig::default(), vec![]).is_err());
}

#[test]
fn cpu_evolution_is_reproducible_and_each_cell_is_elitist() {
    let config = Config {
        width: 5,
        height: 4,
        mutation_rate: 0.3,
        ..Default::default()
    };
    let target = Target::preset("orbit", 32).unwrap();
    let samples = target.samples(8);
    let mut a = evolution::create("cpu", &config, image_problem(12, samples.clone())).unwrap();
    let mut b = evolution::create("cpu", &config, image_problem(12, samples)).unwrap();
    let mut previous = a.fitness().unwrap();
    for _ in 0..20 {
        a.step(1).unwrap();
        let next = a.fitness().unwrap();
        assert!(next.iter().zip(&previous).all(|(a, b)| a <= b));
        previous = next;
        for i in 0..config.population() {
            a.program(i).unwrap().validate().unwrap();
        }
    }
    b.step(20).unwrap();
    assert_eq!(a.fitness().unwrap(), b.fitness().unwrap());
    assert_eq!(
        a.program(a.stats().unwrap().best_index).unwrap(),
        b.program(b.stats().unwrap().best_index).unwrap()
    );
}

#[test]
fn constant_image_has_zero_unpenalized_error() {
    let p = Program {
        nodes: vec![
            Node {
                op: 0,
                a: 0,
                b: 0,
                parameter: 1.0
            };
            3
        ],
    };
    let target = Target {
        name: "white".into(),
        image: image::RgbImage::from_pixel(8, 8, image::Rgb([255, 255, 255])),
    };
    assert_eq!(target::mse(&p, &target.samples(8)), 0.0);
    assert_eq!(
        target::render(&p, 16, 16).get_pixel(7, 9).0,
        [255, 255, 255]
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an NVIDIA GPU and NVRTC; run cargo test -- --ignored"]
fn cuda_matches_reference_and_evolves_on_device() {
    let config = Config {
        width: 5,
        height: 4,
        mutation_rate: 0.2,
        ..Default::default()
    };
    let samples = Target::preset("orbit", 64).unwrap().samples(8);
    let cpu = evolution::create("cpu", &config, image_problem(24, samples.clone())).unwrap();
    let mut gpu = evolution::create("cuda", &config, image_problem(24, samples)).unwrap();
    for i in 0..config.population() {
        assert_eq!(
            cpu.program(i).unwrap(),
            gpu.program(i).unwrap(),
            "initial genome {i}"
        );
    }
    let reference = cpu.fitness().unwrap();
    let initial = gpu.fitness().unwrap();
    let renderer = revo2::problems::image::renderer::ImageRenderer::new(24).unwrap();
    for i in 0..config.population() {
        let expected = target::render(&cpu.program(i).unwrap(), 8, 8);
        let actual = renderer.render(&gpu.genome(i).unwrap(), 8).unwrap();
        assert!(
            expected
                .as_raw()
                .iter()
                .zip(actual.as_raw())
                .all(|(a, b)| a.abs_diff(*b) <= 1),
            "preview mismatch in cell {i}"
        );
    }
    for (i, (a, b)) in reference.iter().zip(&initial).enumerate() {
        assert!((a - b).abs() < 2e-5, "fitness {i}: CPU {a}, CUDA {b}");
    }
    let summary = gpu.stats().unwrap();
    let minimum = initial.iter().copied().fold(f32::INFINITY, f32::min);
    assert!((summary.best_score - minimum).abs() < 1e-6);
    assert_eq!(initial[summary.best_index], minimum);
    assert!((summary.mean_score - initial.iter().sum::<f32>() / initial.len() as f32).abs() < 1e-6);
    gpu.step(20).unwrap();
    assert_eq!(gpu.stats().unwrap().generation, 20);
    assert!(gpu
        .fitness()
        .unwrap()
        .iter()
        .zip(&initial)
        .all(|(a, b)| a <= b));
    for i in 0..config.population() {
        gpu.program(i).unwrap().validate().unwrap();
    }
    let mut repeated = evolution::create(
        "cuda",
        &config,
        image_problem(24, Target::preset("orbit", 64).unwrap().samples(8)),
    )
    .unwrap();
    repeated.step(20).unwrap();
    assert_eq!(gpu.fitness().unwrap(), repeated.fitness().unwrap());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an NVIDIA GPU and NVRTC"]
fn cuda_handles_minimum_and_maximum_program_sizes() {
    for (width, height, nodes) in [(2, 2, 3), (3, 7, 64)] {
        let config = Config {
            width,
            height,
            ..Default::default()
        };
        let samples = Target::preset("waves", 32).unwrap().samples(4);
        let cpu = evolution::create("cpu", &config, image_problem(nodes, samples.clone())).unwrap();
        let mut gpu = evolution::create("cuda", &config, image_problem(nodes, samples)).unwrap();
        for (a, b) in cpu.fitness().unwrap().iter().zip(gpu.fitness().unwrap()) {
            assert!((a - b).abs() < 2e-5, "CPU {a}, CUDA {b} with {nodes} nodes");
        }
        gpu.step(3).unwrap();
        for i in 0..config.population() {
            gpu.program(i).unwrap().validate().unwrap();
        }
    }
}

#[test]
fn uploaded_images_are_normalized_and_invalid_files_rejected() {
    let original = image::RgbImage::from_pixel(12, 6, image::Rgb([128, 64, 32]));
    let bytes = revo2::export::png(&original).unwrap();
    let target = Target::from_bytes("test image", &bytes).unwrap();
    assert_eq!(target.image.dimensions(), (512, 512));
    assert_eq!(target.image.get_pixel(256, 256).0, [128, 64, 32]);
    assert!(Target::from_bytes("invalid", b"not an image").is_err());
}
