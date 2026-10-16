use anyhow::Result;
use clap::{Parser, Subcommand};
use revo2::{
    demo::{Demo, DemoConfig, ProblemKind},
    evolution::{self, Config},
    export::{self, HistoryPoint, RunOutput},
    problems::{
        image::{target, Program, Target},
        packing::Rectangle,
    },
};
use std::{
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Parser)]
#[command(
    version,
    about = "Cellular evolution playground for image programs and rectangle packing"
)]
struct Args {
    #[arg(long,global=true,default_value="cuda",value_parser=["cuda","cpu"])]
    backend: String,
    #[arg(long,global=true,value_parser=["image","packing"])]
    problem: Option<String>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Open the local interactive playground (default).
    Serve {
        #[arg(long, default_value_t = 8787)]
        port: u16,
    },
    /// Evolve either problem without the UI and save a reproducible run.
    Run {
        /// Exact engine settings, e.g. a previous run's config.json.
        #[arg(long, conflicts_with_all=["population","width","height","nodes","seed","samples"])]
        config: Option<PathBuf>,
        #[arg(long, default_value_t = 1000)]
        generations: u32,
        #[arg(long, default_value_t = 32)]
        population: u32,
        /// Population width (X); defaults to 32. Use with --height for rectangles.
        #[arg(long, conflicts_with = "population")]
        width: Option<u32>,
        /// Population height (Y); defaults to 32.
        #[arg(long, conflicts_with = "population")]
        height: Option<u32>,
        #[arg(long, default_value_t = 24)]
        nodes: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        #[arg(long, default_value_t = 24)]
        samples: u32,
        #[arg(long, default_value_t = 100)]
        checkpoint: u32,
        #[arg(long)]
        preset: Option<String>,
        /// JSON array of width/height rectangles; selects packing.
        #[arg(long, conflicts_with = "target")]
        rectangles: Option<PathBuf>,
        #[arg(long)]
        target: Option<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Render a saved program without a GPU.
    Render {
        program: PathBuf,
        #[arg(long, default_value = "render.png")]
        output: PathBuf,
        #[arg(long, default_value_t = 512)]
        size: u32,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();
    let kind = args.problem.as_deref().map(|s| {
        if s == "packing" {
            ProblemKind::Packing
        } else {
            ProblemKind::Image
        }
    });
    match args.command.unwrap_or(Command::Serve { port: 8787 }) {
        Command::Serve { port } => {
            revo2::server::serve(port, &args.backend, kind.unwrap_or_default())
        }
        Command::Render {
            program,
            output,
            size,
        } => {
            anyhow::ensure!((1..=4096).contains(&size), "render size must be 1–4096");
            let value: serde_json::Value = serde_json::from_slice(&std::fs::read(program)?)?;
            if value.get("problem").and_then(|v| v.as_str()) == Some("packing") {
                let rectangles: Vec<Rectangle> =
                    serde_json::from_value(value["rectangles"].clone())?;
                let genome: Vec<u32> = serde_json::from_value(value["genome"].clone())?;
                Demo::from_rectangles(rectangles)?
                    .render(&genome, size)?
                    .save(&output)?;
            } else {
                let p: Program = serde_json::from_value(value)?;
                p.validate()?;
                target::render(&p, size, size).save(&output)?;
            }
            println!("Saved {}", output.display());
            Ok(())
        }
        Command::Run {
            config: config_path,
            rectangles,
            generations,
            population,
            width,
            height,
            nodes,
            seed,
            samples,
            checkpoint,
            preset,
            target,
            output,
        } => {
            anyhow::ensure!(checkpoint > 0, "checkpoint must be positive");
            let mut config: DemoConfig = if let Some(path) = config_path {
                serde_json::from_slice(&std::fs::read(path)?)?
            } else {
                DemoConfig {
                    evolution: Config {
                        width: width.unwrap_or(population),
                        height: height.unwrap_or(population),
                        seed,
                        ..Default::default()
                    },
                    image: revo2::problems::image::ImageConfig {
                        nodes,
                        sample_resolution: samples,
                        ..Default::default()
                    },
                    ..Default::default()
                }
            };
            if let Some(kind) = kind {
                config.problem = kind;
            }
            if rectangles.is_some() {
                config.problem = ProblemKind::Packing;
            }
            config.validate()?;
            let demo = if let Some(path) = rectangles {
                Demo::from_rectangles(serde_json::from_slice(&std::fs::read(path)?)?)?
            } else if let Some(path) = target {
                Demo::from_target(&config, Target::load(&path)?)?
            } else {
                Demo::new(&config, preset.as_deref())?
            };
            let output = output.unwrap_or_else(|| {
                PathBuf::from(format!(
                    "runs/{}",
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                ))
            });
            anyhow::ensure!(
                !output.exists() || std::fs::read_dir(&output)?.next().is_none(),
                "output directory must be empty: {}",
                output.display()
            );
            let mut engine = evolution::create(&args.backend, &config.evolution, demo.problem())?;
            println!(
                "{} | {} cells | {} | {}",
                engine.backend(),
                config.population(),
                demo.problem().name(),
                demo.name()
            );
            let start = Instant::now();
            let mut history = Vec::new();
            let mut stats = engine.stats()?;
            loop {
                let seconds = start.elapsed().as_secs_f64();
                println!(
                    "generation {:>6} | best {:.7} | mean {:.7} | {:.2}s",
                    stats.generation, stats.best_score, stats.mean_score, seconds
                );
                history.push(HistoryPoint {
                    generation: stats.generation,
                    best: stats.best_score,
                    mean: stats.mean_score,
                    seconds,
                });
                if stats.generation >= generations {
                    break;
                }
                engine.step(checkpoint.min(generations - stats.generation))?;
                stats = engine.stats()?;
            }
            let seconds = start.elapsed().as_secs_f64();
            let genome = engine.genome(stats.best_index)?;
            export::save_run(
                &output,
                RunOutput {
                    config: &config,
                    demo: &demo,
                    genome: &genome,
                    stats: &stats,
                    history: &history,
                    backend: engine.backend(),
                    seconds,
                },
            )?;
            println!(
                "Saved solution, previews, history, and run metadata to {}",
                output.display()
            );
            Ok(())
        }
    }
}
