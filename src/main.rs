use anyhow::Result;
use clap::{Parser, Subcommand};
use revo2::{
    evolution::{self, Config},
    export::{self, HistoryPoint, RunOutput},
    program::Program,
    target::{self, Target},
};
use std::{
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Parser)]
#[command(
    version,
    about = "Cellular evolution of image programs, powered by CUDA"
)]
struct Args {
    #[arg(long,global=true,default_value="cuda",value_parser=["cuda","cpu"])]
    backend: String,
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
    /// Evolve without the UI and save a reproducible run.
    Run {
        /// Exact engine settings, e.g. a previous run's config.json.
        #[arg(long, conflicts_with_all=["population","nodes","seed","samples"])]
        config: Option<PathBuf>,
        #[arg(long, default_value_t = 1000)]
        generations: u32,
        #[arg(long, default_value_t = 32)]
        population: u32,
        #[arg(long, default_value_t = 24)]
        nodes: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        #[arg(long, default_value_t = 24)]
        samples: u32,
        #[arg(long, default_value_t = 100)]
        checkpoint: u32,
        #[arg(long, default_value = "orbit")]
        preset: String,
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
    match args.command.unwrap_or(Command::Serve { port: 8787 }) {
        Command::Serve { port } => revo2::server::serve(port, &args.backend),
        Command::Render {
            program,
            output,
            size,
        } => {
            anyhow::ensure!((1..=4096).contains(&size), "render size must be 1–4096");
            let p: Program = serde_json::from_slice(&std::fs::read(program)?)?;
            p.validate()?;
            target::render(&p, size, size).save(&output)?;
            println!("Saved {}", output.display());
            Ok(())
        }
        Command::Run {
            config: config_path,
            generations,
            population,
            nodes,
            seed,
            samples,
            checkpoint,
            preset,
            target,
            output,
        } => {
            anyhow::ensure!(checkpoint > 0, "checkpoint must be positive");
            let config = if let Some(path) = config_path {
                serde_json::from_slice(&std::fs::read(path)?)?
            } else {
                Config {
                    width: population,
                    height: population,
                    nodes,
                    seed,
                    sample_resolution: samples,
                    ..Default::default()
                }
            };
            config.validate()?;
            let target = match target {
                Some(path) => Target::load(&path)?,
                None => Target::preset(&preset, 512)?,
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
            let mut engine = evolution::create(
                &args.backend,
                &config,
                target.samples(config.sample_resolution),
            )?;
            println!(
                "{} | {} cells | {} nodes | {} samples | target {}",
                engine.backend(),
                config.population(),
                config.nodes,
                config.sample_resolution * config.sample_resolution,
                target.name
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
            let program = engine.program(stats.best_index)?;
            export::save_run(
                &output,
                RunOutput {
                    config: &config,
                    target: &target,
                    program: &program,
                    stats: &stats,
                    history: &history,
                    backend: engine.backend(),
                    seconds,
                },
            )?;
            println!(
                "Saved images, program, GLSL, history, and run metadata to {}",
                output.display()
            );
            Ok(())
        }
    }
}
