use crate::{
    demo::{Demo, DemoConfig},
    evolution::Stats,
};
use anyhow::{Context, Result};
use serde::Serialize;
use std::{io::Cursor, path::Path};

#[derive(Clone, Serialize)]
pub struct HistoryPoint {
    pub generation: u32,
    pub best: f32,
    pub mean: f32,
    pub seconds: f64,
}

pub fn png(image: &image::RgbImage) -> Result<Vec<u8>> {
    let mut bytes = Cursor::new(Vec::new());
    image.write_to(&mut bytes, image::ImageFormat::Png)?;
    Ok(bytes.into_inner())
}

pub struct RunOutput<'a> {
    pub config: &'a DemoConfig,
    pub demo: &'a Demo,
    pub genome: &'a [u32],
    pub stats: &'a Stats,
    pub history: &'a [HistoryPoint],
    pub backend: &'a str,
    pub seconds: f64,
}

pub fn save_run(directory: &Path, run: RunOutput<'_>) -> Result<()> {
    anyhow::ensure!(
        !directory.exists() || std::fs::read_dir(directory)?.next().is_none(),
        "output directory is not empty: {}",
        directory.display()
    );
    std::fs::create_dir_all(directory)?;
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(run.config)?,
    )?;
    let metrics = run.demo.export(directory, run.genome)?;
    let metadata = serde_json::json!({"version":env!("CARGO_PKG_VERSION"),"backend":run.backend,"config":run.config,"problem":run.config.problem,"target":run.demo.name(),"stats":run.stats,"evolution_seconds":run.seconds,"metrics":metrics,"note":"Evolution time excludes setup, export and rendering; includes progress observations."});
    std::fs::write(
        directory.join("run.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    let mut csv = String::from("generation,best_score,mean_score,elapsed_seconds\n");
    for p in run.history {
        csv.push_str(&format!(
            "{},{},{},{}\n",
            p.generation, p.best, p.mean, p.seconds
        ));
    }
    std::fs::write(directory.join("history.csv"), csv).context("writing fitness history")?;
    Ok(())
}
