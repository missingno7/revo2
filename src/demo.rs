//! Application adapters. The reusable engine has no rendering or demo types.
use crate::{
    evolution::{Config, Problem},
    problems::{
        image::{target, ImageConfig, ImageProblem, Program, Target},
        packing::{Layout, PackingConfig, PackingProblem, Rectangle},
    },
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::Arc};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProblemKind {
    #[default]
    Image,
    Packing,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DemoConfig {
    pub problem: ProblemKind,
    #[serde(flatten)]
    pub evolution: Config,
    #[serde(flatten)]
    pub image: ImageConfig,
    #[serde(flatten)]
    pub packing: PackingConfig,
}
impl DemoConfig {
    pub fn validate(&self) -> Result<()> {
        self.evolution.validate()?;
        if self.problem == ProblemKind::Image {
            self.image.validate()?;
        }
        Ok(())
    }
    pub fn population(&self) -> usize {
        self.evolution.population()
    }
}
pub enum Demo {
    Image {
        problem: Arc<ImageProblem>,
        target: Target,
    },
    Packing {
        problem: Arc<PackingProblem>,
        preset: String,
    },
}
pub enum Preview {
    Host,
    #[cfg(feature = "cuda")]
    ImageCuda(crate::problems::image::renderer::ImageRenderer),
}
impl Preview {
    pub fn new(demo: &Demo, backend: &str) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if backend == "cuda" {
            if let Demo::Image { problem, .. } = demo {
                return Ok(Self::ImageCuda(
                    crate::problems::image::renderer::ImageRenderer::new(problem.config.nodes)?,
                ));
            }
        }
        let _ = (demo, backend);
        Ok(Self::Host)
    }
    pub fn render(&self, demo: &Demo, genome: &[u32], size: u32) -> Result<image::RgbImage> {
        match self {
            Self::Host => demo.render(genome, size),
            #[cfg(feature = "cuda")]
            Self::ImageCuda(renderer) => renderer.render(genome, size),
        }
    }
}
#[derive(Serialize)]
pub struct Inspection {
    pub program: Option<Program>,
    pub active_nodes: Vec<usize>,
    pub layout: Option<Layout>,
    pub best_layout: Option<Layout>,
    pub validation_mse: Option<f32>,
}
impl Demo {
    pub fn new(c: &DemoConfig, preset: Option<&str>) -> Result<Self> {
        c.validate()?;
        match c.problem {
            ProblemKind::Image => {
                Self::from_target(c, Target::preset(preset.unwrap_or("orbit"), 512)?)
            }
            ProblemKind::Packing => {
                let preset = preset.unwrap_or("mixed");
                Ok(Self::Packing {
                    problem: Arc::new(PackingProblem::preset(preset, &c.packing)?),
                    preset: preset.into(),
                })
            }
        }
    }
    pub fn from_target(c: &DemoConfig, target: Target) -> Result<Self> {
        anyhow::ensure!(
            c.problem == ProblemKind::Image,
            "target upload is only available for image programs"
        );
        Ok(Self::Image {
            problem: Arc::new(ImageProblem::from_target(c.image.clone(), &target)?),
            target,
        })
    }
    pub fn from_rectangles(rectangles: Vec<Rectangle>) -> Result<Self> {
        Ok(Self::Packing {
            problem: Arc::new(PackingProblem::new(rectangles)?),
            preset: "custom".into(),
        })
    }
    pub fn reconfigure(&self, c: &DemoConfig, preset: Option<&str>) -> Result<Self> {
        if c.problem != self.kind() {
            return Self::new(c, preset);
        }
        match self {
            Self::Image { target, .. } => {
                let target = match preset {
                    Some(name) => Target::preset(name, 512)?,
                    None => target.clone(),
                };
                Self::from_target(c, target)
            }
            Self::Packing {
                problem,
                preset: current,
            } => match preset {
                Some("custom") => Self::from_rectangles(problem.rectangles().to_vec()),
                None if current == "custom" => Self::from_rectangles(problem.rectangles().to_vec()),
                None => Self::new(c, Some(current)),
                Some(name) => Self::new(c, Some(name)),
            },
        }
    }
    pub fn kind(&self) -> ProblemKind {
        match self {
            Self::Image { .. } => ProblemKind::Image,
            Self::Packing { .. } => ProblemKind::Packing,
        }
    }
    pub fn problem(&self) -> Arc<dyn Problem> {
        match self {
            Self::Image { problem, .. } => problem.clone(),
            Self::Packing { problem, .. } => problem.clone(),
        }
    }
    pub fn name(&self) -> String {
        match self {
            Self::Image { target, .. } => target.name.clone(),
            Self::Packing { problem, preset } => {
                format!("{preset} · {} rectangles", problem.rectangles().len())
            }
        }
    }
    pub fn preset(&self) -> &str {
        match self {
            Self::Image { target, .. } => &target.name,
            Self::Packing { preset, .. } => preset,
        }
    }
    pub fn reference(&self) -> image::RgbImage {
        match self {
            Self::Image { target, .. } => target.image.clone(),
            Self::Packing { problem, .. } => problem.reference(512),
        }
    }
    pub fn render(&self, g: &[u32], size: u32) -> Result<image::RgbImage> {
        anyhow::ensure!((32..=4096).contains(&size), "render size must be 32–4096");
        match self {
            Self::Image { .. } => Ok(target::render(&ImageProblem::decode(g)?, size, size)),
            Self::Packing { problem, .. } => problem.render(g, size),
        }
    }
    pub fn inspect(&self, selected: &[u32], best: &[u32]) -> Result<Inspection> {
        match self {
            Self::Image { target, .. } => {
                let p = ImageProblem::decode(selected)?;
                let mask = p.active_mask();
                let active_nodes = (0..p.nodes.len())
                    .filter(|i| mask & (1u64 << i) != 0)
                    .collect();
                let validation = target::mse(&ImageProblem::decode(best)?, &target.samples(64));
                Ok(Inspection {
                    program: Some(p),
                    active_nodes,
                    layout: None,
                    best_layout: None,
                    validation_mse: Some(validation),
                })
            }
            Self::Packing { problem, .. } => Ok(Inspection {
                program: None,
                active_nodes: vec![],
                layout: Some(problem.decode(selected)?),
                best_layout: Some(problem.decode(best)?),
                validation_mse: None,
            }),
        }
    }
    pub fn solution(&self, g: &[u32]) -> Result<serde_json::Value> {
        match self {
            Self::Image { .. } => Ok(serde_json::to_value(ImageProblem::decode(g)?)?),
            Self::Packing { problem, .. } => Ok(
                serde_json::json!({"problem":"packing","rectangles":problem.rectangles(),"genome":g,"layout":problem.decode(g)?}),
            ),
        }
    }
    pub fn export(&self, dir: &Path, g: &[u32]) -> Result<serde_json::Value> {
        let solution = self.solution(g)?;
        std::fs::write(
            dir.join("solution.json"),
            serde_json::to_vec_pretty(&solution)?,
        )?;
        self.render(g, 512)?.save(dir.join("best.png"))?;
        match self {
            Self::Image { target, .. } => {
                let p = ImageProblem::decode(g)?;
                target.image.save(dir.join("target.png"))?;
                std::fs::write(dir.join("program.json"), serde_json::to_vec_pretty(&p)?)?;
                std::fs::write(dir.join("program.glsl"), p.glsl())?;
                Ok(
                    serde_json::json!({"validation_mse_128":target::mse(&p,&target.samples(128)),"active_nodes":p.active_mask().count_ones()}),
                )
            }
            Self::Packing { problem, .. } => {
                let layout = problem.decode(g)?;
                std::fs::write(
                    dir.join("rectangles.json"),
                    serde_json::to_vec_pretty(problem.rectangles())?,
                )?;
                std::fs::write(dir.join("layout.svg"), layout.svg())?;
                problem.reference(512).save(dir.join("rectangles.png"))?;
                Ok(
                    serde_json::json!({"bounding_area":layout.area,"utilization":layout.utilization,"rectangle_area":layout.rectangle_area}),
                )
            }
        }
    }
}
