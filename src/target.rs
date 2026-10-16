use crate::program::Program;
use anyhow::{bail, Context, Result};
use image::{imageops::FilterType, Rgb, RgbImage};
use std::path::Path;

#[derive(Clone)]
pub struct Target {
    pub name: String,
    pub image: RgbImage,
}

/// ABI is five tightly packed floats: coordinates followed by target RGB.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Sample {
    pub x: f32,
    pub y: f32,
    pub color: [f32; 3],
}

#[cfg(feature = "cuda")]
unsafe impl cudarc::driver::DeviceRepr for Sample {}

pub fn coordinate(i: u32, size: u32) -> f32 {
    2.0 * (i as f32 + 0.5) / size as f32 - 1.0
}

impl Target {
    pub fn preset(name: &str, size: u32) -> Result<Self> {
        let mut image = RgbImage::new(size, size);
        for (px, py, p) in image.enumerate_pixels_mut() {
            let x = coordinate(px, size);
            let y = coordinate(py, size);
            let r = (x * x + y * y).sqrt();
            let c = match name {
                "rings" => {
                    let wave = (14.0 * r).cos() * 0.5 + 0.5;
                    [
                        wave,
                        (wave * 0.75 + 0.15) * (1.0 - y * 0.3),
                        0.25 + 0.55 * (1.0 - wave),
                    ]
                }
                "sunset" => {
                    if (x * x + (y + 0.1) * (y + 0.1)).sqrt() < 0.38 {
                        [1.0, 0.72, 0.22]
                    } else if y > 0.35 + 0.1 * (8.0 * x).sin() {
                        [0.05, 0.12, 0.22]
                    } else {
                        [0.7 - y * 0.2, 0.25 - y * 0.18, 0.4 - y * 0.1]
                    }
                }
                "waves" => [
                    (6.0 * x + 3.0 * y).sin() * 0.5 + 0.5,
                    (4.0 * y - 2.0 * x).cos() * 0.5 + 0.5,
                    (5.0 * r).sin() * 0.5 + 0.5,
                ],
                "orbit" => {
                    let disc = r < 0.53;
                    let band = (y + 0.32 * x).abs() < 0.1 && r < 0.86;
                    if band {
                        [0.95, 0.68, 0.3]
                    } else if disc {
                        [0.3 + x * 0.15, 0.6 - y * 0.18, 0.78 - y * 0.1]
                    } else {
                        [0.035, 0.05, 0.10]
                    }
                }
                _ => bail!("unknown preset '{name}'; choose rings, sunset, waves, or orbit"),
            };
            *p = Rgb(c.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8));
        }
        Ok(Self {
            name: name.into(),
            image,
        })
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_bytes(
            path.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .as_ref(),
            &bytes,
        )
    }

    pub fn from_bytes(name: &str, bytes: &[u8]) -> Result<Self> {
        // Decode with explicit limits before allocating user-supplied dimensions.
        let mut reader =
            image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(8192);
        limits.max_image_height = Some(8192);
        limits.max_alloc = Some(256 * 1024 * 1024);
        reader.limits(limits);
        let decoded = reader
            .decode()
            .context("target must be a PNG or JPEG image")?;
        let resized = decoded
            .resize_to_fill(512, 512, FilterType::Triangle)
            .to_rgba8();
        // Transparent pixels are composited over the playground's dark background.
        let image = RgbImage::from_fn(resized.width(), resized.height(), |x, y| {
            let p = resized.get_pixel(x, y);
            let a = p[3] as f32 / 255.0;
            Rgb([0, 1, 2].map(|i| (p[i] as f32 * a + 15.0 * (1.0 - a)).round() as u8))
        });
        Ok(Self {
            name: name.chars().take(80).collect(),
            image,
        })
    }

    pub fn samples(&self, resolution: u32) -> Vec<Sample> {
        // A fixed pixel grid makes parent/offspring scores comparable across generations.
        let small =
            image::imageops::resize(&self.image, resolution, resolution, FilterType::Triangle);
        small
            .enumerate_pixels()
            .map(|(x, y, p)| Sample {
                x: coordinate(x, resolution),
                y: coordinate(y, resolution),
                color: p.0.map(|v| v as f32 / 255.0),
            })
            .collect()
    }
}

pub fn render(program: &Program, width: u32, height: u32) -> RgbImage {
    RgbImage::from_fn(width, height, |x, y| {
        Rgb(program
            .evaluate(coordinate(x, width), coordinate(y, height))
            .map(|v| (v * 255.0).round() as u8))
    })
}

pub fn mse(program: &Program, samples: &[Sample]) -> f32 {
    samples
        .iter()
        .map(|s| {
            program
                .evaluate(s.x, s.y)
                .iter()
                .zip(s.color)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
        })
        .sum::<f32>()
        / (3 * samples.len()) as f32
}
