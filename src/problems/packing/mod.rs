use crate::evolution::{rng::Rng, Config, CudaProblem, Problem};
use anyhow::Result;
use image::{Rgb, RgbImage};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Rectangle {
    pub width: u32,
    pub height: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct PackingConfig {
    pub rectangle_count: usize,
    pub rectangle_seed: u64,
}
impl Default for PackingConfig {
    fn default() -> Self {
        Self {
            rectangle_count: 24,
            rectangle_seed: 17,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Placement {
    pub id: usize,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub rotated: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Layout {
    pub placements: Vec<Placement>,
    pub width: u32,
    pub height: u32,
    pub area: u32,
    pub rectangle_area: u32,
    pub utilization: f32,
}
pub struct PackingProblem {
    rectangles: Vec<Rectangle>,
    min_width: u32,
    max_width: u32,
    initial_max: u32,
    total_area: u32,
}
const COLORS: [[u8; 3]; 12] = [
    [237, 182, 108],
    [129, 178, 164],
    [170, 186, 129],
    [191, 146, 116],
    [139, 163, 191],
    [206, 190, 143],
    [168, 141, 171],
    [122, 184, 181],
    [199, 157, 134],
    [165, 187, 163],
    [203, 172, 108],
    [153, 163, 133],
];

impl PackingProblem {
    pub fn rectangles(&self) -> &[Rectangle] {
        &self.rectangles
    }

    pub fn new(rectangles: Vec<Rectangle>) -> Result<Self> {
        anyhow::ensure!(
            !rectangles.is_empty() && rectangles.len() <= 64,
            "packing needs 1–64 rectangles"
        );
        anyhow::ensure!(
            rectangles
                .iter()
                .all(|r| (1..=512).contains(&r.width) && (1..=512).contains(&r.height)),
            "rectangle dimensions must be 1–512"
        );
        let min_width = rectangles
            .iter()
            .map(|r| r.width.max(r.height))
            .max()
            .unwrap();
        let max_width = rectangles.iter().map(|r| r.width.max(r.height)).sum();
        let total_area = rectangles.iter().map(|r| r.width * r.height).sum();
        let initial_max = ((total_area as f32).sqrt() * 2.0).ceil() as u32;
        Ok(Self {
            rectangles,
            min_width,
            max_width,
            initial_max: initial_max.max(min_width).min(max_width),
            total_area,
        })
    }
    pub fn preset(name: &str, config: &PackingConfig) -> Result<Self> {
        anyhow::ensure!(
            (2..=64).contains(&config.rectangle_count),
            "rectangle count must be 2–64"
        );
        let mut rng = Rng::new(config.rectangle_seed, 0, 0);
        let mut rectangles = Vec::new();
        for _ in 0..config.rectangle_count {
            let r = match name {
                "mixed" => Rectangle {
                    width: 16 + rng.next_u32() % 60,
                    height: 16 + rng.next_u32() % 60,
                },
                "tiles" => Rectangle {
                    width: 16 * (1 + rng.next_u32() % 4),
                    height: 16 * (1 + rng.next_u32() % 4),
                },
                "bars" => Rectangle {
                    width: 40 + rng.next_u32() % 80,
                    height: 8 + rng.next_u32() % 20,
                },
                _ => anyhow::bail!("unknown packing preset '{name}'; choose mixed, tiles, or bars"),
            };
            rectangles.push(r);
        }
        Self::new(rectangles)
    }
    pub fn validate_genome(&self, g: &[u32]) -> Result<()> {
        let n = self.rectangles.len();
        anyhow::ensure!(g.len() == 2 * n + 1, "incorrect packing genome length");
        let mut seen = vec![false; n];
        for &id in &g[..n] {
            anyhow::ensure!(
                (id as usize) < n && !seen[id as usize],
                "packing order must be a permutation"
            );
            seen[id as usize] = true;
        }
        anyhow::ensure!(g[n..2 * n].iter().all(|&x| x <= 1), "invalid rotation flag");
        anyhow::ensure!(
            (self.min_width..=self.max_width).contains(&g[2 * n]),
            "invalid decoder width"
        );
        Ok(())
    }
    pub fn decode(&self, g: &[u32]) -> Result<Layout> {
        self.validate_genome(g)?;
        let n = self.rectangles.len();
        let limit = g[2 * n];
        let mut placed: Vec<Placement> = Vec::with_capacity(n);
        let mut width = 0;
        let mut height = 0;
        for &id in &g[..n] {
            let id = id as usize;
            let r = &self.rectangles[id];
            let rotated = g[n + id] != 0;
            let (w, h) = if rotated {
                (r.height, r.width)
            } else {
                (r.width, r.height)
            };
            let mut bx = 0;
            let mut by = u32::MAX;
            for candidate in 0..=placed.len() {
                let x = if candidate == 0 {
                    0
                } else {
                    let p = &placed[candidate - 1];
                    p.x + p.width
                };
                if x + w > limit {
                    continue;
                }
                let mut y = 0;
                for p in &placed {
                    if x < p.x + p.width && p.x < x + w {
                        y = y.max(p.y + p.height);
                    }
                }
                if y < by || (y == by && x < bx) {
                    bx = x;
                    by = y;
                }
            }
            width = width.max(bx + w);
            height = height.max(by + h);
            placed.push(Placement {
                id,
                x: bx,
                y: by,
                width: w,
                height: h,
                rotated,
            });
        }
        let area = width * height;
        Ok(Layout {
            placements: placed,
            width,
            height,
            area,
            rectangle_area: self.total_area,
            utilization: self.total_area as f32 / area as f32,
        })
    }
    pub fn render(&self, g: &[u32], size: u32) -> Result<RgbImage> {
        Ok(self.decode(g)?.render(size))
    }
    pub fn reference(&self, size: u32) -> RgbImage {
        let mut placements = Vec::new();
        let mut x = 0;
        let mut y = 0;
        let mut row_height = 0;
        let mut width = 0;
        let cols = (self.rectangles.len() as f32).sqrt().ceil() as usize;
        for (id, r) in self.rectangles.iter().enumerate() {
            if id > 0 && id % cols == 0 {
                y += row_height + 12;
                x = 0;
                row_height = 0;
            }
            placements.push(Placement {
                id,
                x,
                y,
                width: r.width,
                height: r.height,
                rotated: false,
            });
            x += r.width + 12;
            row_height = row_height.max(r.height);
            width = width.max(x);
        }
        Layout {
            placements,
            width,
            height: y + row_height,
            area: 0,
            rectangle_area: self.total_area,
            utilization: 0.0,
        }
        .render(size)
    }
}
impl Layout {
    pub fn render(&self, size: u32) -> RgbImage {
        let mut image = RgbImage::from_pixel(size, size, Rgb([17, 21, 19]));
        let pad = 16.0;
        let scale = ((size as f32 - 2.0 * pad) / self.width.max(self.height) as f32).max(0.001);
        let ox = (size as f32 - self.width as f32 * scale) * 0.5;
        let oy = (size as f32 - self.height as f32 * scale) * 0.5;
        for p in &self.placements {
            let x0 = (ox + p.x as f32 * scale).round() as u32;
            let y0 = (oy + p.y as f32 * scale).round() as u32;
            let x1 = (ox + (p.x + p.width) as f32 * scale).round() as u32;
            let y1 = (oy + (p.y + p.height) as f32 * scale).round() as u32;
            for y in y0..y1.min(size) {
                for x in x0..x1.min(size) {
                    let edge = x == x0 || y == y0 || x + 1 == x1 || y + 1 == y1;
                    image.put_pixel(
                        x,
                        y,
                        Rgb(if edge {
                            [28, 33, 28]
                        } else {
                            COLORS[p.id % COLORS.len()]
                        }),
                    );
                }
            }
        }
        image
    }
    pub fn svg(&self) -> String {
        let mut svg=format!("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"-4 -4 {} {}\"><rect x=\"-4\" y=\"-4\" width=\"100%\" height=\"100%\" fill=\"#111513\"/>",self.width+8,self.height+8);
        for p in &self.placements {
            let c = COLORS[p.id % COLORS.len()];
            svg.push_str(&format!("<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"rgb({},{},{})\" stroke=\"#1c211c\" stroke-width=\"0.6\"/><text x=\"{}\" y=\"{}\" font-size=\"5\" font-family=\"sans-serif\" fill=\"#263025\" text-anchor=\"middle\" dominant-baseline=\"middle\">{}</text>",p.x,p.y,p.width,p.height,c[0],c[1],c[2],p.x as f32+p.width as f32/2.0,p.y as f32+p.height as f32/2.0,p.id+1));
        }
        svg.push_str("</svg>");
        svg
    }
}
impl Problem for PackingProblem {
    fn name(&self) -> &str {
        "packing"
    }
    fn genome_len(&self) -> usize {
        self.rectangles.len() * 2 + 1
    }
    fn validate(&self) -> Result<()> {
        Self::new(self.rectangles.clone()).map(|_| ())
    }
    fn initialize(&self, g: &mut [u32], rng: &mut Rng) {
        let n = self.rectangles.len();
        for (i, x) in g[..n].iter_mut().enumerate() {
            *x = i as u32;
        }
        for i in (1..n).rev() {
            let j = rng.index(i + 1);
            g.swap(i, j);
        }
        for x in &mut g[n..2 * n] {
            *x = rng.next_u32() % 2;
        }
        g[2 * n] = self.min_width + rng.next_u32() % (self.initial_max - self.min_width + 1);
    }
    fn recombine(&self, a: &[u32], b: &[u32], out: &mut [u32], rng: &mut Rng, cross: bool) {
        out.copy_from_slice(a);
        if !cross {
            return;
        }
        let n = self.rectangles.len();
        let split = rng.index(n + 1);
        let mut end = split;
        for &id in &b[..n] {
            if !a[..split].contains(&id) {
                out[end] = id;
                end += 1;
            }
        }
        for i in n..=2 * n {
            if rng.next_u32().is_multiple_of(2) {
                out[i] = b[i];
            }
        }
    }
    fn mutate(&self, g: &mut [u32], rng: &mut Rng, c: &Config) {
        let n = self.rectangles.len();
        for i in 0..n {
            if rng.unit() < c.mutation_rate {
                let j = rng.index(n);
                g.swap(i, j);
            }
            if rng.unit() < c.mutation_rate {
                g[n + i] ^= 1;
            }
        }
        if rng.unit() < c.mutation_rate {
            let delta =
                ((rng.unit() * 2.0 - 1.0) * c.mutation_scale * self.initial_max as f32) as i32;
            g[2 * n] = (g[2 * n] as i32 + delta).clamp(self.min_width as i32, self.max_width as i32)
                as u32;
        }
    }
    fn score(&self, g: &[u32]) -> f32 {
        let layout = self.decode(g).expect("valid packing genome");
        1.0 - layout.utilization
    }
    fn cuda(&self) -> Option<CudaProblem> {
        let mut data = vec![
            self.min_width,
            self.max_width,
            self.initial_max,
            self.total_area,
        ];
        for r in &self.rectangles {
            data.extend([r.width, r.height]);
        }
        // SAFETY: The validated rectangle count/encoding matches these callbacks.
        Some(unsafe {
            CudaProblem::new(
                format!(
                    "#define RECTANGLES {}\n{}",
                    self.rectangles.len(),
                    include_str!("cuda.cu")
                ),
                data,
                1,
            )
        })
    }
}
