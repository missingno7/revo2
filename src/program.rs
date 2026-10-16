use serde::{Deserialize, Serialize};

pub const INPUTS: usize = 6;
pub const OPS: u32 = 12;
pub const OP_NAMES: [&str; 12] = [
    "constant", "add", "subtract", "multiply", "divide", "sin", "cos", "abs", "min", "max", "tanh",
    "step",
];

/// This layout is shared with kernels/image.cu. Connections only point backwards.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct Node {
    pub op: u32,
    pub a: u32,
    pub b: u32,
    pub parameter: f32,
}

#[cfg(feature = "cuda")]
unsafe impl cudarc::driver::DeviceRepr for Node {}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub nodes: Vec<Node>,
}

/// Integer-only RNG also implemented in CUDA, independent of scheduling.
pub(crate) struct Rng(pub u32);

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

    pub fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    pub fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 * (1.0 / 16777216.0)
    }

    pub fn index(&mut self, n: usize) -> usize {
        (self.next() % n as u32) as usize
    }

    pub fn node(&mut self, index: usize) -> Node {
        Node {
            op: self.next() % OPS,
            a: self.index(INPUTS + index) as u32,
            b: self.index(INPUTS + index) as u32,
            parameter: self.unit() * 8.0 - 4.0,
        }
    }
}

pub fn arity(op: u32) -> usize {
    match op {
        0 => 0,
        5 | 6 | 7 | 10 => 1,
        _ => 2,
    }
}

impl Program {
    pub fn random(nodes: usize, seed: u64, cell: u32) -> Self {
        let mut rng = Rng::new(seed, cell, 0);
        Self {
            nodes: (0..nodes).map(|i| rng.node(i)).collect(),
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            (3..=64).contains(&self.nodes.len()),
            "program needs 3–64 nodes"
        );
        for (i, n) in self.nodes.iter().enumerate() {
            anyhow::ensure!(
                n.op < OPS && (n.a as usize) < INPUTS + i && (n.b as usize) < INPUTS + i,
                "invalid operation or forward connection at node {i}"
            );
            anyhow::ensure!(
                n.parameter.is_finite() && n.parameter.abs() <= 4.0,
                "invalid constant at node {i}"
            );
        }
        Ok(())
    }

    pub fn active_mask(&self) -> u64 {
        let len = self.nodes.len();
        let mut active = 7u64 << (len - 3);
        for i in (0..len).rev() {
            if active & (1u64 << i) == 0 {
                continue;
            }
            let n = self.nodes[i];
            for input in [
                (arity(n.op) >= 1).then_some(n.a),
                (arity(n.op) == 2).then_some(n.b),
            ]
            .into_iter()
            .flatten()
            {
                if input >= INPUTS as u32 {
                    active |= 1u64 << (input as usize - INPUTS);
                }
            }
        }
        active
    }

    pub fn evaluate(&self, x: f32, y: f32) -> [f32; 3] {
        let mut registers = [0.0f32; INPUTS + 64];
        registers[..INPUTS].copy_from_slice(&[
            x,
            y,
            (x * x + y * y).sqrt(),
            (std::f32::consts::PI * x).sin(),
            (std::f32::consts::PI * y).sin(),
            1.0,
        ]);
        for (i, n) in self.nodes.iter().enumerate() {
            let a = registers[n.a as usize];
            let b = registers[n.b as usize];
            let v = match n.op {
                0 => n.parameter,
                1 => a + b,
                2 => a - b,
                3 => a * b,
                4 => {
                    if b.abs() < 0.01 {
                        a
                    } else {
                        a / b
                    }
                }
                5 => (a * n.parameter).sin(),
                6 => (a * n.parameter).cos(),
                7 => a.abs(),
                8 => a.min(b),
                9 => a.max(b),
                10 => a.tanh(),
                // A small dead band keeps near-equal transcendental results
                // from flipping discontinuous branches between CPU and CUDA.
                11 => {
                    if a > b + 0.0001 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                _ => unreachable!("validated opcode"),
            };
            registers[INPUTS + i] = v.clamp(-8.0, 8.0);
        }
        let end = INPUTS + self.nodes.len();
        [registers[end - 3], registers[end - 2], registers[end - 1]]
            .map(|v| (v * 0.5 + 0.5).clamp(0.0, 1.0))
    }

    pub fn glsl(&self) -> String {
        let mut s = String::from("// Generated by Revo 2. Input p is in [-1, 1].\nvec3 revo(vec2 p) {\n  float v0=p.x, v1=p.y, v2=length(p);\n  float v3=sin(3.14159265*p.x), v4=sin(3.14159265*p.y), v5=1.0;\n");
        let active = self.active_mask();
        for (i, n) in self.nodes.iter().enumerate() {
            if active & (1u64 << i) == 0 {
                continue;
            }
            let a = format!("v{}", n.a);
            let b = format!("v{}", n.b);
            let p = format!("{:.9}", n.parameter);
            let expr = match n.op {
                0 => p,
                1 => format!("{a}+{b}"),
                2 => format!("{a}-{b}"),
                3 => format!("{a}*{b}"),
                4 => format!("abs({b})<0.01 ? {a} : {a}/{b}"),
                5 => format!("sin({a}*({p}))"),
                6 => format!("cos({a}*({p}))"),
                7 => format!("abs({a})"),
                8 => format!("min({a},{b})"),
                9 => format!("max({a},{b})"),
                10 => format!("tanh({a})"),
                11 => format!("{a}>{b}+0.0001 ? 1.0 : -1.0"),
                _ => unreachable!(),
            };
            s.push_str(&format!(
                "  float v{}=clamp({}, -8.0, 8.0);\n",
                INPUTS + i,
                expr
            ));
        }
        let end = INPUTS + self.nodes.len();
        s.push_str(&format!(
            "  return clamp(vec3(v{},v{},v{})*0.5+0.5, 0.0, 1.0);\n}}\n",
            end - 3,
            end - 2,
            end - 1
        ));
        s
    }
}
