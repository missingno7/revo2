//! Optional demo-owned GPU preview. The core knows nothing about image output.
use anyhow::{Context, Result};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use std::sync::Arc;

pub struct ImageRenderer {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
    nodes: usize,
}
impl ImageRenderer {
    pub fn new(nodes: usize) -> Result<Self> {
        anyhow::ensure!(
            (3..=64).contains(&nodes),
            "invalid image renderer program size"
        );
        anyhow::ensure!(
            unsafe { cudarc::driver::sys::is_culib_present() }
                && unsafe { cudarc::nvrtc::sys::is_culib_present() },
            "CUDA preview needs the NVIDIA driver and NVRTC"
        );
        let ctx = CudaContext::new(0)?;
        let (major, minor) = ctx.compute_capability()?;
        let interpreter = include_str!("cuda.cu")
            .split("__device__ void problem_initialize")
            .next()
            .unwrap();
        let source=format!("#define NODES {nodes}\n__device__ float bound(float x,float lo,float hi){{return fminf(hi,fmaxf(lo,x));}}\n{interpreter}\n{}",r#"
extern "C" __global__ void preview(const unsigned *genome,unsigned size,unsigned char *pixels){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=size*size)return;
    float x=2.0f*(float(i%size)+0.5f)/float(size)-1.0f,y=2.0f*(float(i/size)+0.5f)/float(size)-1.0f;
    float color[3];evaluate((const Node*)genome,x,y,color);
    for(int c=0;c<3;++c)pixels[3*i+c]=(unsigned char)roundf(color[c]*255.0f);
}
"#);
        let ptx = compile_ptx_with_opts(
            source,
            CompileOptions {
                fmad: Some(false),
                options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
                ..Default::default()
            },
        )
        .context("compiling demo-owned image preview")?;
        let module = ctx.load_module(ptx)?;
        Ok(Self {
            stream: ctx.default_stream(),
            kernel: module.load_function("preview")?,
            nodes,
        })
    }
    pub fn render(&self, genome: &[u32], size: u32) -> Result<image::RgbImage> {
        anyhow::ensure!(
            genome.len() == self.nodes * 4,
            "incorrect image preview genome length"
        );
        super::ImageProblem::decode(genome)?;
        anyhow::ensure!((1..=4096).contains(&size), "render size must be 1–4096");
        let input = self.stream.clone_htod(genome)?;
        let mut pixels = self.stream.alloc_zeros::<u8>((size * size * 3) as usize)?;
        // SAFETY: Validated acyclic genome and a correctly sized RGB pixel buffer.
        unsafe {
            self.stream
                .launch_builder(&self.kernel)
                .arg(&input)
                .arg(&size)
                .arg(&mut pixels)
                .launch(LaunchConfig::for_num_elems(size * size))
        }?;
        let bytes = self.stream.clone_dtoh(&pixels)?;
        Ok(image::RgbImage::from_raw(size, size, bytes).unwrap())
    }
}
