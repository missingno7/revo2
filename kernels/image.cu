// Host ABI is defined by Node and Sample in the Rust modules.
// NODES is specialized once per experiment by NVRTC, never per individual.
#define INPUTS 6
#define LANES 64
struct Node { unsigned op, a, b; float parameter; };
struct Sample { float x, y, color[3]; };
static_assert(sizeof(Node)==16, "Node ABI");
static_assert(sizeof(Sample)==20, "Sample ABI");

__device__ unsigned rng_seed(unsigned long long seed, unsigned cell, unsigned generation) {
    unsigned x=unsigned(seed)^unsigned(seed>>32);
    x^=cell*0x9e3779b9u;
    x^=generation*0x85ebca6bu;
    x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16;
    return x ? x : 1;
}
__device__ unsigned next(unsigned &x) { x^=x<<13; x^=x>>17; x^=x<<5; return x; }
__device__ float unit(unsigned &x) { return float(next(x)>>8)*(1.0f/16777216.0f); }
__device__ float bound(float x, float lo, float hi) { return fminf(hi,fmaxf(lo,x)); }
__device__ int arity(unsigned op) { return op==0 ? 0 : (op==5||op==6||op==7||op==10 ? 1 : 2); }

__device__ unsigned long long active_mask(const Node *p) {
    unsigned long long mask=7ull<<(NODES-3);
    for(int i=NODES-1;i>=0;--i) {
        if(!(mask&(1ull<<i))) continue;
        Node n=p[i];
        if(arity(n.op)>=1 && n.a>=INPUTS) mask|=1ull<<(n.a-INPUTS);
        if(arity(n.op)==2 && n.b>=INPUTS) mask|=1ull<<(n.b-INPUTS);
    }
    return mask;
}

__device__ void evaluate(const Node *p, float x, float y, float *color) {
    float v[INPUTS+NODES];
    v[0]=x; v[1]=y; v[2]=sqrtf(x*x+y*y);
    v[3]=sinf(3.14159265358979323846f*x);
    v[4]=sinf(3.14159265358979323846f*y); v[5]=1.0f;
    for(int i=0;i<NODES;++i) {
        Node n=p[i];
        float a=v[n.a], b=v[n.b], result=0;
        switch(n.op) {
            case 0: result=n.parameter; break;
            case 1: result=a+b; break;
            case 2: result=a-b; break;
            case 3: result=a*b; break;
            case 4: result=fabsf(b)<0.01f ? a : a/b; break;
            case 5: result=sinf(a*n.parameter); break;
            case 6: result=cosf(a*n.parameter); break;
            case 7: result=fabsf(a); break;
            case 8: result=fminf(a,b); break;
            case 9: result=fmaxf(a,b); break;
            case 10: result=tanhf(a); break;
            case 11: result=a>b+0.0001f ? 1.0f : -1.0f; break;
        }
        v[INPUTS+i]=bound(result,-8.0f,8.0f);
    }
    for(int c=0;c<3;++c) color[c]=bound(v[INPUTS+NODES-3+c]*0.5f+0.5f,0.0f,1.0f);
}

extern "C" __global__ void initialize(Node *programs,unsigned count,unsigned long long seed) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    unsigned rng=rng_seed(seed,i,0);
    for(unsigned j=0;j<NODES;++j) {
        Node n;
        n.op=next(rng)%12; n.a=next(rng)%(INPUTS+j); n.b=next(rng)%(INPUTS+j);
        n.parameter=unit(rng)*8.0f-4.0f;
        programs[i*NODES+j]=n;
    }
}

// One block per candidate, with the same program across all lanes. Sample work
// is parallelized within a block; different programs don't diverge within a warp.
extern "C" __global__ void evaluate_population(const Node *programs,const Sample *samples,
    unsigned sample_count,float complexity_weight,float *scores) {
    unsigned i=blockIdx.x, lane=threadIdx.x;
    __shared__ Node program[NODES];
    __shared__ float partial[LANES];
    for(unsigned j=lane;j<NODES;j+=LANES) program[j]=programs[i*NODES+j];
    __syncthreads();
    float error=0;
    for(unsigned s=lane;s<sample_count;s+=LANES) {
        Sample sample=samples[s];
        float color[3]; evaluate(program,sample.x,sample.y,color);
        float pixel_error=0;
        for(int c=0;c<3;++c) { float d=color[c]-sample.color[c]; pixel_error+=d*d; }
        error+=pixel_error;
    }
    partial[lane]=error;
    __syncthreads();
    for(unsigned stride=LANES/2;stride;stride/=2) {
        if(lane<stride) partial[lane]+=partial[lane+stride];
        __syncthreads();
    }
    if(lane==0) scores[i]=partial[0]/float(3*sample_count)+complexity_weight*float(__popcll(active_mask(program)))/float(NODES);
}

__device__ unsigned tournament(unsigned &rng,const unsigned *hood,const float *scores) {
    unsigned a=hood[next(rng)%5],b=hood[next(rng)%5];
    return scores[a]<=scores[b] ? a : b;
}

extern "C" __global__ void breed(const Node *current,const float *scores,Node *offspring,
    unsigned width,unsigned height,unsigned long long seed,unsigned generation,
    float mutation_rate,float mutation_scale,float crossover_rate) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=width*height) return;
    unsigned x=i%width,y=i/width;
    unsigned hood[5]={i,y*width+(x+width-1)%width,y*width+(x+1)%width,
        ((y+height-1)%height)*width+x,((y+1)%height)*width+x};
    unsigned rng=rng_seed(seed,i,generation);
    unsigned first=tournament(rng,hood,scores),second=tournament(rng,hood,scores);
    bool cross=unit(rng)<crossover_rate;
    unsigned split=next(rng)%NODES;
    for(unsigned j=0;j<NODES;++j) {
        Node n=current[((cross&&j>=split)?second:first)*NODES+j];
        if(unit(rng)<mutation_rate) {
            unsigned choice=next(rng)%4;
            if(choice==0) n.op=next(rng)%12;
            else if(choice==1) n.a=next(rng)%(INPUTS+j);
            else if(choice==2) n.b=next(rng)%(INPUTS+j);
            else n.parameter=bound(n.parameter+(unit(rng)*2.0f-1.0f)*mutation_scale,-4.0f,4.0f);
        }
        offspring[i*NODES+j]=n;
    }
}

extern "C" __global__ void replace(const Node *current,const float *scores,
    Node *offspring,float *next_scores,unsigned count) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    if(next_scores[i]>scores[i]) {
        for(int j=0;j<NODES;++j) offspring[i*NODES+j]=current[i*NODES+j];
        next_scores[i]=scores[i];
    }
}

// Reduction and readback only run at observation points, never each generation.
extern "C" __global__ void summarize(const float *scores,unsigned count,float *result) {
    __shared__ float mins[256],sums[256],maxs[256];
    __shared__ unsigned indices[256];
    unsigned lane=threadIdx.x,index=0;
    float best=2.0f,sum=0,worst=0;
    for(unsigned i=lane;i<count;i+=256) {
        float s=scores[i]; sum+=s; worst=fmaxf(worst,s);
        if(s<best || (s==best && i<index)) {best=s;index=i;}
    }
    mins[lane]=best; sums[lane]=sum; maxs[lane]=worst; indices[lane]=index;
    __syncthreads();
    for(unsigned stride=128;stride;stride/=2) {
        if(lane<stride) {
            sums[lane]+=sums[lane+stride];
            maxs[lane]=fmaxf(maxs[lane],maxs[lane+stride]);
            if(mins[lane+stride]<mins[lane] || (mins[lane+stride]==mins[lane]&&indices[lane+stride]<indices[lane])) {
                mins[lane]=mins[lane+stride];indices[lane]=indices[lane+stride];
            }
        }
        __syncthreads();
    }
    if(lane==0) {result[0]=float(indices[0]);result[1]=mins[0];result[2]=sums[0]/float(count);result[3]=maxs[0];}
}

extern "C" __global__ void render_image(const Node *programs,unsigned index,
    unsigned width,unsigned height,unsigned char *pixels) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=width*height) return;
    float x=2.0f*(float(i%width)+0.5f)/float(width)-1.0f;
    float y=2.0f*(float(i/width)+0.5f)/float(height)-1.0f;
    float color[3]; evaluate(programs+index*NODES,x,y,color);
    for(int c=0;c<3;++c) pixels[3*i+c]=(unsigned char)roundf(color[c]*255.0f);
}
