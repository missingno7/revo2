// Generic engine. No problem identifiers, geometry, pixels or opcodes here.
extern "C" __global__ void initialize(unsigned *genomes,const unsigned *data,unsigned count,unsigned long long seed){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;unsigned rng=rng_seed(seed,i,0);problem_initialize(genomes+i*GENOME_WORDS,data,rng);
}
extern "C" __global__ void evaluate_population(const unsigned *genomes,const unsigned *data,float *scores){
    unsigned lane=threadIdx.x;__shared__ unsigned genome[GENOME_WORDS];__shared__ float partial[EVALUATION_LANES];
    for(unsigned j=lane;j<GENOME_WORDS;j+=EVALUATION_LANES)genome[j]=genomes[blockIdx.x*GENOME_WORDS+j];__syncthreads();
    partial[lane]=problem_score(genome,data,lane,EVALUATION_LANES);__syncthreads();
    for(unsigned stride=EVALUATION_LANES/2;stride;stride/=2){if(lane<stride)partial[lane]+=partial[lane+stride];__syncthreads();}
    if(lane==0)scores[blockIdx.x]=isfinite(partial[0])?partial[0]:__int_as_float(0x7f7fffff);
}
__device__ unsigned tournament(unsigned &rng,const unsigned *hood,const float *scores){unsigned a=hood[next(rng)%5],b=hood[next(rng)%5];return scores[a]<=scores[b]?a:b;}
extern "C" __global__ void breed(const unsigned *current,const float *scores,unsigned *offspring,const unsigned *data,
    unsigned width,unsigned height,unsigned long long seed,unsigned generation,float mutation_rate,float mutation_scale,float crossover_rate){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=width*height)return;unsigned x=i%width,y=i/width;
    unsigned hood[5]={i,y*width+(x+width-1)%width,y*width+(x+1)%width,((y+height-1)%height)*width+x,((y+1)%height)*width+x};
    unsigned rng=rng_seed(seed,i,generation);unsigned first=tournament(rng,hood,scores),second=tournament(rng,hood,scores);bool cross=unit(rng)<crossover_rate;
    unsigned *child=offspring+i*GENOME_WORDS;
    problem_recombine(current+first*GENOME_WORDS,current+second*GENOME_WORDS,child,data,rng,cross);problem_mutate(child,data,rng,mutation_rate,mutation_scale);
}
extern "C" __global__ void replace(const unsigned *current,const float *scores,unsigned *offspring,float *next_scores,unsigned count){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    if(next_scores[i]>scores[i]){for(unsigned j=0;j<GENOME_WORDS;++j)offspring[i*GENOME_WORDS+j]=current[i*GENOME_WORDS+j];next_scores[i]=scores[i];}
}
extern "C" __global__ void summarize(const float *scores,unsigned count,float *result){
    __shared__ float mins[256],maxs[256];__shared__ double means[256];__shared__ unsigned indices[256];
    unsigned lane=threadIdx.x,index=0;float inf=__int_as_float(0x7f800000),best=inf,worst=-inf;double mean=0;
    for(unsigned i=lane;i<count;i+=256){float s=scores[i];mean+=double(s)/double(count);worst=fmaxf(worst,s);if(s<best||(s==best&&i<index)){best=s;index=i;}}
    mins[lane]=best;means[lane]=mean;maxs[lane]=worst;indices[lane]=index;__syncthreads();
    for(unsigned stride=128;stride;stride/=2){if(lane<stride){means[lane]+=means[lane+stride];maxs[lane]=fmaxf(maxs[lane],maxs[lane+stride]);
        if(mins[lane+stride]<mins[lane]||(mins[lane+stride]==mins[lane]&&indices[lane+stride]<indices[lane])){mins[lane]=mins[lane+stride];indices[lane]=indices[lane+stride];}}__syncthreads();}
    if(lane==0){result[0]=float(indices[0]);result[1]=mins[0];result[2]=means[0];result[3]=maxs[0];}
}
