__device__ unsigned rng_seed(unsigned long long seed,unsigned cell,unsigned generation) {
    unsigned x=unsigned(seed)^unsigned(seed>>32);x^=cell*0x9e3779b9u;x^=generation*0x85ebca6bu;
    x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return x?x:1;
}
__device__ unsigned next(unsigned &x){x^=x<<13;x^=x>>17;x^=x<<5;return x;}
__device__ float unit(unsigned &x){return float(next(x)>>8)*(1.0f/16777216.0f);}
__device__ float bound(float x,float lo,float hi){return fminf(hi,fmaxf(lo,x));}
