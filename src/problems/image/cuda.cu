// Host ABI is defined by Node and Sample in the Rust modules.
// NODES is specialized once per experiment by NVRTC, never per individual.
#define INPUTS 6
#define LANES 64
struct Node { unsigned op, a, b; float parameter; };
struct Sample { float x, y, color[3]; };
static_assert(sizeof(Node)==16, "Node ABI");
static_assert(sizeof(Sample)==20, "Sample ABI");

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

__device__ void problem_initialize(unsigned *g,const unsigned *data,unsigned &rng){
    Node *p=(Node*)g;
    for(unsigned j=0;j<NODES;++j){Node n;n.op=next(rng)%12;n.a=next(rng)%(INPUTS+j);n.b=next(rng)%(INPUTS+j);n.parameter=unit(rng)*8.0f-4.0f;p[j]=n;}
}
__device__ void problem_recombine(const unsigned *a,const unsigned *b,unsigned *out,const unsigned *data,unsigned &rng,bool cross){
    unsigned split=(next(rng)%NODES)*4;
    for(unsigned j=0;j<GENOME_WORDS;++j)out[j]=(cross&&j>=split)?b[j]:a[j];
}
__device__ void problem_mutate(unsigned *g,const unsigned *data,unsigned &rng,float rate,float scale){
    Node *p=(Node*)g;
    for(unsigned j=0;j<NODES;++j){if(unit(rng)<rate){unsigned choice=next(rng)%4;
        if(choice==0)p[j].op=next(rng)%12;else if(choice==1)p[j].a=next(rng)%(INPUTS+j);else if(choice==2)p[j].b=next(rng)%(INPUTS+j);
        else p[j].parameter=bound(p[j].parameter+(unit(rng)*2.0f-1.0f)*scale,-4.0f,4.0f);
    }}
}
__device__ float problem_score(const unsigned *g,const unsigned *data,unsigned lane,unsigned lanes){
    const Node *p=(const Node*)g;unsigned count=data[0];const Sample *samples=(const Sample*)(data+2);float error=0;
    for(unsigned s=lane;s<count;s+=lanes){Sample sample=samples[s];float color[3];evaluate(p,sample.x,sample.y,color);float pixel_error=0;
        for(int c=0;c<3;++c){float d=color[c]-sample.color[c];pixel_error+=d*d;}error+=pixel_error;
    }
    return error/float(3*count)+(lane==0?__uint_as_float(data[1])*float(__popcll(active_mask(p)))/float(NODES):0.0f);
}
