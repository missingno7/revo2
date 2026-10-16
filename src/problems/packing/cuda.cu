__device__ void problem_initialize(unsigned *g,const unsigned *data,unsigned &rng){
    for(unsigned i=0;i<RECTANGLES;++i)g[i]=i;
    for(unsigned i=RECTANGLES-1;i>0;--i){unsigned j=next(rng)%(i+1),v=g[i];g[i]=g[j];g[j]=v;}
    for(unsigned i=RECTANGLES;i<2*RECTANGLES;++i)g[i]=next(rng)%2;
    g[2*RECTANGLES]=data[0]+next(rng)%(data[2]-data[0]+1);
}
__device__ void problem_recombine(const unsigned *a,const unsigned *b,unsigned *out,const unsigned *data,unsigned &rng,bool cross){
    for(unsigned i=0;i<GENOME_WORDS;++i)out[i]=a[i];if(!cross)return;
    unsigned split=next(rng)%(RECTANGLES+1),end=split;
    for(unsigned i=0;i<RECTANGLES;++i){bool used=false;for(unsigned j=0;j<split;++j)if(a[j]==b[i])used=true;if(!used)out[end++]=b[i];}
    for(unsigned i=RECTANGLES;i<=2*RECTANGLES;++i)if(next(rng)%2==0)out[i]=b[i];
}
__device__ void problem_mutate(unsigned *g,const unsigned *data,unsigned &rng,float rate,float scale){
    for(unsigned i=0;i<RECTANGLES;++i){if(unit(rng)<rate){unsigned j=next(rng)%RECTANGLES,v=g[i];g[i]=g[j];g[j]=v;}if(unit(rng)<rate)g[RECTANGLES+i]^=1;}
    if(unit(rng)<rate){int delta=(int)((unit(rng)*2.0f-1.0f)*scale*float(data[2]));int width=int(g[2*RECTANGLES])+delta;width=width<int(data[0])?int(data[0]):width;width=width>int(data[1])?int(data[1]):width;g[2*RECTANGLES]=unsigned(width);}
}
__device__ float problem_score(const unsigned *g,const unsigned *data,unsigned lane,unsigned lanes){
    unsigned xs[RECTANGLES],ys[RECTANGLES],ws[RECTANGLES],hs[RECTANGLES];unsigned width=0,height=0,limit=g[2*RECTANGLES];
    for(unsigned i=0;i<RECTANGLES;++i){
        unsigned id=g[i],w=data[4+id*2],h=data[5+id*2];if(g[RECTANGLES+id]){unsigned t=w;w=h;h=t;}
        unsigned bx=0,by=0xffffffffu;
        for(unsigned c=0;c<=i;++c){unsigned x=c==0?0:xs[c-1]+ws[c-1];if(x+w>limit)continue;unsigned y=0;
            for(unsigned j=0;j<i;++j)if(x<xs[j]+ws[j]&&xs[j]<x+w)y=y>ys[j]+hs[j]?y:ys[j]+hs[j];
            if(y<by||(y==by&&x<bx)){bx=x;by=y;}
        }
        xs[i]=bx;ys[i]=by;ws[i]=w;hs[i]=h;width=width>bx+w?width:bx+w;height=height>by+h?height:by+h;
    }return 1.0f-float(data[3])/float(width*height);
}
