// Four posterior odds for five target distances. All decision tests are integer.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>
#ifdef NDEBUG
#error Enable assertions for certificate verification
#endif
using I = std::int64_t;
struct Row {std::array<I,5> s,a;};
struct Projection {int offset; I f;};
int main(){
    int D,oct,parts;
    std::array<int,4> limit,stride;
    I P,K,cap;
    if(!(std::cin>>D>>limit[0]>>limit[1]>>limit[2]>>limit[3]>>oct>>parts>>P>>K>>cap))return 2;
    assert(D>0 && D<=100 && oct>0 && oct<=32 && parts>=1 && parts<=2048 && !(parts&(parts-1)));
    assert(P>0 && P<=100000 && K>0 && K<=10000000 && cap>0 && cap<=40);
    I cells=1;
    for(int c=3;c>=0;--c){assert(limit[c]>=0 && limit[c]<=100);stride[c]=int(cells);cells*=limit[c]+2;}
    assert(cells<100000000);
    const int n=oct*parts;
    const I d2=I(D)*D;
    std::vector<Row> table(n+1);
    for(auto &r:table){
        for(I &s:r.s)std::cin>>s;
        for(I &a:r.a)std::cin>>a;
        assert(std::cin && std::is_sorted(r.s.begin(),r.s.end()) && r.s[0]>=0 && r.s[4]>0 && r.s[4]<=K);
        assert(*std::min_element(r.a.begin(),r.a.end())>=0);
        __int128 v=(__int128)d2*r.a[4];
        for(int c=0;c<4;++c)v+=(__int128)limit[c]*limit[c]*r.a[c];
        assert(P*v+(__int128)d2*K*cap*P<std::numeric_limits<I>::max());
    }
    std::array<std::vector<Projection>,4> projections;
    for(int c=0;c<4;++c){
        projections[c].resize((limit[c]+1)*(n+1));
        for(int i=0;i<=limit[c];++i)for(int t=0;t<=n;++t){
            I S=table[t].s[4],pos=I(i)*i*table[t].s[c];
            int b=int(std::sqrt(double(pos)/double(S)));
            while(I(b)*b*S>pos)--b;
            while(I(b+1)*(b+1)*S<=pos)++b;
            I f=(pos-I(b)*b*S)/(2*b+1);
            assert(b>=0 && b<=i && f>=0 && f<S && (b<i || f==0));
            assert(b*b*S+(2*b+1)*f<=pos);
            projections[c][i*(n+1)+t]={b*stride[c],f};
        }
    }
    std::vector<I> v(cells);
    I bestNum=0,bestDen=1,uniformValue=0,tests=0,audits=0;
    std::array<int,4> bestIndex{};
    auto node=[&](std::array<int,4> id,bool audit){
        std::array<const Projection*,4> proj;
        std::array<I,5> weight{0,0,0,0,d2};
        int idx=0; I sum=d2;
        for(int c=0;c<4;++c){
            proj[c]=projections[c].data()+id[c]*(n+1);weight[c]=I(id[c])*id[c];
            idx+=id[c]*stride[c];sum+=weight[c];
        }
        I best=audit?v[idx]:cap*P;
        auto cover=[&](auto &rec,int l,int r)->void {
            if(audit)++audits;else ++tests;
            I cost=0;for(int c=0;c<5;++c)cost+=weight[c]*table[l].a[c];cost*=P;
            if(cost>=d2*best*(K-table[r].s[0]))return;
            std::array<Projection,4> step;int vertex=0;
            for(int c=0;c<4;++c){auto p=proj[c][r];vertex+=p.offset;step[c]={stride[c],p.f};}
            std::sort(step.begin(),step.end(),[](Projection a,Projection b){return a.f>b.f;});
            I mass=table[r].s[4],cont=0,self=0;
            auto add=[&](I w){
                if(w==0)return;
                assert(w>0 && vertex<=idx);
                if(audit || vertex!=idx)cont+=w*v[vertex];
                else self+=w;
            };
            for(auto p:step){add(mass-p.f);vertex+=p.offset;mass=p.f;}
            add(mass);
            if(cost+d2*cont>=d2*best*(K-self))return;
            if(r-l>1){int m=(l+r)/2;rec(rec,l,m);rec(rec,m,r);return;}
            assert(!audit && self<K);
            best=(cost+d2*cont)/(d2*(K-self));
        };
        if(!audit)for(int o=4;o<oct;o+=2)cover(cover,o*parts,o*parts+1);
        for(int o=0;o<oct;++o)cover(cover,o*parts,(o+1)*parts);
        if(audit)return;
        v[idx]=best;
        if(id==std::array<int,4>{D,D,D,D})uniformValue=best;
        if((__int128)d2*best*bestDen>(__int128)bestNum*P*sum){
            bestNum=d2*best;bestDen=P*sum;bestIndex=id;
        }
    };
    bool audit;std::cin>>audit;assert(std::cin);
    for(int pass=0;pass<=(audit?1:0);++pass)
      for(int i=0;i<=limit[0];++i)for(int j=0;j<=limit[1];++j)
        for(int k=0;k<=limit[2];++k)for(int l=0;l<=limit[3];++l)node({i,j,k,l},pass==1);
    std::cout<<bestNum<<" "<<bestDen;
    for(auto i:bestIndex)std::cout<<" "<<i;
    std::cout<<" "<<uniformValue<<" "<<tests<<" "<<audits<<"\n";
}

