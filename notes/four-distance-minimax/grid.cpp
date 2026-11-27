// Exact lower-bound recursion for four Brownian target distances.
// Input rows hold the four downward-rounded survivals followed by the four
// downward-rounded costs, all in units of 1/K. Time endpoints are nested.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#ifdef NDEBUG
#error The certificate requires integer bounds checks.
#endif
#include <vector>

using I = std::int64_t;
struct Kernel { std::array<I,4> s{}, a{}; };
struct Transition { int base; I rem; };

int main(int argc,char **argv) {
    bool audit = argc==2 && std::string(argv[1])=="--audit";
    assert(argc==1 || audit);
    int D, nx, ny, nz, octaves, parts;
    I P, K, cap;
    if (!(std::cin >> D >> nx >> ny >> nz >> octaves >> parts >> P >> K >> cap)) return 2;
    assert(D>0 && D<=200 && nx>=0 && ny>=0 && nz>=0);
    assert(std::max({nx,ny,nz})<=250 && (parts&(parts-1))==0 && parts>=1 && parts<=4096);
    assert(octaves>0 && octaves<=30 && K>0 && P>0 && cap>0);
    const int nb = octaves*parts;
    std::vector<Kernel> table(nb+1);
    for(auto &row:table) {
        for(auto &v:row.s) std::cin>>v;
        for(auto &v:row.a) std::cin>>v;
        assert(std::cin && std::is_sorted(row.s.begin(),row.s.end()));
        assert(row.s[0]>=0 && row.s[3]>0 && row.s[3]<=K);
        assert(*std::min_element(row.a.begin(),row.a.end())>=0);
    }
    // Guard all integer products in the hot loop, using wider arithmetic once.
    const I d2 = I(D)*D;
    std::array<int,3> limits{nx,ny,nz};
    for(const auto &row:table) {
        __int128 cost = (__int128)d2*row.a[3];
        for(int c=0;c<3;++c) cost+=(__int128)limits[c]*limits[c]*row.a[c];
        assert(cost*P+(__int128)d2*K*cap*P < std::numeric_limits<I>::max());
    }
    assert((__int128)251*251*K < std::numeric_limits<I>::max());
    std::array<std::vector<Transition>,3> transitions;
    for (int c=0;c<3;++c) {
        auto &out=transitions[c];
        out.resize((limits[c]+1)*(nb+1));
        for(int index=0;index<=limits[c];++index) {
            for(int t=0;t<=nb;++t) {
                I S=table[t].s[3], pos=I(index)*index*table[t].s[c];
                // The float estimate is corrected by exact integer comparisons.
                int k=std::min(index,int(std::sqrt(double(pos)/double(S))));
                while(I(k)*k*S>pos) --k;
                while(I(k+1)*(k+1)*S<=pos) ++k;
                I rem=(pos-I(k)*k*S)/(2*k+1);
                assert(k>=0 && k<=index && rem>=0 && rem<S && (k<index || rem==0));
                assert(I(k)*k*S+(2*k+1)*rem<=pos);
                out[index*(nb+1)+t]={k,rem};
            }
        }
    }
    const int stride_x=(ny+2)*(nz+2),stride_y=nz+2;
    std::vector<I> values((nx+2)*stride_x,0);
    I best_num=0,best_den=1, tests=0, nodes=0;
    int wi=0,wj=0,wk=0;
    for(int i=0;i<=nx;++i) for(int j=0;j<=ny;++j) for(int k=0;k<=nz;++k) {
        const int target=i*stride_x+j*stride_y+k;
        const I ix=I(i)*i,iy=I(j)*j,iz=I(k)*k;
        const auto *tx=&transitions[0][i*(nb+1)];
        const auto *ty=&transitions[1][j*(nb+1)];
        const auto *tz=&transitions[2][k*(nb+1)];
        I best=cap*P;
        auto bound=[&](int left,int right) {
            ++tests;
            const auto &a=table[left].a;
            I cost=P*(ix*a[0]+iy*a[1]+iz*a[2]+d2*a[3]);
            I S=table[right].s[3], self=0, continuation=0;
            int vertex=tx[right].base*stride_x+ty[right].base*stride_y+tz[right].base;
            I f=tx[right].rem,g=ty[right].rem,h=tz[right].rem;
            int dx=stride_x,dy=stride_y,dz=1;
            if(f<g){std::swap(f,g);std::swap(dx,dy);}
            if(g<h){std::swap(g,h);std::swap(dy,dz);}
            if(f<g){std::swap(f,g);std::swap(dx,dy);}
            auto add=[&](I mass) {
                if(!mass) return;
                assert(mass>0 && vertex<=target);
                if(vertex==target) self+=mass;
                else continuation+=mass*values[vertex];
            };
            add(S-f);vertex+=dx;add(f-g);vertex+=dy;add(g-h);vertex+=dz;add(h);
            assert(self<=K);
            if(self==K) return cap*P;
            return (cost+d2*continuation)/(d2*(K-self));
        };
        // Seed with genuine single-bin bounds, then cover every octave.
        for(int o=4;o<octaves;o+=2) {
            int left=o*parts;
            best=std::min(best,bound(left,left+1));
        }
        auto cover=[&](auto &rec,int l,int r)->void {
            I value=bound(l,r);
            if(value>=best) return;
            if(r-l==1){best=value;return;}
            int m=(l+r)/2;
            rec(rec,l,m);rec(rec,m,r);
        };
        for(int o=0;o<octaves;++o) cover(cover,o*parts,(o+1)*parts);
        values[target]=best;
        ++nodes;
        I num=d2*best,den=P*(d2+ix+iy+iz);
        if((__int128)num*best_den>(__int128)best_num*den) {
            best_num=num;best_den=den;wi=i;wj=j;wk=k;
        }
    }
    I audit_checks=0;
    if(audit) {
        // Fresh Bellman check on the completed table, without solving for self.
        // Cube corner masses are computed by subset min/max instead of sorting.
        for(int i=0;i<=nx;++i) for(int j=0;j<=ny;++j) for(int k=0;k<=nz;++k) {
            const I target_value=values[i*stride_x+j*stride_y+k];
            const auto *x=&transitions[0][i*(nb+1)];
            const auto *y=&transitions[1][j*(nb+1)];
            const auto *z=&transitions[2][k*(nb+1)];
            auto cover=[&](auto &rec,int l,int r)->void {
                ++audit_checks;
                const auto &a=table[l].a;
                I cost=P*(I(i)*i*a[0]+I(j)*j*a[1]+I(k)*k*a[2]+d2*a[3]);
                if(cost>=d2*target_value*(K-table[r].s[0])) return;
                const I f=x[r].rem,g=y[r].rem,h=z[r].rem,S=table[r].s[3];
                const auto *v=&values[x[r].base*stride_x+y[r].base*stride_y+z[r].base];
                auto pos=[](I a){return std::max(I(0),a);};
                I continuation=(S-std::max({f,g,h}))*v[0]
                      +pos(f-std::max(g,h))*v[stride_x]
                      +pos(g-std::max(f,h))*v[stride_y]
                      +pos(h-std::max(f,g))*v[1]
                      +pos(std::min(f,g)-h)*v[stride_x+stride_y]
                      +pos(std::min(f,h)-g)*v[stride_x+1]
                      +pos(std::min(g,h)-f)*v[stride_y+1]
                      +std::min({f,g,h})*v[stride_x+stride_y+1];
                if(cost+d2*continuation>=d2*K*target_value) return;
                assert(r-l>1);
                int mid=(l+r)/2;rec(rec,l,mid);rec(rec,mid,r);
            };
            for(int o=0;o<octaves;++o) cover(cover,o*parts,(o+1)*parts);
        }
    }
    std::cout<<best_num<<" "<<best_den<<" "<<wi<<" "<<wj<<" "<<wk
             <<" "<<nodes<<" "<<tests<<" "<<audit_checks<<"\n";
}

