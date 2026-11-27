// Lower bounds for all three-round Brownian reset blocks. The input consists
// solely of directed integer bounds produced by verify_three_rounds.py.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <limits>
#include <unordered_map>
#include <vector>
#ifdef NDEBUG
#error Assertions must be enabled
#endif
using I=std::int64_t;
using U=std::uint32_t;
constexpr int N=1024, K=8192, LOG=1<<26, EXP=4096;
constexpr int START=-48*N, END=96*N, LEFT=-10*N, RIGHT=24*N;
constexpr int STEP=4, COUNT=(RIGHT-LEFT)/STEP, INF=100000000;
constexpr int MAX_LAG=16*N, MIN_LAG=-64*N;
static_assert((I)60*K*K < std::numeric_limits<U>::max());
static_assert((I)COUNT*60*K < std::numeric_limits<U>::max());

static_assert(((__int128)(END-START+1)*100*LOG+1)*EXP+LOG
              < std::numeric_limits<I>::max());
struct KernelTable {
    std::vector<I> loss;
    std::vector<U> cost, exp;
    KernelTable():loss(END-START+1),cost(loss.size()),exp(24*EXP+2) {
        for(size_t i=0;i<loss.size();++i) {
            std::cin>>loss[i]>>cost[i];
            assert(std::cin && loss[i]>=0 && I(loss[i])<100LL*LOG && cost[i]<=60*K);
        }
        for(auto &v:exp){std::cin>>v;assert(std::cin && v<=K);}
        assert(exp.back()==0);
    }
};

struct Slab {
    int gl,gr,target;
    // For lag a, T_g(v-a) is the survival of a geometric sequence ending
    // at e^(v-a). Smaller lags and smaller g both give lower bounds.
    // Histories ordered by residue for contiguous integer dot products.
    std::array<std::vector<U>,STEP> lanes;
    std::vector<U> own;
    std::unordered_map<int,std::vector<U>> weighted;
    std::unordered_map<I,U> cache;
    I boxes=0,visits=0,min_num=std::numeric_limits<I>::max();
    Slab(const KernelTable &t,int gl_,int gr_,int target_):gl(gl_),gr(gr_),target(target_),own(COUNT) {
        std::vector<I> sum(gl,1);  // one negative-log tick for the missing past
        for(int x=START;x<=END;++x) {
            int i=x-START, r=i%gl;
            sum[r]+=t.loss[i]; // T_g includes the round at x
            I ix=(sum[r]*EXP+LOG-1)/LOG;
            lanes[i%STEP].push_back(t.exp[std::min<I>(ix,t.exp.size()-1)]);
        }
        const U *prev=history(std::min(gl,32*N));  // smaller lag is a lower bound
        for(int i=0;i<COUNT;++i)own[i]=(prev[i]*t.cost[LEFT-START+i*STEP])/K;
    }
    const U* history(int lag) const {
        int i=LEFT+STEP-lag-START;
        assert(i>=0 && i+(COUNT-1)*STEP<=END-START);
        return &lanes[i%STEP][i/STEP];
    }
    U integral(int a,int b) {
        a=std::min(a,MAX_LAG); b=std::min(b,MAX_LAG);
        if(a<MIN_LAG || b<MIN_LAG)return 0;
        // Quantize lags downwards, using monotonicity. Exact multiples
        // are left untouched; C++ division truncates towards zero.
        auto down=[](int x){return x>=0 ? (x/2)*2 : -((1-x)/2)*2;};
        a=down(a);b=down(b);
        if(a>b)std::swap(a,b);
        I key=(I)(a-MIN_LAG)*(MAX_LAG-MIN_LAG+1)+b-MIN_LAG;
        auto saved=cache.find(key);if(saved!=cache.end())return saved->second;
        auto pair=weighted.try_emplace(a);
        auto &w=pair.first->second;
        if(pair.second){
            w.resize(COUNT);
            const U *pa=history(a);
            for(int i=0;i<COUNT;++i)w[i]=(own[i]*pa[i])/K;
        }
        U value=0;
        const U *pb=history(b);
        for(int i=0;i<COUNT;++i)value+=(w[i]*pb[i])/K;
        cache.emplace(key,value);
        return value;
    }
    void cover(int dl,int dr,int el,int er) {
        ++visits;
        // log sizes are (0,-d,-e); choose the largest detrended size first.
        if(3*dr < -gr || 3*er < -2*gr)return;
        // Independently unfold the integral around each of the three rounds.
        // Its other two latest preceding deadlines have the two lags below.
        I score=integral(gl+dl,gl+el);
        if(dr!=INF)score+=integral(-dr,gl+el-dr);
        if(er!=INF)score+=integral(-er,dl-er);
        if(100*STEP*score>I(target)*K*gr){
            ++boxes;min_num=std::min(min_num,score);return;
        }
        int wd=dr==INF?0:dr-dl,we=er==INF?0:er-el;
        if(std::max(wd,we)<2) {
            std::cerr<<"Unresolved "<<gl<<" "<<gr<<" "<<dl<<" "<<dr<<" "<<el<<" "<<er
                     <<" bound "<<(double)STEP*score/(K*gr)<<"\n";
            std::exit(3);
        }
        if(wd>=we){int m=dl+wd/2;cover(dl,m,el,er);cover(m,dr,el,er);}
        else {int m=el+we/2;cover(dl,dr,el,m);cover(dl,dr,m,er);}
    }
    void verify() {
        std::vector<int> cuts={-64*N,-32*N,-16*N,-8*N,-4*N,-2*N,-N,-N/2,0,N/2,N,2*N,4*N,8*N,16*N,INF};
        for(size_t i=1;i<cuts.size();++i)for(size_t j=1;j<cuts.size();++j)
            cover(cuts[i-1],cuts[i],cuts[j-1],cuts[j]);
    }
};
int main(){
    constexpr int target=354;
    KernelTable table;
    I count=0,visits=0,queries=0;
    I num=1000,den=1;
    for(int gl=N/512;gl<64*N;){
        int gr=std::min(64*N,gl+std::max(1,gl/256));
        Slab s(table,gl,gr,target);s.verify();
        assert(s.boxes>0);
        I n=STEP*s.min_num,d=K*gr;
        if((__int128)n*den<(__int128)num*d){num=n;den=d;}
        count+=s.boxes;visits+=s.visits;queries+=s.cache.size();gl=gr;
    }
    std::cout<<count<<" "<<visits<<" "<<queries<<" "<<num<<" "<<den<<"\n";
}

