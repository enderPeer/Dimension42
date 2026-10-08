// Bounded CPU sample of six-byte NANO behaviors. Does not establish global rarity.
// usage: rare6_probe prefixes.txt five_byte_catalog.txt sample_count
// Uses original 16-byte NANO, 64 steps, a in M6, b in M7, first OUT.
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using Sig = std::array<uint8_t,16>;
static const int pa[16]={3,200,17,0,255,91,128,44,7,160,66,1,230,99,12,181};
static const int pb[16]={5,13,250,0,1,77,128,210,100,33,66,254,9,140,31,2};
static uint64_t signature_hash(const Sig &s){uint64_t h=1469598103934665603ull;for(auto x:s)h=(h^x)*1099511628211ull;return h?h:1;}
struct Hash {size_t operator()(const Sig&s)const{return signature_hash(s);}};
struct Result {int answer,steps;bool modified;};
static Result run(uint64_t p,int a,int b){
    uint8_t m[16]={0};for(int i=0;i<6;i++)m[i]=(p>>(8*(5-i)))&255;
    m[6]=a;m[7]=b;unsigned A=0,pc=0;bool modified=false;
    for(int step=1;step<=64;step++){
        unsigned ins=m[pc],op=ins>>4,n=ins&15;pc=(pc+1)%6;int write=-1;
        switch(op){
        case 1:A=n;break;case 2:A=(A+n)&255;break;case 3:A=(A-n)&255;break;
        case 4:A=m[n];break;case 5:write=A;break;case 6:A=(A+m[n])&255;break;
        case 7:pc=n%6;break;case 8:if(A==0)pc=n%6;break;
        case 9:write=(m[n]+1)&255;break;case 10:write=(m[n]-1)&255;break;
        case 11:{unsigned k=n&7;A=((A<<k)|(A>>(8-k)))&255;break;}
        case 12:A^=m[n];break;case 13:return {(int)m[n],step,modified};
        case 14:write=m[n]^255;break;case 15:return {-1,step,modified};
        }
        if(write>=0){modified|=n<6&&m[n]!=write;m[n]=write;}
    }return {-1,64,modified};
}
struct Entry{uint32_t count=0;uint64_t example=0;unsigned steps=0;bool modified=false;};
int main(int argc,char**argv){
    if(argc!=4){std::cerr<<"usage: rare6_probe prefixes catalog count\n";return 2;}
    std::vector<unsigned>prefixes;std::ifstream pf(argv[1]);unsigned pref;while(pf>>pref)prefixes.push_back(pref);
    if(prefixes.empty())return 2;
    std::unordered_set<uint64_t>known;std::ifstream cf(argv[2]);std::string line;
    while(std::getline(cf,line)){if(line.empty()||line[0]=='#')continue;std::istringstream s(line);uint64_t h;s>>std::hex>>h;known.insert(h);}
    if(known.empty())return 2;
    const unsigned long long n=std::stoull(argv[3]);std::mt19937_64 rng(642026);
    std::unordered_set<uint64_t>sampled;sampled.reserve(n);
    std::unordered_map<Sig,Entry,Hash>catalog;uint64_t useful=0;
    while(sampled.size()<n){
        uint64_t p=(uint64_t(prefixes[rng()%prefixes.size()])<<32)|(rng()&0xffffffffull);
        if(!sampled.insert(p).second)continue;
        Sig sig{};bool ok=true,mod=false;unsigned steps=0;
        for(int t=0;t<16;t++){auto r=run(p,pa[t],pb[t]);if(r.answer<0){ok=false;break;}sig[t]=r.answer;mod|=r.modified;steps+=r.steps;}
        if(!ok||std::all_of(sig.begin(),sig.end(),[&](uint8_t x){return x==sig[0];}))continue;
        useful++;auto&e=catalog[sig];e.count++;e.example=p;e.steps=steps;e.modified=mod;
    }
    std::vector<std::pair<Sig,Entry>>candidates;unsigned singletons=0,absent=0;
    for(const auto&kv:catalog){
        if(kv.second.count!=1)continue;singletons++;
        if(known.count(signature_hash(kv.first)))continue;absent++;
        std::set<uint8_t>distinct(kv.first.begin(),kv.first.end());
        if(kv.second.modified&&distinct.size()>=4)candidates.push_back(kv);
    }
    std::sort(candidates.begin(),candidates.end(),[](const auto&a,const auto&b){return a.second.steps<b.second.steps;});
    std::cout<<"{\"sampled_programs\":"<<n<<",\"useful_on_probes\":"<<useful<<",\"probe_signatures\":"<<catalog.size()
             <<",\"singleton_signatures_in_sample\":"<<singletons<<",\"singletons_absent_from_five_byte_catalog\":"<<absent
             <<",\"probe_filtered_candidates\":"<<candidates.size()<<",\"candidates\":[";
    unsigned emitted=0;
    for(const auto&kv:candidates){
        if(emitted==40)break;uint64_t p=kv.second.example;
        unsigned failures=0,modified_inputs=0,max_steps=0;std::array<unsigned,256>hist{};std::vector<int>truth(65536);
        bool depends_a=false,depends_b=false;
        for(int a=0;a<256;a++)for(int b=0;b<256;b++){
            auto r=run(p,a,b);truth[a*256+b]=r.answer;
            if(r.answer<0)failures++;else hist[r.answer]++;
            modified_inputs+=r.modified;max_steps=std::max(max_steps,(unsigned)r.steps);
            if(a&&r.answer!=truth[b])depends_a=true;
            if(b&&r.answer!=truth[a*256])depends_b=true;
        }
        if(failures||(!depends_a&&!depends_b))continue;
        if(emitted++)std::cout<<",";
        std::cout<<"{\"program\":\""<<std::hex<<std::uppercase<<std::setw(12)<<std::setfill('0')<<p<<std::dec
                 <<"\",\"sample_occurrences\":1,\"full_input_pairs\":65536,\"missing_answers\":"<<failures
                 <<",\"depends_a\":"<<(depends_a?"true":"false")<<",\"depends_b\":"<<(depends_b?"true":"false")
                 <<",\"self_modifying_inputs\":"<<modified_inputs<<",\"max_steps_to_output\":"<<max_steps
                 <<",\"distinct_outputs\":"<<std::count_if(hist.begin(),hist.end(),[](unsigned x){return x>0;})
                 <<",\"most_common_output_count\":"<<*std::max_element(hist.begin(),hist.end())<<",\"probe_steps_total\":"<<kv.second.steps<<"}";
    }
    std::cout<<"]}\n";
    return 0;
}
