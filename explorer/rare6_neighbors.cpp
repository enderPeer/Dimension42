// Local mutation sensitivity, not a global rarity count.
#define main rare6_sampling_main
#include "rare6_probe.cpp"
#undef main

int main(int argc,char**argv){
    if(argc<2)return 2;
    std::cout<<"[";
    for(int k=1;k<argc;k++){
        uint64_t p=std::stoull(argv[k],nullptr,16);
        std::vector<int>truth(65536);
        for(int a=0;a<256;a++)for(int b=0;b<256;b++)truth[a*256+b]=run(p,a,b).answer;
        std::vector<uint64_t>neutral;
        for(int bit=0;bit<48;bit++){
            uint64_t q=p^(1ull<<bit);bool same=true;
            for(int t=0;t<16;t++)if(run(q,pa[t],pb[t]).answer!=truth[pa[t]*256+pb[t]]){same=false;break;}
            if(!same)continue;
            for(int a=0;a<256&&same;a++)for(int b=0;b<256;b++)if(run(q,a,b).answer!=truth[a*256+b]){same=false;break;}
            if(same)neutral.push_back(q);
        }
        if(k>1)std::cout<<",";
        std::cout<<"{\"program\":\""<<std::hex<<std::uppercase<<std::setw(12)<<std::setfill('0')<<p<<std::dec
                 <<"\",\"sensitive_bits\":"<<48-neutral.size()<<",\"neutral_one_bit_neighbors\":[";
        for(unsigned i=0;i<neutral.size();i++){
            if(i)std::cout<<",";
            std::cout<<"\""<<std::hex<<std::uppercase<<std::setw(12)<<std::setfill('0')<<neutral[i]<<std::dec<<"\"";
        }
        std::cout<<"]}";
    }
    std::cout<<"]\n";
    return 0;
}
