// Bounded, uniform sample of six-byte NANO FUNCTION phenotypes.
// Build: g++ -O3 -std=c++17 phenotype6_sample.cpp -o phenotype6_sample
// Run:   ./phenotype6_sample UNIQUE_PROGRAM_COUNT SEED > sample.json
// Standalone source; no external headers or existing project files required.
// Counts are frequencies in this sample, NEVER exhaustive six-byte rarity.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
using Signature = std::array<uint8_t, 16>;
constexpr uint64_t MASK48 = (uint64_t(1) << 48) - 1;
// This cap keeps the sampled-ID set bounded (roughly 330--450 MiB on common
// 64-bit libstdc++ builds). Actual peak RSS must be measured by the caller.
constexpr uint64_t MAX_SAMPLES = 8388608;
constexpr size_t MAX_VERIFY = 64;
constexpr int PA[16] = {3,200,17,0,255,91,128,44,7,160,66,1,230,99,12,181};
constexpr int PB[16] = {5,13,250,0,1,77,128,210,100,33,66,254,9,140,31,2};
constexpr const char* OPS[16] = {
    "NOP","LDI","ADDI","SUBI","LD","ST","ADD","JMP",
    "JZ","INC","DEC","ROL","XOR","OUT","NOT","HLT"
};

struct Result { int answer; unsigned steps; bool modified; };

Result execute(uint64_t program, int a, int b) {
    uint8_t memory[16] = {};
    for (int i=0; i<6; ++i) memory[i] = (program >> (8*(5-i))) & 255;
    memory[6] = static_cast<uint8_t>(a);
    memory[7] = static_cast<uint8_t>(b);
    unsigned accumulator=0, pc=0;
    bool modified=false;
    for (unsigned step=1; step<=64; ++step) {
        const unsigned instruction=memory[pc], op=instruction>>4, n=instruction&15;
        pc=(pc+1)%6;
        int write=-1;
        switch (op) {
        case 0: break;
        case 1: accumulator=n; break;
        case 2: accumulator=(accumulator+n)&255; break;
        case 3: accumulator=(accumulator-n)&255; break;
        case 4: accumulator=memory[n]; break;
        case 5: write=static_cast<int>(accumulator); break;
        case 6: accumulator=(accumulator+memory[n])&255; break;
        case 7: pc=n%6; break;
        case 8: if (accumulator==0) pc=n%6; break;
        case 9: write=(memory[n]+1)&255; break;
        case 10: write=(memory[n]-1)&255; break;
        case 11: {
            const unsigned rotation=n&7;
            accumulator=((accumulator<<rotation)|(accumulator>>(8-rotation)))&255;
            break;
        }
        case 12: accumulator^=memory[n]; break;
        case 13: return {static_cast<int>(memory[n]),step,modified};
        case 14: write=memory[n]^255; break;
        case 15: return {-1,step,modified};
        }
        if (write>=0) {
            modified |= n<6 && memory[n]!=write;
            memory[n]=static_cast<uint8_t>(write);
        }
    }
    return {-1,64,modified};
}

struct SignatureHash {
    size_t operator()(const Signature& signature) const {
        uint64_t h=14695981039346656037ull;
        for (auto byte: signature) h=(h^byte)*1099511628211ull;
        return static_cast<size_t>(h);
    }
};

struct Entry {
    uint64_t count=0;
    uint64_t example=MASK48;
    uint64_t modified_programs=0;
    uint64_t probe_steps_total=0;
};

struct Candidate {
    Signature signature;
    Entry entry;
    unsigned distinct;
};

std::string hex(uint64_t value, unsigned width) {
    std::ostringstream out;
    out<<std::uppercase<<std::hex<<std::setfill('0')<<std::setw(width)<<value;
    return out.str();
}

std::string signature_hex(const Signature& signature) {
    std::string result;
    for (auto byte: signature) result+=hex(byte,2);
    return result;
}

unsigned distinct_values(const Signature& signature) {
    std::array<bool,256> seen{};
    for (auto byte:signature) seen[byte]=true;
    return static_cast<unsigned>(std::count(seen.begin(),seen.end(),true));
}

uint64_t argument(const char* value) {
    const std::string s(value);
    if (s.empty() || s[0]=='-') throw std::invalid_argument("expected unsigned integer");
    size_t used=0;
    const uint64_t parsed=std::stoull(s,&used,10);
    if (used!=s.size()) throw std::invalid_argument("expected unsigned decimal integer");
    return parsed;
}

bool write_verification(const Candidate& candidate, bool comma) {
    const uint64_t program=candidate.entry.example;
    std::vector<int16_t> truth(65536);
    std::array<uint32_t,256> histogram{};
    uint32_t failures=0, modified_inputs=0;
    unsigned min_steps=65, max_steps=0;
    bool depends_a=false, depends_b=false;
    uint64_t step_sum=0, truth_hash=14695981039346656037ull;
    for (int a=0;a<256;++a) for (int b=0;b<256;++b) {
        const Result result=execute(program,a,b);
        truth[a*256+b]=static_cast<int16_t>(result.answer);
        modified_inputs+=result.modified;
        if (result.answer<0) ++failures;
        else {
            ++histogram[result.answer];
            truth_hash=(truth_hash^static_cast<uint8_t>(result.answer))*1099511628211ull;
            min_steps=std::min(min_steps,result.steps);
            max_steps=std::max(max_steps,result.steps);
            step_sum+=result.steps;
        }
        if (a && result.answer!=truth[b]) depends_a=true;
        if (b && result.answer!=truth[a*256]) depends_b=true;
    }
    if (failures) return false;
    if (comma) std::cout<<',';
    std::cout<<"{\"program\":\""<<hex(program,12)<<"\",\"probe_signature\":\""
        <<signature_hex(candidate.signature)<<"\",\"sample_occurrences_of_probe_signature\":1"
        <<",\"input_pairs_checked\":65536,\"missing_answers\":"<<failures
        <<",\"total_function\":"<<(failures==0?"true":"false")
        <<",\"depends_a\":"<<(depends_a?"true":"false")
        <<",\"depends_b\":"<<(depends_b?"true":"false")
        <<",\"self_modifying_inputs\":"<<modified_inputs
        <<",\"distinct_successful_outputs\":"
        <<std::count_if(histogram.begin(),histogram.end(),[](uint32_t count){return count>0;})
        <<",\"most_common_output_count\":"<<*std::max_element(histogram.begin(),histogram.end())
        <<",\"min_steps_to_output\":"<<(failures==65536?0:min_steps)
        <<",\"max_steps_to_output\":"<<max_steps
        <<",\"successful_output_steps_total\":"<<step_sum
        <<",\"truth_table_fnv1a64\":\""<<hex(truth_hash,16)<<"\""
        <<",\"disassembly\":[";
    for (int i=0;i<6;++i) {
        const unsigned instruction=(program>>(8*(5-i)))&255;
        if (i) std::cout<<',';
        std::cout<<"{\"address\":"<<i<<",\"byte\":\""<<hex(instruction,2)
            <<"\",\"opcode\":\""<<OPS[instruction>>4]<<"\",\"operand\":"<<(instruction&15)<<'}';
    }
    std::cout<<"],\"examples\":[";
    constexpr int EXAMPLES[][2]={{0,0},{1,0},{0,1},{3,5},{17,250},{128,128},{255,255},{255,0},{0,255}};
    bool first=true;
    for (const auto& pair:EXAMPLES) {
        const Result result=execute(program,pair[0],pair[1]);
        if (!first) std::cout<<',';
        first=false;
        std::cout<<"{\"a\":"<<pair[0]<<",\"b\":"<<pair[1]<<",\"output\":";
        if (result.answer<0) std::cout<<"null"; else std::cout<<result.answer;
        std::cout<<",\"steps\":"<<result.steps<<'}';
    }
    std::cout<<"]}";
    return true;
}
} // namespace

int main(int argc,char** argv) {
    try {
        if (argc!=3) {
            std::cerr<<"usage: phenotype6_sample UNIQUE_PROGRAM_COUNT SEED\n";
            return 2;
        }
        const uint64_t requested=argument(argv[1]), seed=argument(argv[2]);
        if (requested==0 || requested>MAX_SAMPLES) {
            std::cerr<<"sample count must be in [1,"<<MAX_SAMPLES<<"] to bound memory\n";
            return 2;
        }
        const auto started=std::chrono::steady_clock::now();
        std::mt19937_64 rng(seed);
        std::unordered_set<uint64_t> sampled;
        sampled.max_load_factor(1.0f);
        sampled.reserve(static_cast<size_t>(requested));
        std::unordered_map<Signature,Entry,SignatureHash> catalog;
        uint64_t accepted=0, rejected=0, constant=0, nonconstant=0;
        uint64_t observed_modified=0, accepted_modified=0, duplicate_draws=0;
        while (sampled.size()<requested) {
            const uint64_t program=rng()&MASK48;
            if (!sampled.insert(program).second) { ++duplicate_draws; continue; }
            Signature signature{};
            bool answers_all=true, modified=false;
            unsigned steps=0;
            for (int t=0;t<16;++t) {
                const Result result=execute(program,PA[t],PB[t]);
                modified|=result.modified;
                if (result.answer<0) { answers_all=false; break; }
                signature[t]=static_cast<uint8_t>(result.answer);
                steps+=result.steps;
            }
            observed_modified+=modified;
            if (!answers_all) { ++rejected; continue; }
            ++accepted;
            accepted_modified+=modified;
            if (distinct_values(signature)==1) ++constant; else ++nonconstant;
            auto& entry=catalog[signature];
            ++entry.count;
            entry.example=std::min(entry.example,program);
            entry.modified_programs+=modified;
            entry.probe_steps_total+=steps;
            if (sampled.size()%1048576==0)
                std::cerr<<"accepted="<<accepted<<" unique_sampled="<<sampled.size()<<'\n';
        }
        const double sampling_seconds=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-started).count();
        // Release the dominant allocation before the complete-input checks.
        std::unordered_set<uint64_t>().swap(sampled);
        std::vector<Candidate> ordered, candidates;
        uint64_t singleton_signatures=0;
        for (const auto& item:catalog) {
            Candidate candidate{item.first,item.second,distinct_values(item.first)};
            ordered.push_back(candidate);
            if (item.second.count==1) {
                ++singleton_signatures;
                if (candidate.distinct>=4) candidates.push_back(candidate);
            }
        }
        std::sort(ordered.begin(),ordered.end(),[](const Candidate& a,const Candidate& b){
            return a.signature<b.signature;
        });
        std::sort(candidates.begin(),candidates.end(),[](const Candidate& a,const Candidate& b){
            if (a.entry.modified_programs!=b.entry.modified_programs)
                return a.entry.modified_programs>b.entry.modified_programs;
            if (a.entry.probe_steps_total!=b.entry.probe_steps_total)
                return a.entry.probe_steps_total>b.entry.probe_steps_total;
            return a.entry.example<b.entry.example;
        });
        const size_t verifying=std::min(MAX_VERIFY,candidates.size());
        std::cout<<"{\"schema\":\"nano6-function-probe-sample-v1\",\"program_bytes\":6"
            <<",\"program_space_size\":281474976710656,\"step_budget\":64,\"memory_bytes\":16"
            <<",\"isa\":\"original NANO\",\"input_definition\":{\"a\":6,\"b\":7}"
            <<",\"initialization\":\"program in M0..M5; a in M6; b in M7; remaining memory, A and PC zero\""
            <<",\"answer_definition\":\"first OUT within 64 instructions; HLT without OUT and timeout are missing\""
            <<",\"sampler\":\"mt19937_64 masked to 48 bits; reject duplicate program IDs; full program space\""
            <<",\"seed\":"<<seed<<",\"sampled_unique_programs\":"<<requested
            <<",\"duplicate_random_draws\":"<<duplicate_draws
            <<",\"sampling_seconds\":"<<std::fixed<<std::setprecision(6)<<sampling_seconds
            <<",\"rarity_scope\":\"frequency of exact 16-probe signature in this uniform sample only; not global rarity\""
            <<",\"probes\":[";
        for (int t=0;t<16;++t) {
            if (t) std::cout<<',';
            std::cout<<'['<<PA[t]<<','<<PB[t]<<']';
        }
        std::cout<<"],\"accepted_all_probes\":"<<accepted
            <<",\"rejected_missing_output_on_a_probe\":"<<rejected
            <<",\"constant_on_probes_programs\":"<<constant
            <<",\"nonconstant_on_probes_programs\":"<<nonconstant
            <<",\"self_modification_definition\":\"an executed instruction changes a byte in M0..M5 before first OUT or stop\""
            <<",\"observed_self_modifying_programs\":"<<observed_modified
            <<",\"observed_self_modifying_scope\":\"evaluated probes only; rejected programs stop at their first missing answer\""
            <<",\"accepted_self_modifying_programs\":"<<accepted_modified
            <<",\"signature_count\":"<<ordered.size()
            <<",\"singleton_signature_count\":"<<singleton_signatures
            <<",\"singleton_candidates_with_at_least_four_probe_outputs\":"<<candidates.size()
            <<",\"verification_selection\":\"singletons with at least four probe outputs; code modification first, then probe-step sum descending, program ID ascending; export only total functions after full verification\""
            <<",\"truth_table_hash_definition\":\"FNV-1a-64, offset 14695981039346656037, prime 1099511628211, successful answer bytes in a-major then b order; fingerprint only, not an equivalence proof\""
            <<",\"fully_checked_candidate_count\":"<<verifying<<",\"signatures\":[";
        bool first=true;
        for (const Candidate& candidate:ordered) {
            if (!first) std::cout<<',';
            first=false;
            std::cout<<"{\"signature\":\""<<signature_hex(candidate.signature)
                <<"\",\"sample_count\":"<<candidate.entry.count<<",\"example\":\""
                <<hex(candidate.entry.example,12)<<"\",\"distinct_probe_outputs\":"<<candidate.distinct
                <<",\"self_modifying_sample_programs\":"<<candidate.entry.modified_programs
                <<",\"probe_steps_total\":"<<candidate.entry.probe_steps_total<<'}';
        }
        std::cout<<"],\"verified_candidates\":[";
        size_t passed=0;
        std::vector<uint64_t> rejected_candidates;
        for (size_t i=0;i<verifying;++i) {
            if (write_verification(candidates[i],passed>0)) ++passed;
            else rejected_candidates.push_back(candidates[i].entry.example);
        }
        const double elapsed=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-started).count();
        std::cout<<"],\"verified_total_function_count\":"<<passed
            <<",\"rejected_full_verification_programs\":[";
        for (size_t i=0;i<rejected_candidates.size();++i) {
            if (i) std::cout<<',';
            std::cout<<'\"'<<hex(rejected_candidates[i],12)<<'\"';
        }
        std::cout<<"],\"total_seconds\":"<<elapsed<<"}\n";
        return std::cout?0:3;
    } catch (const std::exception& error) {
        std::cerr<<"phenotype6_sample: "<<error.what()<<'\n';
        return 2;
    }
}
