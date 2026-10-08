"""Summarize the complete class map and independently verify sampled functions.

This does not claim an exhaustive fine-phenotype catalogue or global rarity.
Inputs occupy M6 and M7; functions return their first OUT within 64 steps.
"""
import argparse
import collections
import hashlib
import json
import pathlib
import time

import demo
import nano_ref

ROOT = pathlib.Path(__file__).resolve().parent
CLASS_NAMES = ['IDLE', 'HALT', 'ACTIVE', 'SELF_MOD', 'OUTPUT', 'DRAW']
PROBES = [(3,5),(200,13),(17,250),(0,0),(255,1),(91,77),(128,128),(44,210),
          (7,100),(160,33),(66,66),(1,254),(230,9),(99,140),(12,31),(181,2)]


def first(code, a, b):
    return demo.run(code, {6: a, 7: b})[0]


def truth_fnv(table):
    h = 14695981039346656037
    for value in table:
        h = ((h ^ value) * 1099511628211) & ((1 << 64) - 1)
    return f'{h:016x}'


def census():
    manifest = ROOT / 'results/L6/chunks.jsonl'
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    assert len(rows) == 65536 and {r['chunk'] for r in rows} == set(range(65536))
    counts = [0] * 6
    for row in rows:
        assert sum(row['counts']) == 1 << 32
        for i, value in enumerate(row['counts']):
            counts[i] += value
    assert sum(counts) == 1 << 48
    return rows, {
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'programs': 1 << 48, 'chunks': len(rows),
        'classes': [{'class':name,'programs':value,'percent':100 * value / (1 << 48)} for name,value in zip(CLASS_NAMES,counts)],
        'scope': 'Complete zero-input 16-byte NANO map, 64 steps; mutually exclusive priority classes, not fine phenotypes.',
    }


def verify_program(code, output, locations):
    code = code.upper()
    number = int(code, 16)
    started = time.perf_counter()
    values = [first(code, a, b) for a in range(256) for b in range(256)]
    assert None not in values, f'{code} does not return on every input pair'
    table = bytes(values)
    hist = collections.Counter(values)
    depends_a = any(table[a*256+b] != table[b] for a in range(1,256) for b in range(256))
    depends_b = any(table[a*256+b] != table[a*256] for a in range(256) for b in range(1,256))
    neutral = []
    witnesses = []
    for bit in range(48):
        mutated = number ^ (1 << bit)
        mutated_code = f'{mutated:012X}'
        witness = None
        for a,b in PROBES:
            answer = first(mutated_code, a, b)
            if answer != table[a*256+b]:
                witness = {'bit':bit,'a':a,'b':b,'original':table[a*256+b],'mutated':answer}
                break
        if witness is None:
            for a in range(256):
                for b in range(256):
                    answer = first(mutated_code, a, b)
                    if answer != table[a*256+b]:
                        witness = {'bit':bit,'a':a,'b':b,'original':table[a*256+b],'mutated':answer}
                        break
                if witness is not None:
                    break
        if witness is None:
            neutral.append(mutated_code)
        else:
            witnesses.append(witness)
    trace_cases = []
    for a,b in [(0,0),(3,5),(17,250),(255,255)]:
        answer, trace = demo.run(code, {6:a,7:b}, trace=True)
        trace_cases.append({'a':a,'b':b,'answer':answer,'steps':len(trace),'trace':trace})
    filename = f'{code.lower()}-truth.bin'
    (output / filename).write_bytes(table)
    original_class, original_hash = nano_ref.run(number, 6)
    record = locations[number >> 32]
    return {
        'program':code,
        'disassembly':[f'{demo.OPS[value>>4]} {value&15:X}' for value in bytes.fromhex(code)],
        'function_definition':'First OUT at or before 64 steps; a=M6, b=M7, other non-code memory zero, A=0.',
        'full_input_pairs':65536, 'missing_answers':0,
        'depends_a':depends_a,'depends_b':depends_b,
        'distinct_outputs':len(hist),'most_common_output_count':max(hist.values()),
        'a_0_to_15_b_0':[table[a*256] for a in range(16)],
        'a_0_b_0_to_15':list(table[:16]),
        'truth_table_file':filename,'truth_table_order':'a-major, b-minor; one byte per pair',
        'truth_table_sha256':hashlib.sha256(table).hexdigest(),'truth_table_fnv1a64':truth_fnv(table),
        'sensitive_bits':len(witnesses),'neutral_one_bit_neighbors':neutral,'mutation_witnesses':witnesses,
        'example_traces':trace_cases,
        'original_zero_input_map':{'class_code':original_class,'class':CLASS_NAMES[original_class-1],
            'state_hash':original_hash,'node':record['node'],'file':record['file'],'byte_offset':number & 0xffffffff},
        'verification_seconds':round(time.perf_counter()-started,6),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=pathlib.Path,default=ROOT/'results/L6_phenotypes')
    parser.add_argument('--programs',nargs='*',default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows, summary = census()
    locations = {r['chunk']:r for r in rows}
    (args.output/'full-map-census.json').write_text(json.dumps(summary,indent=2)+'\n')
    reports = []
    for code in args.programs:
        report = verify_program(code,args.output,locations)
        reports.append(report)
        (args.output/f'{code.lower()}-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ['program','distinct_outputs','depends_a','depends_b','sensitive_bits','truth_table_fnv1a64','verification_seconds']}),flush=True)
    print(json.dumps({'total_programs':summary['programs'],'classes':summary['classes'],'verified_functions':len(reports)}))


if __name__ == '__main__':
    main()
