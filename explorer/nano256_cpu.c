/* Experimental NANO-256 CPU interpreter. See docs/NANO256.md.
 * CLI: nano256_cpu HEX [steps [first_output [address=value ...]]]
 * Batch: nano256_cpu --batch; one CLI argument line per request on stdin.
 * Code 1..16 bytes; data 256 bytes; steps 1..65535, default 512.
 * This file is independent of the running 16-byte NANO explorers.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>

static uint32_t hash_byte(uint32_t h, unsigned b) { return (h ^ (b & 255)) * 16777619u; }
static uint32_t hash_word(uint32_t h, uint32_t n) {
    for (unsigned i = 0; i < 4; i++) h = hash_byte(h, n >> (8 * i));
    return h;
}
static int number(const char *s, unsigned limit, unsigned *out) {
    char *end; errno = 0;
    if (!s[0] || s[0] == '-') return 0;
    unsigned long n = strtoul(s, &end, 0);
    if (errno || *end || n > limit) return 0;
    *out = (unsigned)n; return 1;
}
static int nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    c = (char)tolower((unsigned char)c);
    return c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
}
static void array_json(const uint8_t *v, unsigned n) {
    putchar('[');
    for (unsigned i = 0; i < n; i++) printf("%s%u", i ? "," : "", v[i]);
    putchar(']');
}
static int run_request(int argc, char **argv) {
    uint8_t m[256] = {0}, seen[256], original[16], out[8] = {0};
    unsigned initial_set[256] = {0};
    unsigned length = 0, steps_limit = 512, first = 0;
    if (argc < 1) return 0;
    const char *hex = argv[0]; int high = -1;
    for (const char *s = hex; *s; s++) {
        if (isspace((unsigned char)*s)) continue;
        int n = nibble(*s);
        if (n < 0) return 0;
        if (high < 0) high = n;
        else { if (length == 16) return 0; original[length] = m[length] = (uint8_t)((high << 4) | n); length++; high = -1; }
    }
    if (!length || high >= 0) return 0;
    if (argc > 1 && (!number(argv[1], 65535, &steps_limit) || !steps_limit)) return 0;
    if (argc > 2 && !number(argv[2], 1, &first)) return 0;
    for (int i = 3; i < argc; i++) {
        char *equal = strchr(argv[i], '='); unsigned address, value;
        if (!equal) return 0;
        *equal = '\0';
        int valid = number(argv[i], 255, &address) && number(equal + 1, 255, &value);
        *equal = '=';
        if (!valid || address < length || initial_set[address]) return 0;
        initial_set[address] = 1; m[address] = (uint8_t)value;
    }
    memcpy(seen, m, sizeof m);
    unsigned a = 0, bank = 0, pc = 0, steps = 0, count = 0;
    unsigned ta = 0, tb = 0, halted = 0, modified = 0;
    const char *reason = "step_limit";
    while (steps < steps_limit) {
        unsigned ins = m[pc], op = ins >> 4, n = ins & 15, address = bank * 16 + n;
        pc = (pc + 1) % length; steps++;
        int write = -1, emitted = 0;
        switch (op) {
        case 0: bank = n; break;
        case 1: a = n; break;
        case 2: a = (a + n) & 255; break;
        case 3: a = (a - n) & 255; break;
        case 4: a = m[address]; break;
        case 5: write = (int)a; break;
        case 6: a = (a + m[address]) & 255; break;
        case 7: pc = n % length; break;
        case 8: if (!a) pc = n % length; break;
        case 9: write = (m[address] + 1) & 255; break;
        case 10: write = (m[address] - 1) & 255; break;
        case 11: { unsigned k = n & 7; a = ((a << k) | (a >> (8 - k))) & 255; break; }
        case 12: a ^= m[address]; break;
        case 13: if (count < 8) out[count] = m[address]; count++; emitted = 1; break;
        case 14: write = m[address] ^ 255; break;
        case 15: halted = 1; reason = "halt"; break;
        }
        if (write >= 0) {
            if (address < length && m[address] != (unsigned)write) modified = 1;
            m[address] = (uint8_t)write; seen[address] |= m[address];
        }
        ta |= a; tb |= bank;
        if (halted) break;
        if (first && emitted) { reason = "first_output"; break; }
    }
    unsigned draw = 0, code_changed = 0, active = ta | tb;
    for (unsigned i = 8; i < 16; i++) draw |= seen[i];
    for (unsigned i = 0; i < length; i++) code_changed |= m[i] != original[i];
    for (unsigned i = length; i < 8; i++) active |= seen[i];
    for (unsigned i = 16; i < 256; i++) active |= seen[i];
    unsigned cls = draw ? 6 : count ? 5 : code_changed ? 4 : active ? 3 : halted ? 2 : 1;
    const char *classes[] = {"IDLE", "HALT", "ACTIVE", "SELF-MOD", "OUTPUT", "DRAW"};
    uint32_t h = 2166136261u;
    for (const char *s = "NANO256v1"; *s; s++) h = hash_byte(h, (unsigned char)*s);
    unsigned fields[] = {length, a, bank, pc, halted, ta, tb, cls, modified};
    for (unsigned i = 0; i < sizeof fields / sizeof fields[0]; i++) h = hash_byte(h, fields[i]);
    h = hash_word(h, steps); h = hash_word(h, count);
    for (unsigned i = 0; i < 256; i++) h = hash_byte(h, m[i]);
    for (unsigned i = 0; i < 256; i++) h = hash_byte(h, seen[i]);
    for (unsigned i = 0; i < 8; i++) h = hash_byte(h, out[i]);
    printf("{\"isa\":\"nano256-bank-v1\",\"program_hex\":\"");
    for (unsigned i = 0; i < length; i++) printf("%02X", original[i]);
    printf("\",\"memory_bytes\":256,\"max_steps\":%u,\"steps\":%u,\"a\":%u,\"bank\":%u,\"pc\":%u,"
           "\"halted\":%s,\"stop_reason\":\"%s\",\"class\":\"%s\",\"self_modified\":%s,"
           "\"output_count\":%u,\"outputs_first8\":", steps_limit, steps, a, bank, pc,
           halted ? "true" : "false", reason, classes[cls - 1], modified ? "true" : "false", count);
    array_json(out, count < 8 ? count : 8);
    printf(",\"memory\":"); array_json(m, 256);
    printf(",\"seen_memory_bits\":"); array_json(seen, 256);
    printf(",\"state_hash\":%u}\n", h); fflush(stdout);
    return 1;
}
int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--batch") == 0) {
        char line[8192];
        while (fgets(line, sizeof line, stdin)) {
            char *parts[260]; int count = 0;
            for (char *s = strtok(line, " \t\r\n"); s; s = strtok(NULL, " \t\r\n")) {
                if (count == 260) { fprintf(stderr, "Too many fields\n"); return 2; }
                parts[count++] = s;
            }
            if (!count) continue;
            if (!run_request(count, parts)) { fprintf(stderr, "Invalid batch request\n"); return 2; }
        }
        return ferror(stdin) ? 2 : 0;
    }
    if (argc < 2 || !run_request(argc - 1, argv + 1)) {
        fprintf(stderr, "usage: nano256_cpu HEX [steps [first_output [address=value ...]]] | --batch\n");
        return 2;
    }
    return 0;
}
