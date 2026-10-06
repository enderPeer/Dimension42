/* Count class bytes read from stdin (e.g. zstd -dc chunk.zst | ./nano_count).
 * Prints: bytes=N counts=c1,..,c6 bad=B   (bad = bytes that are not a valid class 1..6) */
#include <stdio.h>

int main(void)
{
    static unsigned char buf[1 << 20];
    unsigned long long hist[256] = {0}, total = 0;
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, stdin)) > 0) {
        for (size_t i = 0; i < n; i++) hist[buf[i]]++;
        total += n;
    }
    unsigned long long bad = total;
    for (int k = 1; k <= 6; k++) bad -= hist[k];
    printf("bytes=%llu counts=%llu,%llu,%llu,%llu,%llu,%llu bad=%llu\n",
           total, hist[1], hist[2], hist[3], hist[4], hist[5], hist[6], bad);
    return 0;
}
