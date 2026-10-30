/*
 * usedis: disassemble USSE instructions with PVR_PSP2's useasm tables.
 *
 *   usedis [-t id:rev] file offset count
 *   usedis [-t id:rev] -           (stdin lines: "HI LO [note]", "# text")
 *
 * Reads count 64-bit instructions (two little-endian words each) starting
 * at byte offset in file, or word pairs from stdin.  The default target is
 * the Vita's SGX543 rev 216.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sgxdefs.h"
#include "use.h"
#include "usedisasm.h"

int main(int argc, char **argv)
{
	SGX_CORE_INFO target = { SGX_CORE_ID_543, 216 };
	PCSGX_CORE_DESC desc;
	unsigned id, rev, i;
	long offset, count;
	FILE *f;
	int a = 1;

	if (argc > 2 && !strcmp(argv[1], "-t") && sscanf(argv[2], "%u:%u", &id, &rev) == 2) {
		target.eID = id;
		target.uiRev = rev;
		a = 3;
	}
	if (argc - a == 1 && !strcmp(argv[a], "-")) {
		char line[512], note[256];
		IMG_UINT32 w[2];

		desc = UseAsmGetCoreDesc(&target);
		if (!desc)
			return 1;
		i = 0;
		while (fgets(line, sizeof(line), stdin)) {
			char text[1024];

			if (line[0] == '#') {
				fputs(line, stdout);
				continue;
			}
			note[0] = 0;
			if (sscanf(line, "%x %x %255[^\n]", &w[1], &w[0], note) < 2)
				continue;
			text[0] = 0;
			UseDisassembleInstruction(desc, w[0], w[1], text);
			printf("%3u: %08X %08X  %-60s %s\n", i++, w[1], w[0], text, strcmp(note, "code") ? note : "");
		}
		return 0;
	}
	if (argc - a != 3) {
		fprintf(stderr, "usage: usedis [-t id:rev] file offset count\n");
		return 2;
	}
	offset = strtol(argv[a + 1], NULL, 0);
	count = strtol(argv[a + 2], NULL, 0);
	desc = UseAsmGetCoreDesc(&target);
	if (!desc) {
		fprintf(stderr, "unknown target %u:%u\n", target.eID, target.uiRev);
		return 1;
	}
	f = fopen(argv[a], "rb");
	if (!f || fseek(f, offset, SEEK_SET)) {
		perror(argv[a]);
		return 1;
	}
	for (i = 0; i < count; i++) {
		IMG_UINT32 w[2];
		char text[1024];

		if (fread(w, 4, 2, f) != 2)
			break;
		text[0] = 0;
		UseDisassembleInstruction(desc, w[0], w[1], text);
		printf("%3u: %08X %08X  %s\n", i, w[1], w[0], text);
	}
	return 0;
}
