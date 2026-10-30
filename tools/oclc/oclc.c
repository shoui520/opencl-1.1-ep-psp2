/*
 * oclc: drive IMG's OpenCL C front end (liboclcompiler.so, DDK 1.12, ARMv7
 * musl) to turn OpenCL C into UniFlex, which the libusc.so shim compiles for
 * the Vita.  Runs under qemu-arm on the host.
 *
 *   oclc [-o out.bin] [-O "build options"] [-x] [-n] [-k] kernel.cl
 *
 * The source is compiled after prelude.cl (-n leaves it out), and the front
 * end's bugs are patched in memory (-k leaves it as it is).  The structures
 * below were recovered from libPVROCL's use of the compiler.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static const char prelude[] =
#include "prelude.inc"
	;

typedef struct {
	uint32_t flags;		/* libPVROCL passes a non-zero value */
	uint32_t reserved;
	void *usc;		/* set by OCLInitCompiler */
	void *front_end;	/* set by OCLCompilePrograms */
} ocl_compiler;

typedef struct {
	ocl_compiler *compiler;
	const char **sources;
	uint32_t source_count;
	uint32_t source_length;
	uint32_t unk10;
	const char *options;
	uint32_t unk18;
	uint32_t unk1c;
	uint32_t unk20;
	uint32_t unk24;
	uint32_t unk28;		/* libPVROCL passes a non-zero value */
	uint32_t unk2c;
} ocl_compile_args;

typedef struct {
	uint32_t kernel_count;
	uint8_t *kernels;	/* KERNEL_RECORD_SIZE bytes each */
	uint32_t unk8;
	uint32_t unkc;
	uint32_t unk10;
} ocl_kernel_group;

#define KERNEL_RECORD_SIZE	0x74

typedef struct {
	uint32_t group_count;
	ocl_kernel_group *groups;
	uint32_t unk8;
	int32_t status;
	char *log;
} ocl_compile_result;

typedef int (*init_fn)(ocl_compiler *);
typedef ocl_compile_result *(*compile_fn)(ocl_compile_args *);

static char *read_file(const char *path, size_t *len)
{
	FILE *f = fopen(path, "rb");
	char *buf;
	long n;

	if (!f)
		return NULL;
	fseek(f, 0, SEEK_END);
	n = ftell(f);
	fseek(f, 0, SEEK_SET);
	buf = calloc(1, n + 1);
	if (buf && fread(buf, 1, n, f) != (size_t)n) {
		free(buf);
		buf = NULL;
	}
	fclose(f);
	*len = n;
	return buf;
}

/*
 * The USC code generator's getelementptr visitor (0x8abf60) folds a constant
 * offset into the GEP's users as "StaticOffset" metadata and gives the GEP the
 * base's register.  PHIs and calls get inttoptr(ptrtoint(gep) + offset)
 * instead.  Only loads, stores (of the address) and GEPs read the metadata,
 * and a bitcast only for the load or store using it, so (size_t)&p[3],
 * atomic_inc(&s->c) (a call through a bitcast), c ? &a[1] : &b[2] and
 * *q = &p[1] lost the offset.  A hook at 0x8ac180 sends every other user down
 * the PHI/call path, and that path (0x8ac1ec) replaces the operand in any
 * user, not just PHIs and calls.
 */
#define FE_GEP_PUSH	0x8ac0e8	/* user takes the metadata */
#define FE_GEP_MATERIALISE 0x8ac18c	/* user is collected for the explicit add */
#define FE_USE_GET_USER	0xc00680	/* llvm::Use::getUser */

/* LLVM 3.0 Value/User layout and value IDs (InstructionVal + opcode) */
#define V_ID(v)		(((const uint8_t *)(v))[4])
#define V_USES(v)	(*(void *const *)((const uint8_t *)(v) + 0xc))
#define USE_NEXT(u)	(*(void *const *)((const uint8_t *)(u) + 4))
#define USER_OP(u, i)	(*(void *const *)(*(const uint8_t *const *)((const uint8_t *)(u) + 0x14) + 12 * (i)))
enum { V_LOAD = 0x30, V_STORE = 0x31, V_GEP = 0x32, V_BITCAST = 0x41 };

static uintptr_t fe_base;

static int reads_static_offset(const void *user, const void *v, int via_cast)
{
	void *(*get_user)(const void *) = (void *(*)(const void *))(fe_base + FE_USE_GET_USER);
	const void *u;

	switch (V_ID(user)) {
	case V_LOAD:
		return 1;
	case V_STORE:
		return USER_OP(user, 0) != v;
	case V_GEP:
		return !via_cast;
	case V_BITCAST:
		if (via_cast)
			return 0;
		for (u = V_USES(user); u; u = USE_NEXT(u))
			if (!reads_static_offset(get_user(u), user, 1))
				return 0;
		return 1;
	}
	return 0;
}

/* Called from gep_hook with the user and the GEP (r0, r4): where to continue. */
__attribute__((used)) static uintptr_t gep_user_target(const void *user, const void *gep)
{
	return fe_base + (reads_static_offset(user, gep, 0) ? FE_GEP_PUSH : FE_GEP_MATERIALISE);
}

void gep_hook(void);
__asm__(
	"	.text\n"
	"	.arm\n"
	"	.align 2\n"
	"	.type gep_hook, %function\n"
	"gep_hook:\n"
	"	sub sp, sp, #8\n"
	"	push {r0-r3, ip, lr}\n"
	"	mov r1, r4\n"
	"	bl gep_user_target\n"
	"	str r0, [sp, #24]\n"
	"	pop {r0-r3, ip, lr}\n"
	"	ldr pc, [sp], #8\n"
	"	.size gep_hook, . - gep_hook\n");

/*
 * "Memory movement optimisation" (a module pass in the 1.12 front end) loses
 * stores: after "a = g[0]; g[0] = a + 1; h[1] = 9;" a later read of g[0]
 * gives a.  Make its runOnModule return false.
 */
#define HOOK	1	/* code[1] is replaced by gep_hook's address */
static const struct fe_patch {
	const char *what;
	uint32_t offset;
	uint32_t expect[2];
	uint32_t code[2];
} fe_patches[] = {
	{ "memory movement optimisation", 0x8fd8c8,
	  { 0xe92d47f0, 0xe241702c },	/* push {r4-sl, lr}; sub r7, r1, #44 */
	  { 0xe3a00000, 0xe12fff1e } },	/* mov r0, #0; bx lr */
	{ "GEP offset folding", 0x8ac180,
	  { 0xe2433044, 0xe3530001 },	/* sub r3, r3, #0x44; cmp r3, #1 */
	  { 0xe51ff004, HOOK } },	/* ldr pc, [pc, #-4]; .word gep_hook */
	{ "GEP offset folding", 0x8ac1ec,
	  { 0xe3530044, 0x0a0000c5 },	/* cmp r3, #0x44; beq 0x8ac50c */
	  { 0xea0000c6, 0x0a0000c5 } },	/* b 0x8ac50c */
};

static int patch_front_end(void *sym)
{
	long page = sysconf(_SC_PAGESIZE);
	Dl_info info;
	unsigned i;

	if (!dladdr(sym, &info))
		return -1;
	fe_base = (uintptr_t)info.dli_fbase;
	for (i = 0; i < sizeof(fe_patches) / sizeof(fe_patches[0]); i++) {
		const struct fe_patch *p = &fe_patches[i];
		uint32_t *at = (uint32_t *)((uint8_t *)info.dli_fbase + p->offset);
		uintptr_t start = (uintptr_t)at & ~(uintptr_t)(page - 1);
		size_t len = (uintptr_t)(at + 2) - start;
		uint32_t code[2] = { p->code[0], p->code[1] };

		if (memcmp(at, p->expect, sizeof(p->expect))) {
			fprintf(stderr, "oclc: unexpected front end build, cannot patch %s\n", p->what);
			return -1;
		}
		if (code[1] == HOOK)
			code[1] = (uint32_t)(uintptr_t)gep_hook;
		if (mprotect((void *)start, len, PROT_READ | PROT_WRITE | PROT_EXEC))
			return -1;
		memcpy(at, code, sizeof(code));
		__builtin___clear_cache((char *)at, (char *)(at + 2));
		mprotect((void *)start, len, PROT_READ | PROT_EXEC);
	}
	return 0;
}

static void hexdump(const void *p, size_t n)
{
	const uint32_t *w = p;
	size_t i;

	for (i = 0; i < n / 4; i++)
		printf("%s%08X", i % 8 ? " " : (i ? "\n    " : "    "), w[i]);
	printf("\n");
}

/*
 * -m: one line per kernel, in the order the kernels were compiled, then one
 * per argument with the 12 words of its argument table entry:
 *
 *   kernel <name> <argument count> <program index, -1 if unknown>
 *   arg <index> <word0> ... <word11>
 *   private <word0> <bytes per work-item> <constant channel of the base>
 *
 * (word2: address space, 0 value, 1 local, 2 global, 3 constant; word3: the
 * constant register; word4: element size; word5: components; word6: pointer)
 *
 * A private array the front end keeps in memory is at base + (get_global_id(1)
 * * get_global_size(0) + get_global_id(0)) * bytes; its record (3 words each,
 * kernel record words 10 and 11) names the channel the runtime puts base in.
 */
static int (*program_index)(const void *);

/* Which compiled program (prefix.N.* of usc_shim) a kernel record is. */
static int record_program(const uint8_t *rec)
{
	const uint32_t *w = (const uint32_t *)rec;
	unsigned i;
	int n;

	if (!program_index) {
		void *h = dlopen("libusc.so", RTLD_NOW | RTLD_NOLOAD);

		program_index = h ? (int (*)(const void *))dlsym(h, "usc_shim_program_index") : NULL;
		if (!program_index)
			return -1;
	}
	for (i = 0; i < KERNEL_RECORD_SIZE / 4; i++) {
		n = program_index((const void *)(uintptr_t)w[i]);
		if (n >= 0)
			return n;
	}
	return -1;
}

static void write_meta(FILE *f, const uint8_t *rec)
{
	const uint32_t *w = (const uint32_t *)rec;
	const uint32_t *a = (const uint32_t *)(uintptr_t)w[3];
	uint32_t i, j;

	fprintf(f, "kernel %s %u %d\n", *(char **)rec ? *(char **)rec : "?", w[2], record_program(rec));
	for (i = 0; i < w[2]; i++) {
		fprintf(f, "arg %u", i);
		for (j = 0; j < 12; j++)
			fprintf(f, " %u", a[i * 12 + j]);
		fprintf(f, "\n");
	}
	a = (const uint32_t *)(uintptr_t)w[11];
	for (i = 0; a && i < w[10]; i++)
		fprintf(f, "private %u %u %u\n", a[3 * i], a[3 * i + 1], a[3 * i + 2]);
}

int main(int argc, char **argv)
{
	const char *lib = "liboclcompiler.so", *options = "", *path = NULL;
	ocl_compiler compiler;
	ocl_compile_args args;
	ocl_compile_result *res;
	init_fn init;
	compile_fn compile;
	char *src;
	size_t len;
	void *h;
	int i, verbose = 0, use_prelude = 1, keep_passes = 0;
	const char *meta_path = NULL;
	FILE *meta = NULL;
	uint32_t g, k;

	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-O") && i + 1 < argc)
			options = argv[++i];
		else if (!strcmp(argv[i], "-L") && i + 1 < argc)
			lib = argv[++i];
		else if (!strcmp(argv[i], "-x"))
			verbose = 1;
		else if (!strcmp(argv[i], "-n"))
			use_prelude = 0;
		else if (!strcmp(argv[i], "-k"))
			keep_passes = 1;
		else if (!strcmp(argv[i], "-m") && i + 1 < argc)
			meta_path = argv[++i];
		else
			path = argv[i];
	}
	if (!path) {
		fprintf(stderr, "usage: oclc [-O options] [-L liboclcompiler.so] [-x] [-n] [-k] [-m meta] kernel.cl\n");
		return 2;
	}
	src = read_file(path, &len);
	if (!src) {
		perror(path);
		return 1;
	}
	if (use_prelude) {
		static const char line[] = "\n#line 1\n";
		char *all = malloc(sizeof(prelude) - 1 + sizeof(line) - 1 + len + 1);

		memcpy(all, prelude, sizeof(prelude) - 1);
		memcpy(all + sizeof(prelude) - 1, line, sizeof(line) - 1);
		memcpy(all + sizeof(prelude) - 1 + sizeof(line) - 1, src, len + 1);
		len += sizeof(prelude) - 1 + sizeof(line) - 1;
		free(src);
		src = all;
	}

	h = dlopen(lib, RTLD_NOW);
	if (!h) {
		fprintf(stderr, "%s\n", dlerror());
		return 1;
	}
	init = (init_fn)dlsym(h, "OCLInitCompiler");
	compile = (compile_fn)dlsym(h, "OCLCompilePrograms");
	if (!keep_passes && patch_front_end((void *)compile))
		return 1;

	memset(&compiler, 0, sizeof(compiler));
	compiler.flags = 95;
	if (!init(&compiler)) {
		fprintf(stderr, "OCLInitCompiler failed\n");
		return 1;
	}

	memset(&args, 0, sizeof(args));
	args.compiler = &compiler;
	args.sources = (const char **)&src;
	args.source_count = 1;
	args.source_length = len;
	args.options = options;
	args.unk28 = 95;
	res = compile(&args);
	if (!res) {
		fprintf(stderr, "OCLCompilePrograms returned NULL\n");
		return 1;
	}
	printf("status %d, %u groups\n", res->status, res->group_count);
	if (meta_path && !res->status) {
		meta = fopen(meta_path, "w");
		if (!meta) {
			perror(meta_path);
			return 1;
		}
	}
	if (res->log && *res->log)
		printf("log:\n%s\n", res->log);
	for (g = 0; g < res->group_count; g++) {
		ocl_kernel_group *grp = &res->groups[g];

		printf("group %u: %u kernels, %08X %08X %08X\n", g, grp->kernel_count, grp->unk8, grp->unkc,
		       grp->unk10);
		for (k = 0; k < grp->kernel_count; k++) {
			uint8_t *rec = grp->kernels + k * KERNEL_RECORD_SIZE;

			printf("  kernel %u: %s\n", k, *(char **)rec ? *(char **)rec : "?");
			if (meta)
				write_meta(meta, rec);
			if (verbose) {
				const uint32_t *w = (const uint32_t *)rec;
				unsigned i;

				hexdump(rec, KERNEL_RECORD_SIZE);
				/* the argument table: 0x30 bytes per argument */
				printf("    args:\n");
				hexdump((const void *)(uintptr_t)w[3], w[2] * 0x30);
				/* tables of unknown layout: w[10] entries at w[11], w[25] at w[26] */
				if (w[10] && w[11]) {
					printf("    w11 (%u):\n", w[10]);
					hexdump((const void *)(uintptr_t)w[11], 0x40);
				}
				if (w[25] && w[26]) {
					printf("    w26 (%u):\n", w[25]);
					hexdump((const void *)(uintptr_t)w[26], 0x40);
				}
			}
		}
	}
	if (meta)
		fclose(meta);
	return res->status ? 1 : 0;
}
