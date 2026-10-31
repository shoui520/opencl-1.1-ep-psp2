/*
 * libusc.so replacement loaded under IMG's OpenCL front end (liboclcompiler).
 *
 * The front end hands every kernel to PVRUniFlexCompileToUspBin as a UniFlex
 * program.  This library contains PVR_PSP2's USC (built with its entry points
 * renamed psp2_*) and retargets each program to the Vita's SGX543MP4+ before
 * compiling it.  Environment:
 *
 *   OCLC_USC=img        forward everything to IMG's own USC instead
 *                       (OCLC_IMG_USC=path, default libusc_img.so)
 *   OCLC_DUMP=prefix    write prefix.N.uf (decoded UniFlex), prefix.N.params
 *                       (raw program parameters) and prefix.N.usp (output)
 *   OCLC_TARGET=id:rev  compile for another core (e.g. 8:115, the SGX544 the
 *                       front end asks for) instead of the Vita's
 *   OCLC_RAW=1          also list the raw operand fields in the .uf dump
 *   OCLC_GXM=1          retarget the kernel for dispatch as a GXM vertex
 *                       program (see gxm_retarget) and, with OCLC_DUMP, write
 *                       the USP-finalised hardware shader to prefix.N.hw
 *   OCLC_VERBOSE=1      log each compile to stderr
 */
#include <dlfcn.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "usc.h"
#include "usp.h"

#define VITA_SGX_REV	216
#define VITA_SGX_CORES	4

/* UNIFLEX_PROGRAM_PARAMETERS is laid out identically in the 1.12 front end
 * and this USC up to the end of the constant buffer descriptors.  The 1.12
 * structure has one more descriptor; the OpenCL front end leaves the fields
 * after it zero. */
#define PARAMS_COMMON_SIZE	offsetof(UNIFLEX_PROGRAM_PARAMETERS, auFlatShadedTextureCoordinates)
#define PARAMS_IMG_SIZE		(sizeof(UNIFLEX_PROGRAM_PARAMETERS) + sizeof(UNIFLEX_CONSTBUFFERDESC))

IMG_PVOID psp2_PVRUniFlexCreateContext(USC_ALLOCFN, USC_FREEFN, USC_PRINTFN, USC_PDUMPFN, IMG_PVOID,
				       USC_STARTFN, USC_FINISHFN, IMG_PVOID);
IMG_UINT32 psp2_PVRUniFlexCompileToUspBin(IMG_PVOID, IMG_UINT32 *, PUNIFLEX_INST, PUNIFLEX_CONSTDEF,
					  PUNIFLEX_PROGRAM_PARAMETERS, PUSC_USP_PC_SHADER *);
IMG_VOID psp2_PVRUniFlexDestroyUspBin(IMG_PVOID, PUSC_USP_PC_SHADER);
IMG_VOID psp2_PVRUniFlexDestroyContext(IMG_PVOID);

static struct {
	IMG_PVOID (*create)(USC_ALLOCFN, USC_FREEFN, USC_PRINTFN, USC_PDUMPFN, IMG_PVOID,
			    USC_STARTFN, USC_FINISHFN, IMG_PVOID);
	IMG_UINT32 (*compile)(IMG_PVOID, IMG_UINT32 *, PUNIFLEX_INST, PUNIFLEX_CONSTDEF,
			      PUNIFLEX_PROGRAM_PARAMETERS, PUSC_USP_PC_SHADER *);
	IMG_VOID (*destroy_bin)(IMG_PVOID, PUSC_USP_PC_SHADER);
	IMG_VOID (*destroy)(IMG_PVOID);
} img;

static int use_img(void)
{
	static int state = -1;
	const char *mode, *path;
	void *h;

	if (state >= 0)
		return state;
	mode = getenv("OCLC_USC");
	state = mode && !strcmp(mode, "img");
	if (!state)
		return 0;
	path = getenv("OCLC_IMG_USC");
	h = dlopen(path ? path : "libusc_img.so", RTLD_NOW | RTLD_LOCAL);
	if (!h) {
		fprintf(stderr, "usc_shim: %s\n", dlerror());
		exit(1);
	}
	img.create = dlsym(h, "PVRUniFlexCreateContext");
	img.compile = dlsym(h, "PVRUniFlexCompileToUspBin");
	img.destroy_bin = dlsym(h, "PVRUniFlexDestroyUspBin");
	img.destroy = dlsym(h, "PVRUniFlexDestroyContext");
	return 1;
}

/*
 * Kernel dispatch draws one point per work-item.  The front end reads
 * get_global_id(0/1) from vi0.x/y; GXM puts the vertex and instance indices
 * of the draw there (small numbers, the same for every chunk of a dispatch),
 * so add a per-draw base from c18.xy, which the front end leaves unused.
 * The point's position (o0) comes from c19 and lies outside the clip volume.
 */
#define GXM_ID_BASE_CONST	18
#define GXM_POSITION_CONST	19

static void set_reg(UF_REGISTER *r, UF_REGTYPE type, IMG_UINT32 num, UF_REGFORMAT fmt)
{
	memset(r, 0, sizeof(*r));
	r->eType = type;
	r->uNum = num;
	r->eFormat = fmt;
	r->u.uSwiz = UFREG_SWIZ_NONE;
}

static PUNIFLEX_INST gxm_retarget(PUNIFLEX_INST prog, PUNIFLEX_CONSTDEF consts,
				  UNIFLEX_PROGRAM_PARAMETERS *params, PUNIFLEX_INST added[2])
{
	PUNIFLEX_INST inst, add, mov;
	IMG_UINT32 temp = 0;
	int i;

	if (!consts || consts->uCount < (GXM_POSITION_CONST + 1) * 4)
		return NULL;
	for (i = GXM_ID_BASE_CONST * 4; i < (GXM_POSITION_CONST + 1) * 4; i++)
		if (consts->puConstStaticFlags[i / 32] & (1u << (i % 32)))
			return NULL;
	for (inst = prog; inst; inst = inst->psILink) {
		if (inst->sDest.eType == UFREG_TYPE_TEMP && inst->sDest.uNum >= temp)
			temp = inst->sDest.uNum + 1;
		if (inst->sDest2.eType == UFREG_TYPE_TEMP && inst->sDest2.uNum >= temp)
			temp = inst->sDest2.uNum + 1;
		for (i = 0; i < UNIFLEX_MAXIMUM_SOURCE_ARGUMENT_COUNT; i++)
			if (inst->asSrc[i].eType == UFREG_TYPE_TEMP && inst->asSrc[i].uNum >= temp)
				temp = inst->asSrc[i].uNum + 1;
	}
	for (inst = prog; inst; inst = inst->psILink) {
		for (i = 0; i < UNIFLEX_MAXIMUM_SOURCE_ARGUMENT_COUNT; i++) {
			UF_REGISTER *r = &inst->asSrc[i];

			if (r->eType == UFREG_TYPE_VSINPUT) {
				if (r->uNum != 0 || r->eRelativeIndex != UFREG_RELATIVEINDEX_NONE)
					return NULL;
				r->eType = UFREG_TYPE_TEMP;
				r->uNum = temp;
			}
		}
	}

	add = calloc(1, sizeof(*add));
	mov = calloc(1, sizeof(*mov));
	add->eOpCode = UFOP_ADD;
	set_reg(&add->sDest, UFREG_TYPE_TEMP, temp, UF_REGFORMAT_U32);
	add->sDest.u.byMask = 3;
	set_reg(&add->asSrc[0], UFREG_TYPE_VSINPUT, 0, UF_REGFORMAT_U32);
	set_reg(&add->asSrc[1], UFREG_TYPE_CONST, GXM_ID_BASE_CONST, UF_REGFORMAT_U32);
	add->uPredicate = UF_PRED_NONE;
	mov->eOpCode = UFOP_MOV;
	set_reg(&mov->sDest, UFREG_TYPE_VSOUTPUT, 0, UF_REGFORMAT_F32);
	mov->sDest.u.byMask = 0xf;
	set_reg(&mov->asSrc[0], UFREG_TYPE_CONST, GXM_POSITION_CONST, UF_REGFORMAT_F32);
	mov->uPredicate = UF_PRED_NONE;

	/* after the main program's label */
	inst = prog->eOpCode == UFOP_LABEL ? prog : NULL;
	if (inst) {
		mov->psILink = inst->psILink;
		inst->psILink = add;
	} else {
		mov->psILink = prog;
		prog = add;
	}
	add->psILink = mov;
	added[0] = add;
	added[1] = mov;

	memset(params->puValidShaderOutputs, 0, sizeof(params->puValidShaderOutputs));
	params->puValidShaderOutputs[0] = 0xf;
	return prog;
}

static IMG_UINT32 dest_regs(PUNIFLEX_INST inst)
{
	IMG_UINT32 mask = inst->sDest.u.byMask & 0xf, chans = 0;

	/* a MEMLD of n elements fills n / (channels in the mask) registers */
	if (inst->eOpCode != UFOP_MEMLD)
		return 1;
	for (; mask; mask &= mask - 1)
		chans++;
	return chans && inst->asSrc[2].uNum > chans ? inst->asSrc[2].uNum / chans : 1;
}

/*
 * The front end reads a temporary it never writes for a null pointer (in a
 * comparison like p == 0), and USC leaves such a register uninitialised.  So
 * each temporary read but never written is set to 0 after the main label: a
 * write at the start can't change what a program that writes before reading
 * computes.  Returns the program; the added MOVs are zeros[0]..zeros[1].
 */
static PUNIFLEX_INST zero_unwritten_temps(PUNIFLEX_INST prog, PUNIFLEX_INST zeros[2])
{
	PUNIFLEX_INST inst, prev = NULL;
	IMG_UINT32 count = 0, i, j;
	unsigned char *written;

	for (inst = prog; inst; inst = inst->psILink) {
		if (inst->sDest.eType == UFREG_TYPE_TEMP && inst->sDest.uNum + dest_regs(inst) > count)
			count = inst->sDest.uNum + dest_regs(inst);
		if (inst->sDest2.eType == UFREG_TYPE_TEMP && inst->sDest2.uNum >= count)
			count = inst->sDest2.uNum + 1;
		for (i = 0; i < UNIFLEX_MAXIMUM_SOURCE_ARGUMENT_COUNT; i++)
			if (inst->asSrc[i].eType == UFREG_TYPE_TEMP && inst->asSrc[i].uNum >= count)
				count = inst->asSrc[i].uNum + 1;
	}
	written = calloc(count + 1, 1);
	if (!written)
		return prog;
	for (inst = prog; inst; inst = inst->psILink) {
		PCINPUT_INST_DESC desc = PVRUniFlexGetInputInstDesc(inst->eOpCode);

		if (!desc)
			continue;
		if (desc->uNumDests >= 1 && inst->sDest.eType == UFREG_TYPE_TEMP &&
		    inst->sDest.eRelativeIndex == UFREG_RELATIVEINDEX_NONE)
			for (j = 0; j < dest_regs(inst); j++)
				written[inst->sDest.uNum + j] = 1;
		if (desc->uNumDests >= 2 && inst->sDest2.eType == UFREG_TYPE_TEMP &&
		    inst->sDest2.eRelativeIndex == UFREG_RELATIVEINDEX_NONE)
			written[inst->sDest2.uNum] = 1;
	}
	for (inst = prog; inst; inst = inst->psILink) {
		PCINPUT_INST_DESC desc = PVRUniFlexGetInputInstDesc(inst->eOpCode);

		for (i = 0; desc && i < desc->uNumSrcArgs && i < UNIFLEX_MAXIMUM_SOURCE_ARGUMENT_COUNT; i++) {
			const UF_REGISTER *r = &inst->asSrc[i];
			PUNIFLEX_INST mov;

			if (r->eType != UFREG_TYPE_TEMP || r->eRelativeIndex != UFREG_RELATIVEINDEX_NONE ||
			    written[r->uNum])
				continue;
			mov = calloc(1, sizeof(*mov));
			if (!mov)
				break;
			mov->eOpCode = UFOP_MOV;
			set_reg(&mov->sDest, UFREG_TYPE_TEMP, r->uNum, r->eFormat);
			mov->sDest.u.byMask = 0xf;
			set_reg(&mov->asSrc[0], UFREG_TYPE_IMMEDIATE, 0, r->eFormat);
			mov->uPredicate = UF_PRED_NONE;
			if (prev)
				prev->psILink = mov;
			else
				zeros[0] = mov;
			prev = mov;
			written[r->uNum] = 1;
		}
	}
	if (prev) {
		zeros[1] = prev;
		inst = prog->eOpCode == UFOP_LABEL ? prog : NULL;
		if (inst) {
			prev->psILink = inst->psILink;
			inst->psILink = zeros[0];
		} else {
			prev->psILink = prog;
			prog = zeros[0];
		}
	}
	free(written);
	return prog;
}

/*
 * The front end ends a block with the copies into the phi registers of the
 * block it jumps to, but doesn't always order them so each copy reads the
 * value from before the copies: at the end of a loop doing b = a; a = t;
 * it can copy t into a and then a (now t) into b.  A copy reads such a value
 * when its source is the destination of an earlier copy in the run that
 * wrote every channel it reads, and the register has a definition in
 * another block, so it's a phi register (a temporary assembled a channel at
 * a time, or written in this block only, is read as written).  Each such
 * source is saved in a new temporary before the run and read from there.
 */
#define MAX_PHI_SAVES	256

static struct phi_save {
	PUNIFLEX_INST mov, prev, reader;
	UF_REGISTER src;
} phi_saves[MAX_PHI_SAVES];
static int phi_save_count;

static IMG_UINT32 src_chans(const UNIFLEX_INST *inst)
{
	IMG_UINT32 c, s, mask = 0;

	for (c = 0; c < 4; c++)
		if (inst->sDest.u.byMask & (1 << c)) {
			s = (inst->asSrc[0].u.uSwiz >> (c * 3)) & 7;
			if (s < 4)
				mask |= 1 << s;
		}
	return mask;
}

static int is_plain_mov(const UNIFLEX_INST *inst)
{
	return inst->eOpCode == UFOP_MOV && inst->uPredicate == UF_PRED_NONE &&
	       inst->sDest.eType == UFREG_TYPE_TEMP && inst->sDest.eRelativeIndex == UFREG_RELATIVEINDEX_NONE &&
	       inst->sDest.byMod == 0;
}

static void order_phi_copies(PUNIFLEX_INST prog)
{
	PUNIFLEX_INST inst, prev, first, i, j;
	IMG_UINT32 count = 0, block, next_temp, k;
	IMG_INT32 *def_block;

	phi_save_count = 0;
	for (inst = prog; inst; inst = inst->psILink) {
		if (inst->sDest.eType == UFREG_TYPE_TEMP && inst->sDest.uNum + dest_regs(inst) > count)
			count = inst->sDest.uNum + dest_regs(inst);
		if (inst->sDest2.eType == UFREG_TYPE_TEMP && inst->sDest2.uNum >= count)
			count = inst->sDest2.uNum + 1;
		for (k = 0; k < UNIFLEX_MAXIMUM_SOURCE_ARGUMENT_COUNT; k++)
			if (inst->asSrc[k].eType == UFREG_TYPE_TEMP && inst->asSrc[k].uNum >= count)
				count = inst->asSrc[k].uNum + 1;
	}
	next_temp = count;
	/* the block of each temporary's definitions, -1 for none, -2 for several */
	def_block = malloc(count * sizeof(*def_block));
	if (!def_block)
		return;
	memset(def_block, 0xff, count * sizeof(*def_block));
	block = 0;
	for (inst = prog; inst; inst = inst->psILink) {
		PCINPUT_INST_DESC desc = PVRUniFlexGetInputInstDesc(inst->eOpCode);

		if (inst->eOpCode == UFOP_BLOCK || inst->eOpCode == UFOP_LABEL)
			block++;
		if (!desc || desc->uNumDests < 1 || inst->sDest.eType != UFREG_TYPE_TEMP)
			continue;
		for (k = 0; k < dest_regs(inst); k++) {
			IMG_INT32 *d = &def_block[inst->sDest.uNum + k];

			*d = *d == -1 || *d == (IMG_INT32)block ? (IMG_INT32)block : -2;
		}
	}
	block = 0;
	prev = NULL;
	for (inst = prog; inst; prev = inst, inst = inst->psILink) {
		if (inst->eOpCode == UFOP_BLOCK || inst->eOpCode == UFOP_LABEL)
			block++;
		if (!prev || !is_plain_mov(inst) || is_plain_mov(prev))
			continue;
		/* a run of copies starting at inst: does it end in a jump? */
		for (j = inst; j && is_plain_mov(j); j = j->psILink)
			;
		if (!j || j->eOpCode != UFOP_JUMP)
			continue;
		first = inst;
		for (j = first->psILink; j->eOpCode != UFOP_JUMP; j = j->psILink) {
			const UF_REGISTER *r = &j->asSrc[0];
			PUNIFLEX_INST writer = NULL, mov;
			struct phi_save *save;

			if (r->eType != UFREG_TYPE_TEMP || r->eRelativeIndex != UFREG_RELATIVEINDEX_NONE ||
			    r->uNum >= count || def_block[r->uNum] == (IMG_INT32)block)
				continue;
			for (i = first; i != j; i = i->psILink)
				if (i->sDest.uNum == r->uNum)
					writer = i;
			if (!writer || (src_chans(j) & ~writer->sDest.u.byMask & 0xf) ||
			    phi_save_count == MAX_PHI_SAVES || !(mov = calloc(1, sizeof(*mov))))
				continue;
			mov->eOpCode = UFOP_MOV;
			set_reg(&mov->sDest, UFREG_TYPE_TEMP, next_temp++, r->eFormat);
			mov->sDest.u.byMask = src_chans(j);
			set_reg(&mov->asSrc[0], UFREG_TYPE_TEMP, r->uNum, r->eFormat);
			mov->asSrc[0].u.uSwiz = UFREG_SWIZ_NONE;
			mov->uPredicate = UF_PRED_NONE;
			mov->psILink = prev->psILink;
			prev->psILink = mov;
			save = &phi_saves[phi_save_count++];
			save->mov = mov;
			save->prev = prev;
			save->reader = j;
			save->src = j->asSrc[0];
			j->asSrc[0].uNum = mov->sDest.uNum;
		}
	}
	free(def_block);
}

/* Undo order_phi_copies. */
static void restore_phi_copies(void)
{
	while (phi_save_count > 0) {
		struct phi_save *save = &phi_saves[--phi_save_count];

		save->prev->psILink = save->mov->psILink;
		save->reader->asSrc[0] = save->src;
		free(save->mov);
	}
}

/*
 * The front end reads an element of a __constant array indexed at run time
 * as I32 whatever its type, and a float one is then moved to F32, which USC
 * takes as a conversion from integer.  Such a MOV, from a relatively indexed
 * I32 or U32 constant or from a temporary only defined by a copy of one, is
 * made a REINTERP.  Returns the number made; restore_const_reads undoes it.
 */
#define MAX_CONST_READS	256

static PUNIFLEX_INST const_reads[MAX_CONST_READS];
static int const_read_count;

static int is_int_fmt(UF_REGFORMAT fmt)
{
	return fmt == UF_REGFORMAT_I32 || fmt == UF_REGFORMAT_U32;
}

static int is_indexed_const_read(const UF_REGISTER *r)
{
	return r->eType == UFREG_TYPE_CONST && r->eRelativeIndex != UFREG_RELATIVEINDEX_NONE &&
	       is_int_fmt(r->eFormat) && r->byMod == 0;
}

static int reinterpret_const_reads(PUNIFLEX_INST prog)
{
	PUNIFLEX_INST inst;
	IMG_UINT32 count = 0;
	/* per temporary: 1 if its only definition copies an indexed constant,
	 * 2 if it has any other definition */
	unsigned char *from_const;

	const_read_count = 0;
	for (inst = prog; inst; inst = inst->psILink)
		if (inst->sDest.eType == UFREG_TYPE_TEMP && inst->sDest.uNum + dest_regs(inst) > count)
			count = inst->sDest.uNum + dest_regs(inst);
	from_const = calloc(count + 1, 1);
	if (!from_const)
		return 0;
	for (inst = prog; inst; inst = inst->psILink) {
		PCINPUT_INST_DESC desc = PVRUniFlexGetInputInstDesc(inst->eOpCode);
		IMG_UINT32 k;

		if (desc && desc->uNumDests >= 2 && inst->sDest2.eType == UFREG_TYPE_TEMP &&
		    inst->sDest2.uNum < count)
			from_const[inst->sDest2.uNum] = 2;
		if (!desc || desc->uNumDests < 1 || inst->sDest.eType != UFREG_TYPE_TEMP)
			continue;
		if (inst->eOpCode == UFOP_MOV && inst->uPredicate == UF_PRED_NONE &&
		    inst->sDest.eRelativeIndex == UFREG_RELATIVEINDEX_NONE && is_int_fmt(inst->sDest.eFormat) &&
		    is_indexed_const_read(&inst->asSrc[0]) && !from_const[inst->sDest.uNum]) {
			from_const[inst->sDest.uNum] = 1;
			continue;
		}
		for (k = 0; k < dest_regs(inst); k++)
			from_const[inst->sDest.uNum + k] = 2;
	}
	for (inst = prog; inst && const_read_count < MAX_CONST_READS; inst = inst->psILink) {
		const UF_REGISTER *r = &inst->asSrc[0];

		if (inst->eOpCode != UFOP_MOV || inst->sDest.eFormat != UF_REGFORMAT_F32 ||
		    inst->sDest.byMod != 0 || r->byMod != 0 || !is_int_fmt(r->eFormat))
			continue;
		if (is_indexed_const_read(r) ||
		    (r->eType == UFREG_TYPE_TEMP && r->eRelativeIndex == UFREG_RELATIVEINDEX_NONE &&
		     r->uNum < count && from_const[r->uNum] == 1)) {
			inst->eOpCode = UFOP_REINTERP;
			const_reads[const_read_count++] = inst;
		}
	}
	free(from_const);
	return const_read_count;
}

static void restore_const_reads(void)
{
	while (const_read_count > 0)
		const_reads[--const_read_count]->eOpCode = UFOP_MOV;
}

/*
 * The front end's float to ushort_sat conversion moves the float to a U16
 * temporary, then fixes NaN up with a MOVCBIT selecting between that
 * temporary and 0 into the same register, as U32: the USC refuses a register
 * with both formats ("Using register with inconcistent formatting").  A
 * MOVCBIT whose integer destination is one of its selected sources in a
 * narrower integer format gets that format; restore_select_formats undoes it.
 */
#define MAX_SELECTS	256

static struct select_save {
	PUNIFLEX_INST inst;
	UF_REGFORMAT fmt;
} selects[MAX_SELECTS];
static int select_count;

static int is_narrow_int_fmt(UF_REGFORMAT fmt)
{
	return fmt == UF_REGFORMAT_I16 || fmt == UF_REGFORMAT_U16 ||
	       fmt == UF_REGFORMAT_I8_UN || fmt == UF_REGFORMAT_U8_UN;
}

static void narrow_select_dests(PUNIFLEX_INST prog)
{
	PUNIFLEX_INST inst;
	int k;

	select_count = 0;
	for (inst = prog; inst && select_count < MAX_SELECTS; inst = inst->psILink) {
		if (inst->eOpCode != UFOP_MOVCBIT ||
		    inst->sDest.eType != UFREG_TYPE_TEMP || !is_int_fmt(inst->sDest.eFormat))
			continue;
		for (k = 1; k <= 2; k++) {
			const UF_REGISTER *r = &inst->asSrc[k];

			if (r->eType == UFREG_TYPE_TEMP && r->uNum == inst->sDest.uNum &&
			    r->eRelativeIndex == UFREG_RELATIVEINDEX_NONE && is_narrow_int_fmt(r->eFormat)) {
				selects[select_count].inst = inst;
				selects[select_count++].fmt = inst->sDest.eFormat;
				inst->sDest.eFormat = r->eFormat;
				break;
			}
		}
	}
}

static void restore_select_formats(void)
{
	while (select_count > 0) {
		select_count--;
		selects[select_count].inst->sDest.eFormat = selects[select_count].fmt;
	}
}

static int is_block_open(UF_OPCODE op)
{
	switch (op) {
	case UFOP_IF: case UFOP_IFC: case UFOP_IFP: case UFOP_IFNZBIT: case UFOP_ELSE:
	case UFOP_LOOP: case UFOP_REP: case UFOP_GLSLLOOP: case UFOP_GLSLSTATICLOOP:
	case UFOP_LABEL: case UFOP_SWITCH:
		return 1;
	default:
		return 0;
	}
}

static int is_block_close(UF_OPCODE op)
{
	switch (op) {
	case UFOP_ENDIF: case UFOP_ELSE: case UFOP_ENDLOOP: case UFOP_ENDREP:
	case UFOP_GLSLENDLOOP: case UFOP_RET: case UFOP_ENDSWITCH:
		return 1;
	default:
		return 0;
	}
}

/*
 * For x % y on 8 or 16-bit integers the front end extends both operands to
 * U32 or I32, divides with a DIV2 in that format, then reads the result in
 * the narrow one: a register with two formats again.  The remainder is
 * smaller in magnitude than the divisor, so it reads the same in the wide
 * format; such a read gets that format back.  Only a DIV2 earlier in the
 * same block, both of whose operands are MOVs there from the format read,
 * counts.
 * restore_div2_reads undoes it.
 */
#define MAX_DIV2_READS	256

static UF_REGISTER *div2_reads[MAX_DIV2_READS];
static UF_REGFORMAT div2_read_fmts[MAX_DIV2_READS];
static int div2_read_count;

static UF_REGFORMAT wide_fmt(UF_REGFORMAT fmt)
{
	return fmt == UF_REGFORMAT_I16 || fmt == UF_REGFORMAT_I8_UN ? UF_REGFORMAT_I32 : UF_REGFORMAT_U32;
}

/* the format a temporary is an extending MOV from, or INVALID */
static UF_REGFORMAT extended_from(PUNIFLEX_INST const *defs, IMG_UINT32 count, const UF_REGISTER *r)
{
	PUNIFLEX_INST def;

	if (r->eType != UFREG_TYPE_TEMP || r->eRelativeIndex != UFREG_RELATIVEINDEX_NONE ||
	    r->uNum >= count || r->byMod != 0)
		return UF_REGFORMAT_INVALID;
	def = defs[r->uNum];
	if (!def || def->eOpCode != UFOP_MOV || def->uPredicate != UF_PRED_NONE ||
	    def->sDest.byMod != 0 || def->asSrc[0].byMod != 0 ||
	    !is_narrow_int_fmt(def->asSrc[0].eFormat) || def->sDest.eFormat != wide_fmt(def->asSrc[0].eFormat))
		return UF_REGFORMAT_INVALID;
	return def->asSrc[0].eFormat;
}

static void widen_div2_reads(PUNIFLEX_INST prog)
{
	PUNIFLEX_INST inst;
	/* per temporary: its definition in this block, and for a DIV2 from
	 * operands zero-extended from one format, that format */
	PUNIFLEX_INST *defs;
	UF_REGFORMAT *narrow;
	IMG_UINT32 count = 0, i, k;

	div2_read_count = 0;
	for (inst = prog; inst; inst = inst->psILink) {
		if (inst->sDest.eType == UFREG_TYPE_TEMP && inst->sDest.uNum + dest_regs(inst) > count)
			count = inst->sDest.uNum + dest_regs(inst);
		if (inst->sDest2.eType == UFREG_TYPE_TEMP && inst->sDest2.uNum >= count)
			count = inst->sDest2.uNum + 1;
	}
	defs = calloc(count + 1, sizeof(*defs));
	narrow = malloc((count + 1) * sizeof(*narrow));
	if (!defs || !narrow) {
		free(defs);
		free(narrow);
		return;
	}
	for (i = 0; i <= count; i++)
		narrow[i] = UF_REGFORMAT_INVALID;
	for (inst = prog; inst && div2_read_count < MAX_DIV2_READS; inst = inst->psILink) {
		PCINPUT_INST_DESC desc = PVRUniFlexGetInputInstDesc(inst->eOpCode);
		UF_REGFORMAT fmt = UF_REGFORMAT_INVALID;

		if (!desc || is_block_open(inst->eOpCode) || is_block_close(inst->eOpCode) ||
		    inst->eOpCode == UFOP_BLOCK || inst->eOpCode == UFOP_JUMP) {
			memset(defs, 0, (count + 1) * sizeof(*defs));
			continue;
		}
		for (i = 0; i < desc->uNumSrcArgs && i < UNIFLEX_MAXIMUM_SOURCE_ARGUMENT_COUNT &&
			    div2_read_count < MAX_DIV2_READS; i++) {
			UF_REGISTER *r = &inst->asSrc[i];

			if (r->eType != UFREG_TYPE_TEMP || r->eRelativeIndex != UFREG_RELATIVEINDEX_NONE ||
			    r->uNum >= count || !defs[r->uNum] || defs[r->uNum]->eOpCode != UFOP_DIV2 ||
			    !is_narrow_int_fmt(r->eFormat) || narrow[r->uNum] != r->eFormat)
				continue;
			div2_reads[div2_read_count] = r;
			div2_read_fmts[div2_read_count++] = r->eFormat;
			r->eFormat = wide_fmt(r->eFormat);
		}
		if (inst->eOpCode == UFOP_DIV2 && inst->uPredicate == UF_PRED_NONE &&
		    extended_from(defs, count, &inst->asSrc[0]) == extended_from(defs, count, &inst->asSrc[1])) {
			fmt = extended_from(defs, count, &inst->asSrc[0]);
			if (inst->sDest.eFormat != wide_fmt(fmt) || inst->sDest2.eFormat != wide_fmt(fmt))
				fmt = UF_REGFORMAT_INVALID;
		}
		if (desc->uNumDests >= 1 && inst->sDest.eType == UFREG_TYPE_TEMP)
			for (k = 0; k < dest_regs(inst); k++) {
				defs[inst->sDest.uNum + k] = inst;
				narrow[inst->sDest.uNum + k] = fmt;
			}
		if (desc->uNumDests >= 2 && inst->sDest2.eType == UFREG_TYPE_TEMP) {
			defs[inst->sDest2.uNum] = inst;
			narrow[inst->sDest2.uNum] = fmt;
		}
	}
	free(defs);
	free(narrow);
}

static void restore_div2_reads(void)
{
	while (div2_read_count > 0) {
		div2_read_count--;
		div2_reads[div2_read_count]->eFormat = div2_read_fmts[div2_read_count];
	}
}

/*
 * A float SETB subtracts its sources and tests the difference against zero,
 * so it gets equal infinities wrong (their difference is NaN): only < is
 * right for every pair, false with a NaN.  And the front end lowers each
 * unordered comparison as the ordered one, so none is true for a NaN:
 * if (a >= b) (lowered as !(a < b), LLVM's ult, a SETBLT) is taken for one.
 * tools/oclc/patch_frontend.py has the front end emit the opcodes below for
 * the unordered comparisons (and ordered !=) instead, on the operands it
 * would give the ordered SETB.  Those and float SETBGE, SETBEQ and SETBNE
 * become, from l = x < y, g = y < x and o = |x| >= 0 & |y| >= 0 (ordered:
 * |x| - 0 is NaN only for a NaN):
 *	x >= y		!l & o		ult(x, y)	l | !o
 *	x == y		!(l | g) & o	uge(x, y)	!l
 *	x != y		l | g | !o	ueq(x, y)	!(l | g)
 *					one(x, y)	l | g
 * Except: IMG's == (a library function) tests x * x for a NaN with
 * uge(x * x, 0), which the ordered lowering made false for one; that stays
 * ordered.  restore_compares undoes it all.
 */
#define UFOP_SHIM_SETBULT	((UF_OPCODE)0xF0)
#define UFOP_SHIM_SETBUGE	((UF_OPCODE)0xF1)
#define UFOP_SHIM_SETBUEQ	((UF_OPCODE)0xF2)
#define UFOP_SHIM_SETBONE	((UF_OPCODE)0xF3)
#define MAX_COMPARES		1024

static struct compare_save {
	PUNIFLEX_INST inst, last;
	UNIFLEX_INST orig;
} compares[MAX_COMPARES];
static int compare_count;

static const char *shim_opcode_name(UF_OPCODE op)
{
	switch ((int)op) {
	case UFOP_SHIM_SETBULT: return "SETBULT";
	case UFOP_SHIM_SETBUGE: return "SETBUGE";
	case UFOP_SHIM_SETBUEQ: return "SETBUEQ";
	case UFOP_SHIM_SETBONE: return "SETBONE";
	default: return NULL;
	}
}

/* USC compares as its destination's format: an integer one compares bits */
static int is_f32_setb(const UNIFLEX_INST *inst)
{
	return (inst->eOpCode == UFOP_SETBGE || inst->eOpCode == UFOP_SETBEQ ||
		inst->eOpCode == UFOP_SETBNE) && inst->sDest.eFormat == UF_REGFORMAT_F32 &&
	       inst->asSrc[0].eFormat == UF_REGFORMAT_F32 && inst->asSrc[1].eFormat == UF_REGFORMAT_F32;
}

/* whether every channel r reads is a static 0 */
static int is_zero(const UF_REGISTER *r, PUNIFLEX_CONSTDEF consts)
{
	IMG_UINT32 k;

	if (r->byMod)
		return 0;
	if (r->eType == UFREG_TYPE_IMMEDIATE)
		return r->uNum == 0;
	if (r->eType != UFREG_TYPE_CONST || r->eRelativeIndex != UFREG_RELATIVEINDEX_NONE || !consts)
		return 0;
	for (k = 0; k < 4; k++) {
		IMG_UINT32 sw = (r->u.uSwiz >> (3 * k)) & 7, i = r->uNum * 4 + sw;

		if (sw == UFREG_SWIZ_0)
			continue;
		if (sw > UFREG_SWIZ_A || i >= consts->uCount ||
		    !(consts->puConstStaticFlags[i / 32] & (1u << (i % 32))) ||
		    ((const IMG_UINT32 *)consts->pfConst)[i] != 0)
			return 0;
	}
	return 1;
}

static int same_reg(const UF_REGISTER *a, const UF_REGISTER *b)
{
	return a->eType == b->eType && a->uNum == b->uNum && a->u.uSwiz == b->u.uSwiz &&
	       a->byMod == b->byMod && a->eRelativeIndex == UFREG_RELATIVEINDEX_NONE &&
	       b->eRelativeIndex == UFREG_RELATIVEINDEX_NONE;
}

/* the comparison's sequence: the first instruction is the comparison itself */
struct compare_seq {
	PUNIFLEX_INST inst, last;
	UF_REGISTER like;	/* its destination, for the temporaries */
	IMG_UINT32 *next;	/* the next free temporary */
};

static UF_REGISTER seq_emit(struct compare_seq *q, UF_OPCODE op, const UF_REGISTER *a,
			    const UF_REGISTER *b)
{
	UF_REGISTER t = q->like, r;

	t.uNum = (*q->next)++;
	t.eType = UFREG_TYPE_TEMP;
	t.byMod = 0;
	t.eRelativeIndex = UFREG_RELATIVEINDEX_NONE;
	t.uArrayTag = 0;
	if (!q->last) {
		q->inst->eOpCode = op;
		q->inst->sDest = t;
		q->inst->asSrc[0] = *a;
		if (b)
			q->inst->asSrc[1] = *b;
		q->inst->uPredicate = UF_PRED_NONE;
		q->last = q->inst;
	} else {
		PUNIFLEX_INST inst = calloc(1, sizeof(*inst));

		if (!inst)
			abort();
		inst->eOpCode = op;
		inst->sDest = t;
		inst->asSrc[0] = *a;
		if (b)
			inst->asSrc[1] = *b;
		inst->uPredicate = UF_PRED_NONE;
		inst->psILink = q->last->psILink;
		q->last->psILink = inst;
		q->last = inst;
	}
	r = t;
	r.u.uSwiz = UFREG_SWIZ_NONE;
	return r;
}

/* |r| >= 0: whether r isn't a NaN */
static UF_REGISTER seq_ordered(struct compare_seq *q, const UF_REGISTER *r)
{
	UF_REGISTER a = *r, zero;

	a.byMod = (a.byMod & ~UFREG_SOURCE_NEGATE) | UFREG_SOURCE_ABS;
	set_reg(&zero, UFREG_TYPE_IMMEDIATE, 0, UF_REGFORMAT_F32);
	return seq_emit(q, UFOP_SETBGE, &a, &zero);
}

static void expand_compares(PUNIFLEX_INST prog, PUNIFLEX_CONSTDEF consts)
{
	PUNIFLEX_INST inst, *defs;
	IMG_UINT32 next = 0, count, i;

	compare_count = 0;
	for (inst = prog; inst; inst = inst->psILink) {
		if (inst->sDest.eType == UFREG_TYPE_TEMP && inst->sDest.uNum + dest_regs(inst) > next)
			next = inst->sDest.uNum + dest_regs(inst);
		if (inst->sDest2.eType == UFREG_TYPE_TEMP && inst->sDest2.uNum >= next)
			next = inst->sDest2.uNum + 1;
		for (i = 0; i < UNIFLEX_MAXIMUM_SOURCE_ARGUMENT_COUNT; i++)
			if (inst->asSrc[i].eType == UFREG_TYPE_TEMP && inst->asSrc[i].uNum >= next)
				next = inst->asSrc[i].uNum + 1;
	}
	/* each temporary's definition earlier in the block */
	count = next;
	defs = calloc(count + 1, sizeof(*defs));
	if (!defs)
		abort();
	for (inst = prog; inst; inst = inst->psILink) {
		struct compare_seq q;
		struct compare_save *s;
		UF_REGISTER x, y, l, g, o, e, d;
		UF_OPCODE op = inst->eOpCode;

		if (is_block_open(op) || is_block_close(op) || op == UFOP_BLOCK || op == UFOP_JUMP ||
		    op == UFOP_CALL) {
			memset(defs, 0, (count + 1) * sizeof(*defs));
			continue;
		}
		if (!shim_opcode_name(op) && !is_f32_setb(inst)) {
			if (inst->sDest.eType == UFREG_TYPE_TEMP && inst->sDest.uNum < count)
				defs[inst->sDest.uNum] = inst;
			continue;
		}
		x = inst->asSrc[0];
		y = inst->asSrc[1];
		/* IMG's NaN test, uge(x * x, 0): ordered */
		if (op == UFOP_SHIM_SETBUGE && x.eType == UFREG_TYPE_TEMP && x.uNum < count &&
		    x.eRelativeIndex == UFREG_RELATIVEINDEX_NONE && !x.byMod && defs[x.uNum] &&
		    defs[x.uNum]->eOpCode == UFOP_MUL && same_reg(&defs[x.uNum]->asSrc[0], &defs[x.uNum]->asSrc[1]) &&
		    is_zero(&y, consts))
			op = UFOP_SETBGE;
		if (compare_count == MAX_COMPARES) {
			fprintf(stderr, "usc_shim: too many float comparisons\n");
			abort();
		}
		s = &compares[compare_count++];
		s->inst = inst;
		s->orig = *inst;
		q.inst = inst;
		q.last = NULL;
		q.like = inst->sDest;
		q.next = &next;
		l = seq_emit(&q, UFOP_SETBLT, &x, &y);
		if (op == UFOP_SETBEQ || op == UFOP_SETBNE || op == UFOP_SHIM_SETBUEQ || op == UFOP_SHIM_SETBONE) {
			g = seq_emit(&q, UFOP_SETBLT, &y, &x);
			e = seq_emit(&q, UFOP_OR, &l, &g);
		}
		if (op == UFOP_SETBGE || op == UFOP_SETBEQ || op == UFOP_SETBNE || op == UFOP_SHIM_SETBULT) {
			UF_REGISTER ox = seq_ordered(&q, &x), oy = seq_ordered(&q, &y);

			o = seq_emit(&q, UFOP_AND, &ox, &oy);
		}
		switch ((int)op) {
		case UFOP_SETBGE:
			d = seq_emit(&q, UFOP_NOT, &l, NULL);
			seq_emit(&q, UFOP_AND, &d, &o);
			break;
		case UFOP_SETBEQ:
			d = seq_emit(&q, UFOP_NOT, &e, NULL);
			seq_emit(&q, UFOP_AND, &d, &o);
			break;
		case UFOP_SETBNE:
			d = seq_emit(&q, UFOP_NOT, &o, NULL);
			seq_emit(&q, UFOP_OR, &e, &d);
			break;
		case UFOP_SHIM_SETBULT:
			d = seq_emit(&q, UFOP_NOT, &o, NULL);
			seq_emit(&q, UFOP_OR, &l, &d);
			break;
		case UFOP_SHIM_SETBUGE:
			seq_emit(&q, UFOP_NOT, &l, NULL);
			break;
		case UFOP_SHIM_SETBUEQ:
			seq_emit(&q, UFOP_NOT, &e, NULL);
			break;
		}
		/* the last one writes the destination as the comparison did
		   (one(x, y) is l | g, already last) */
		q.last->sDest = s->orig.sDest;
		q.last->uPredicate = s->orig.uPredicate;
		s->last = q.last;
		if (s->orig.sDest.eType == UFREG_TYPE_TEMP && s->orig.sDest.uNum < count)
			defs[s->orig.sDest.uNum] = q.last;
		inst = q.last;
	}
	free(defs);
}

static void restore_compares(void)
{
	while (compare_count > 0) {
		struct compare_save *s = &compares[--compare_count];
		PUNIFLEX_INST inst, next;

		for (inst = s->inst->psILink; inst; inst = next) {
			next = inst == s->last ? NULL : inst->psILink;
			free(inst);
		}
		*s->inst = s->orig;
	}
}

/* Take the instructions first..last back out of the program and free them;
 * returns the program. */
static PUNIFLEX_INST unlink_added(PUNIFLEX_INST prog, PUNIFLEX_INST first, PUNIFLEX_INST last)
{
	PUNIFLEX_INST inst, next;

	if (prog == first) {
		prog = last->psILink;
	} else {
		for (inst = prog; inst && inst->psILink != first; inst = inst->psILink)
			;
		if (inst)
			inst->psILink = last->psILink;
	}
	for (inst = first; inst; inst = next) {
		next = inst == last ? NULL : inst->psILink;
		free(inst);
	}
	return prog;
}

static IMG_PVOID IMG_CALLCONV usp_alloc(IMG_UINT32 size)
{
	return malloc(size);
}

static IMG_VOID IMG_CALLCONV usp_free(IMG_PVOID p)
{
	free(p);
}

static IMG_VOID IMG_CALLCONV usp_print(const IMG_CHAR *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	fprintf(stderr, "usp: ");
	vfprintf(stderr, fmt, ap);
	fprintf(stderr, "\n");
	va_end(ap);
}

static void put32(FILE *f, IMG_UINT32 v)
{
	fwrite(&v, 4, 1, f);
}

static void put_const_loads(FILE *f, const USP_HW_CONST_LOAD *l, IMG_UINT32 count)
{
	IMG_UINT32 i;

	for (i = 0; i < count; i++) {
		put32(f, l[i].uSrcIdx);
		put32(f, l[i].uSrcShift);
		put32(f, l[i].uDestIdx);
		put32(f, l[i].uDestShift);
		put32(f, l[i].eFormat);
	}
}

static void put_static_consts(FILE *f, PUNIFLEX_CONSTDEF consts)
{
	IMG_UINT32 i, count = 0;

	for (i = 0; consts && i < consts->uCount; i++)
		if (consts->puConstStaticFlags[i / 32] & (1u << (i % 32)))
			count++;
	put32(f, count);
	for (i = 0; consts && i < consts->uCount; i++)
		if (consts->puConstStaticFlags[i / 32] & (1u << (i % 32))) {
			put32(f, i);
			put32(f, ((const IMG_UINT32 *)consts->pfConst)[i]);
		}
}

/*
 * prefix.N.hw: "UHW2", then the USP_HW_SHADER fields below as 32-bit words,
 * the instructions (two words each), the register constant loads and the
 * memory constant loads (5 words each: src, src shift, dest, dest shift, fmt),
 * then the front end's static constants: a count and (channel, value) pairs.
 */
static int dump_hw(const char *prefix, int n, PUSC_USP_PC_SHADER pc, PUNIFLEX_CONSTDEF consts)
{
	IMG_PVOID ctx, shader;
	PUSP_HW_SHADER hw;
	char path[512];
	FILE *f;
	int i;

	ctx = PVRUniPatchCreateContext(usp_alloc, usp_free, usp_print);
	if (!ctx)
		return -1;
	/*
	 * An empty instruction at the program's start, for the phase instruction
	 * the hardware needs there (SGX543 has unlimited phases; the GLES driver
	 * does the same).
	 */
	PVRUniPatchSetPreambleInstCount(ctx, 1);
	shader = PVRUniPatchCreateShader(ctx, (PUSP_PC_SHADER)pc);
	hw = shader ? PVRUniPatchFinaliseShader(ctx, shader) : NULL;
	if (!hw) {
		fprintf(stderr, "usc_shim: USP could not finalise program %d\n", n);
		if (shader)
			PVRUniPatchDestroyShader(ctx, shader);
		PVRUniPatchDestroyContext(ctx);
		return -1;
	}
	snprintf(path, sizeof(path), "%s.%d.hw", prefix, n);
	f = fopen(path, "wb");
	if (f) {
		put32(f, 0x32574855);
		put32(f, hw->uFlags);
		put32(f, hw->uInstCount);
		put32(f, hw->uProgStartInstIdx);
		put32(f, hw->uSAUpdateInstCount);
		put32(f, hw->uPARegCount);
		put32(f, hw->uReservedPARegCount);
		put32(f, hw->uTempRegCount);
		put32(f, hw->uSARegCount);
		put32(f, hw->sSARegUse.uFlags);
		put32(f, hw->sSARegUse.uMemConstBaseAddrReg);
		put32(f, hw->sSARegUse.uMemConst2BaseAddrReg);
		put32(f, hw->sSARegUse.uScratchMemBaseAddrReg);
		put32(f, hw->sSARegUse.uRegConstsFirstReg);
		put32(f, hw->sSARegUse.uRegConstsRegCount);
		put32(f, hw->sSARegUse.uIndexedTempBaseAddrReg);
		put32(f, hw->uMemConstCount);
		put32(f, hw->uRegConstCount);
		put32(f, hw->uRegTexStateCount + hw->uMemTexStateCount + hw->uPSInputCount);
		put32(f, hw->uIndexedTempTotalSize);
		put32(f, hw->uScratchAreaSize);
		put32(f, (IMG_UINT32)hw->iSAAddressAdjust);
		put32(f, hw->uVSInputPARegCount);
		for (i = 0; i < EURASIA_USE_PRIMATTR_BANK_SIZE / 32; i++)
			put32(f, hw->auVSInputsUsed[i]);
		put32(f, hw->uShaderOutputsCount);
		fwrite(hw->puInsts, 8, hw->uInstCount, f);
		put_const_loads(f, hw->psRegConstLoads, hw->uRegConstCount);
		put_const_loads(f, hw->psMemConstLoads, hw->uMemConstCount);
		put_static_consts(f, consts);
		fclose(f);
	}
	PVRUniPatchDestroyHWShader(ctx, hw);
	PVRUniPatchDestroyShader(ctx, shader);
	PVRUniPatchDestroyContext(ctx);
	return 0;
}

static void dump_output(const char *prefix, int n, PUSC_USP_PC_SHADER pc)
{
	const IMG_UINT32 *hdr = (const IMG_UINT32 *)pc;
	char path[512];
	FILE *f;

	snprintf(path, sizeof(path), "%s.%d.usp", prefix, n);
	f = fopen(path, "wb");
	if (!f)
		return;
	fwrite(pc, 1, 12 + hdr[2], f);
	fclose(f);
}

static IMG_UINT32 params_unknown_flags(IMG_UINT32 flags)
{
	/* This USC assigns meanings to bits 0-29 of uFlags. */
	return flags & 0xC0000000U;
}

static void dump_program(const char *prefix, int n, IMG_UINT32 flags, PUNIFLEX_INST insts,
			 PUNIFLEX_CONSTDEF consts, const void *params, size_t params_size)
{
	char path[512];
	char line[2048];
	FILE *f;
	PUNIFLEX_INST inst;
	IMG_UINT32 i;
	IMG_INT32 indent = 0;
	int raw = getenv("OCLC_RAW") != NULL;

	snprintf(path, sizeof(path), "%s.%d.params", prefix, n);
	f = fopen(path, "wb");
	if (f) {
		fwrite(params, 1, params_size, f);
		fclose(f);
	}

	snprintf(path, sizeof(path), "%s.%d.uf", prefix, n);
	f = fopen(path, "w");
	if (!f)
		return;
	fprintf(f, "; compile flags 0x%08X\n", flags);
	if (consts) {
		fprintf(f, "; %u constant channels\n", consts->uCount);
		for (i = 0; i < consts->uCount; i++)
			if (consts->puConstStaticFlags[i / 32] & (1u << (i % 32)))
				fprintf(f, ";   c%u.%u = 0x%08X\n", i / 4, i % 4,
					((const IMG_UINT32 *)consts->pfConst)[i]);
	}
	for (inst = insts; inst; inst = inst->psILink) {
		IMG_UINT32 texmask = 0;

		if (is_block_close(inst->eOpCode) && indent > 0)
			indent--;
		line[0] = 0;
		if (shim_opcode_name(inst->eOpCode)) {
			/* USC can't decode the front end's (patched) unordered comparisons */
			UNIFLEX_INST copy = *inst;

			copy.eOpCode = UFOP_SETBLT;
			PVRUniFlexDecodeInputInst(&copy, indent, line, &texmask);
			fprintf(f, "%s\t; %s\n", line, shim_opcode_name(inst->eOpCode));
		} else {
			PVRUniFlexDecodeInputInst(inst, indent, line, &texmask);
			fprintf(f, "%s\n", line);
		}
		if (raw) {
			int a;

			for (a = -1; a < UNIFLEX_MAXIMUM_SOURCE_ARGUMENT_COUNT; a++) {
				const UF_REGISTER *r = a < 0 ? &inst->sDest : &inst->asSrc[a];

				if (r->eType == 0 && r->uNum == 0 && r->eRelativeIndex == 0 && !r->uArrayTag)
					continue;
				fprintf(f, ";\t\t%s%d: num=%u type=%d fmt=%d swiz/mask=%04X mod=%02X rel=%d/%d/%u.%u tag=%u stride=%u\n",
					a < 0 ? "dst" : "src", a < 0 ? 0 : a, r->uNum, r->eType, r->eFormat,
					a < 0 ? r->u.byMask : r->u.uSwiz, r->byMod, r->eRelativeIndex, r->eRelativeType,
					r->uRelativeNum, r->byRelativeChan, r->uArrayTag, r->u1.uRelativeStrideInBytes);
			}
		}
		if (is_block_open(inst->eOpCode))
			indent++;
	}
	fclose(f);
}

IMG_PVOID IMG_CALLCONV PVRUniFlexCreateContext(USC_ALLOCFN pfnAlloc, USC_FREEFN pfnFree, USC_PRINTFN pfnPrint,
					       USC_PDUMPFN pfnPDump, IMG_PVOID pvPDumpFnDrvParam,
					       USC_STARTFN pfnStart, USC_FINISHFN pfnFinish,
					       IMG_PVOID pvMetricsFnDrvParam)
{
	if (use_img())
		return img.create(pfnAlloc, pfnFree, pfnPrint, pfnPDump, pvPDumpFnDrvParam, pfnStart, pfnFinish,
				  pvMetricsFnDrvParam);
	return psp2_PVRUniFlexCreateContext(pfnAlloc, pfnFree, pfnPrint, pfnPDump, pvPDumpFnDrvParam, pfnStart,
					    pfnFinish, pvMetricsFnDrvParam);
}

/* The programs compiled so far, so oclc can tell which kernel each one is. */
#define MAX_PROGRAMS	256
static const void *programs[MAX_PROGRAMS];

int usc_shim_program_index(const void *pc)
{
	int i;

	for (i = 0; i < MAX_PROGRAMS; i++)
		if (pc && programs[i] == pc)
			return i;
	return -1;
}

IMG_UINT32 IMG_CALLCONV PVRUniFlexCompileToUspBin(IMG_PVOID pvContext, IMG_UINT32 *pui32Flags, PUNIFLEX_INST psSWProc,
						  PUNIFLEX_CONSTDEF psConstants,
						  PUNIFLEX_PROGRAM_PARAMETERS psProgramParameters,
						  PUSC_USP_PC_SHADER *ppsPCShader)
{
	const char *prefix = getenv("OCLC_DUMP");
	static int dump_index;
	const char *target = getenv("OCLC_TARGET");
	UNIFLEX_PROGRAM_PARAMETERS params;
	PUNIFLEX_INST added[2] = { NULL, NULL }, zeros[2] = { NULL, NULL };
	int gxm = getenv("OCLC_GXM") != NULL;
	IMG_UINT32 err;
	int n = dump_index++;

	if (prefix)
		dump_program(prefix, n, *pui32Flags, psSWProc, psConstants, psProgramParameters, PARAMS_IMG_SIZE);

	expand_compares(psSWProc, psConstants);
	if (use_img()) {
		err = img.compile(pvContext, pui32Flags, psSWProc, psConstants, psProgramParameters, ppsPCShader);
	} else {
		if (params_unknown_flags(psProgramParameters->uFlags) || psProgramParameters->uFlags2)
			fprintf(stderr, "usc_shim: warning: program %d uses compile flags 0x%08X/0x%08X unknown to this USC\n",
				n, params_unknown_flags(psProgramParameters->uFlags), psProgramParameters->uFlags2);
		memset(&params, 0, sizeof(params));
		memcpy(&params, psProgramParameters, PARAMS_COMMON_SIZE);
		/* Constant-only calculations would otherwise go to a secondary
		 * (SA update) program, which kernel dispatch does not run. */
		if (!getenv("OCLC_KEEP_SA"))
			params.uFlags &= ~UF_EXTRACTCONSTANTCALCS;
		params.sTarget.eID = SGX_CORE_ID_543;
		params.sTarget.uiRev = VITA_SGX_REV;
		params.uMPCoreCount = VITA_SGX_CORES;
		if (target) {
			unsigned id, rev;

			if (sscanf(target, "%u:%u", &id, &rev) == 2) {
				params.sTarget.eID = id;
				params.sTarget.uiRev = rev;
				params.uMPCoreCount = 0;
			}
		}
		if (gxm) {
			psSWProc = gxm_retarget(psSWProc, psConstants, &params, added);
			if (!psSWProc) {
				fprintf(stderr, "usc_shim: program %d can't be retargeted for GXM\n", n);
				return UF_ERR_GENERIC;
			}
		}
		reinterpret_const_reads(psSWProc);
		order_phi_copies(psSWProc);
		narrow_select_dests(psSWProc);
		widen_div2_reads(psSWProc);
		psSWProc = zero_unwritten_temps(psSWProc, zeros);
		if (prefix && (gxm || zeros[0] || phi_save_count || const_read_count || select_count ||
			       div2_read_count || compare_count))
			dump_program(prefix, n + 1000, *pui32Flags, psSWProc, psConstants, &params, sizeof(params));
		err = psp2_PVRUniFlexCompileToUspBin(pvContext, pui32Flags, psSWProc, psConstants, &params,
						     ppsPCShader);
		/* unlink our instructions before the front end frees its own */
		if (zeros[0])
			psSWProc = unlink_added(psSWProc, zeros[0], zeros[1]);
		if (added[0])
			psSWProc = unlink_added(psSWProc, added[0], added[1]);
		restore_phi_copies();
		restore_const_reads();
		restore_select_formats();
		restore_div2_reads();
	}
	restore_compares();
	if (!err && n < MAX_PROGRAMS)
		programs[n] = *ppsPCShader;
	if (prefix && !err && *ppsPCShader) {
		dump_output(prefix, n, *ppsPCShader);
		if (gxm && dump_hw(prefix, n, *ppsPCShader, psConstants))
			err = UF_ERR_GENERIC;
	}
	if (getenv("OCLC_VERBOSE"))
		fprintf(stderr, "usc_shim: program %d type %d -> %u\n", n, psProgramParameters->eShaderType, err);
	return err;
}

IMG_VOID IMG_CALLCONV PVRUniFlexDestroyUspBin(IMG_PVOID pvContext, PUSC_USP_PC_SHADER psPCShader)
{
	if (use_img())
		img.destroy_bin(pvContext, psPCShader);
	else
		psp2_PVRUniFlexDestroyUspBin(pvContext, psPCShader);
}

IMG_VOID IMG_CALLCONV PVRUniFlexDestroyContext(IMG_PVOID pvContext)
{
	if (use_img())
		img.destroy(pvContext);
	else
		psp2_PVRUniFlexDestroyContext(pvContext);
}
