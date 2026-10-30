#!/usr/bin/env python3
"""Package a USP-finalised kernel (prefix.N.hw from usc_shim) as a GXM vertex
program.

The program (built with a one-instruction USP preamble, which becomes its
phase instruction) is drawn as indexed, instanced points: the PDS writes the vertex
index to pa0 and the instance index to pa1 (get_global_id(0/1) less the
per-draw base the kernel adds from c18).  All of the kernel's constants are
loaded into secondary attribute registers by one DMA of the default uniform
buffer, which the runtime fills from the register constant map written next
to the program.

    mkgxp.py [--index-pa N] [--instance-pa N] [--no-instanced] [--no-indexed]
             file.hw out.gxp out.map

out.map: "VKM1", sa_count, load_count, static_count, then load_count
(sa register, constant channel) pairs and static_count (channel, value)
pairs, all 32-bit.
"""
import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'oclc'))
import hwshader  # noqa: E402

GXP_MAGIC = 0x00505847

# SceGxmProgram flags
FLAG_PER_INSTANCE = 0x00000002	# a vertex program's task runs in per-instance mode
FLAG_UNK10000 = 0x00010000	# set on every vertex program the Sony compiler emits
FLAG_INSTANCED = 0x00020000
FLAG_INDEXED = 0x00040000

# USP_HWSHADER_FLAGS_*
PER_INSTANCE_MODE = 0x4		# dynamic branches or mutexes: instances must not run in lockstep
ABS_BRANCHES_USED = 0x800
SAPROG_LABEL_AT_END = 0x100

DEFAULT_UNIFORM_BUFFER = 14
DMA_MAX_WORDS = 256		# a PDS DMA's burst size field (libgxm: size - 1 in bits 0-7)

# The runtime's constant table (GXC_CONST_CHANNELS in runtime/gpu/gxc_kernel.h)
CONST_CHANNELS = 1024
BUFFER_FLAGS_DMA = 0x10000000	# the DMA block is rewritten when the buffer moves

INDEX_PA = 0
INSTANCE_PA = 1
POSITION_OUTPUTS = 0x04001000	# four output registers, position only

PHAS_END = (0x00000000, 0xFA440700)		# phas #0, #0, pixel, end, parallel
EMITVTX_END = (0xA0200000, 0xFB275000)	# emitvtx.end.freep #0, #0

CONST_F32 = 0
CONST_STATIC = 4


class PackError(Exception):
    pass


def check(hw):
    if hw['flags'] & ABS_BRANCHES_USED:
        raise PackError('absolute branches')
    if hw['sa_update_inst_count']:
        raise PackError('needs a secondary (SA update) program')
    if hw['memconst_count']:
        raise PackError('needs memory constants')
    if hw['texture_count']:
        raise PackError('samples textures')
    if hw['scratch_size']:
        raise PackError('needs scratch memory (%d bytes)' % hw['scratch_size'])
    if hw['indexed_temp_size']:
        raise PackError('needs indexed temporaries (%d)' % hw['indexed_temp_size'])
    if hw['sa_use_flags'] & ~hwshader.SA_REGCONSTS:
        raise PackError('secondary attribute use 0x%x' % hw['sa_use_flags'])
    start = hw['prog_start']
    if start >= hw['inst_count'] or tuple(hw['insts'][start]) != (0, 0):
        raise PackError('no empty preamble instruction at the program start (%d)' % start)
    for l in hw['reg_consts']:
        # a static load's source fields hold the value
        if l['fmt'] == CONST_STATIC:
            continue
        if l['fmt'] != CONST_F32 or l['src_shift'] or l['dest_shift']:
            raise PackError('register constant load %r' % l)
        if l['dest'] >= hw['sa_regs']:
            raise PackError('register constant load past the SA registers: %r' % l)
        if l['src'] >= CONST_CHANNELS:
            raise PackError('constant channel %d past the runtime\'s %d' % (l['src'], CONST_CHANNELS))


def build(hw, index_pa=INDEX_PA, instance_pa=INSTANCE_PA, indexed=True, instanced=True):
    check(hw)
    # The task starts at the program (after any subroutines it calls), on the
    # phase instruction that opens its one phase, in the USP's preamble slot.
    insts = list(hw['insts']) + [EMITVTX_END]
    insts[hw['prog_start']] = PHAS_END
    sa_regs = hw['sa_regs']
    pa_regs = max(hw['pa_regs'], instance_pa + 1, index_pa + 1)
    flags = FLAG_UNK10000 | (FLAG_INSTANCED if instanced else 0) | (FLAG_INDEXED if indexed else 0)
    if hw['flags'] & PER_INSTANCE_MODE:
        flags |= FLAG_PER_INSTANCE

    header_size = 0x98
    varyings_off = header_size
    # The PDS writes the vertex index to the register in the first u16 after
    # outputs2 and the instance index to the one in the second (tested on
    # hardware; tools/gxp/gxpdump.py names them the other way round).
    varyings = struct.pack('<IIIIIIIHH', 0, 0, 0, 0, POSITION_OUTPUTS, 0, 0,
                           index_pa, instance_pa)
    code_off = (varyings_off + len(varyings) + 7) & ~7
    code = b''.join(struct.pack('<II', lo, hi) for lo, hi in insts)
    entry_off = code_off + len(code)
    dma_off = entry_off + 4
    # The default uniform buffer into sa0.., in DMAs of at most DMA_MAX_WORDS
    dma = b''.join(struct.pack('<HHHH', DEFAULT_UNIFORM_BUFFER, w, w, min(DMA_MAX_WORDS, sa_regs - w))
                   for w in range(0, sa_regs, DMA_MAX_WORDS))
    dma_count = len(dma) // 8
    end = dma_off + len(dma)
    size = (end + 3) & ~3

    def rel(field, target):
        return target - field

    h = bytearray(header_size)
    struct.pack_into('<IBBHIII', h, 0x00, GXP_MAGIC, 1, 4, 0, size, 0, 0)
    struct.pack_into('<IIII', h, 0x14, flags,
                     BUFFER_FLAGS_DMA if sa_regs else 0, 0, 0)
    struct.pack_into('<Iii', h, 0x24, 0, rel(0x28, end), rel(0x2c, varyings_off))
    struct.pack_into('<HHHHHH', h, 0x30, pa_regs, sa_regs, hw['temp_regs'], 0, 0, 1)
    struct.pack_into('<Ii', h, 0x3c, len(insts), rel(0x40, code_off))
    struct.pack_into('<Iii', h, 0x44, 0, rel(0x48, code_off), rel(0x4c, entry_off))
    # scratch, thread, literal, data, texture buffers; default uniform buffer words
    struct.pack_into('<IIIIII', h, 0x50, 0, 0, 0, 0, 0, sa_regs)
    struct.pack_into('<iI', h, 0x68, rel(0x68, end), 0)
    struct.pack_into('<Ii', h, 0x70, 0, rel(0x74, end))
    struct.pack_into('<Ii', h, 0x78, 0, rel(0x7c, end))
    struct.pack_into('<Ii', h, 0x80, 0, rel(0x84, end))
    struct.pack_into('<Ii', h, 0x88, 0, rel(0x8c, end))
    struct.pack_into('<Ii', h, 0x90, dma_count, rel(0x94, dma_off))

    out = bytearray(size)
    out[0:header_size] = h
    out[varyings_off:varyings_off + len(varyings)] = varyings
    out[code_off:code_off + len(code)] = code
    struct.pack_into('<i', out, entry_off, hw['prog_start'])
    out[dma_off:dma_off + len(dma)] = dma
    # a GUID for the patcher's cache: anything that tells programs apart
    guid = 0
    for i in range(0, len(out), 4):
        guid = (guid * 31 + struct.unpack_from('<I', out, i)[0]) & 0xffffffff
    struct.pack_into('<II', out, 0x0c, guid, guid)
    return bytes(out)


def const_map(hw):
    loads = [(l['dest'], l['src']) for l in hw['reg_consts'] if l['fmt'] == CONST_F32]
    statics = dict(hw['static_consts'])
    for l in hw['reg_consts']:
        if l['fmt'] == CONST_STATIC:
            # the USP folded a literal straight into the load
            loads.append((l['dest'], 0x80000000 | len(statics)))
            statics[0x80000000 | len(statics)] = l['src']
    body = b''.join(struct.pack('<II', d, s) for d, s in loads)
    body += b''.join(struct.pack('<II', c, v) for c, v in sorted(statics.items()))
    return struct.pack('<4sIII', b'VKM1', hw['sa_regs'], len(loads), len(statics)) + body


def main():
    ap = argparse.ArgumentParser(usage=__doc__)
    ap.add_argument('--index-pa', type=int, default=INDEX_PA)
    ap.add_argument('--instance-pa', type=int, default=INSTANCE_PA)
    ap.add_argument('--no-instanced', action='store_true')
    ap.add_argument('--no-indexed', action='store_true')
    ap.add_argument('hw')
    ap.add_argument('gxp')
    ap.add_argument('map')
    a = ap.parse_args()
    hw = hwshader.parse(open(a.hw, 'rb').read())
    try:
        gxp = build(hw, a.index_pa, a.instance_pa, not a.no_indexed, not a.no_instanced)
    except PackError as e:
        sys.exit('mkgxp: %s: %s' % (a.hw, e))
    open(a.gxp, 'wb').write(gxp)
    open(a.map, 'wb').write(const_map(hw))


if __name__ == '__main__':
    main()
