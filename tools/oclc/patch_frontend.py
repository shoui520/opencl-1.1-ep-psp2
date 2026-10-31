#!/usr/bin/env python3
"""Patch IMG's OpenCL front end (liboclcompiler.so, DDK 1.12.3670983) so its
unordered float comparisons can be told from the ordered ones.

    patch_frontend.py IN OUT

The function at 0x8a38ec turns an LLVM comparison into a UniFlex SETB.  It
switches on the predicate, and each unordered float predicate shares the
case of the ordered one: ult(a, b) (what if (a >= b) branches on) becomes
SETBLT a, b, false for a NaN, and one (ordered !=) becomes SETBNE, true for
a NaN.  The six cases are copied into a code cave with the opcode replaced
by one usc_shim.c expands (UFOP_SHIM_SETBULT and so on), and their entries
in the switch's jump table point at the copies.  The cave is 0x899e44,
clang's #define _WIN32 for Windows targets: nothing refers to it.
"""
import hashlib
import struct
import sys

SHA256 = 'e668dc4de37b70fb3c4a19ba98a15a6a52b7720b4f6a63d914fae63295434e5a'

JUMP_TABLE = 0x8a3914           # the predicate - 1th entry is a B to its case
TO_LABEL_UNSWAPPED = 0x8a3adc   # operands in order
TO_LABEL_SWAPPED = 0x8a39c8     # operands swapped; r8 = 0
CAVE = 0x899e44
CAVE_SIZE = 284

SETBULT, SETBUGE, SETBUEQ, SETBONE = 0xF0, 0xF1, 0xF2, 0xF3
# LLVM's FCmpInst predicates
FCMP_ONE, FCMP_UEQ, FCMP_UGT, FCMP_UGE, FCMP_ULT, FCMP_ULE = 6, 9, 10, 11, 12, 13
R3, R7, R8, R11, SP = 3, 7, 8, 11, 13


def mov(rd, imm):
    assert 0 <= imm < 256
    return 0xE3A00000 | rd << 12 | imm


def str_r3_sp4():
    return 0xE58D3004


def b(at, target):
    off = (target - (at + 8)) >> 2
    assert -(1 << 23) <= off < (1 << 23)
    return 0xEA000000 | (off & 0xFFFFFF)


def cave_code():
    """the six cases: as the originals (r11 float, r8 equality, r7 signed,
    [sp + 4] the opcode), with the shim's opcodes"""
    code, entry, fix = [], {}, []

    def at():
        return CAVE + 4 * len(code)

    def label(name):
        entry[name] = at()

    # ueq and one: the cases of oeq/une, unswapped, r11 = 0, r8 = 1
    label(FCMP_UEQ); code.append(mov(R3, SETBUEQ)); fix.append((len(code), 'eq')); code.append(0)
    label(FCMP_ONE); code.append(mov(R3, SETBONE))
    label('eq')
    code += [mov(R11, 0), mov(R7, 0), str_r3_sp4(), mov(R8, 1)]
    code.append(b(at(), TO_LABEL_UNSWAPPED))
    # ugt and ule: ogt's and ole's, swapped, r11 = 1
    label(FCMP_UGT); code.append(mov(R3, SETBULT)); fix.append((len(code), 'sw')); code.append(0)
    label(FCMP_ULE); code.append(mov(R3, SETBUGE))
    label('sw')
    code += [mov(R11, 1), str_r3_sp4(), mov(R7, 0)]
    code.append(b(at(), TO_LABEL_SWAPPED))
    # ult and uge: olt's and oge's, unswapped, r11 = 1, r8 = 0
    label(FCMP_ULT); code.append(mov(R3, SETBULT)); fix.append((len(code), 'un')); code.append(0)
    label(FCMP_UGE); code.append(mov(R3, SETBUGE))
    label('un')
    code += [mov(R8, 0), mov(R7, 0), str_r3_sp4(), mov(R11, 1)]
    code.append(b(at(), TO_LABEL_UNSWAPPED))
    for i, name in fix:
        code[i] = b(CAVE + 4 * i, entry[name])
    assert 4 * len(code) <= CAVE_SIZE
    return code, entry


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    data = bytearray(open(sys.argv[1], 'rb').read())
    if hashlib.sha256(data).hexdigest() != SHA256:
        sys.exit('patch_frontend: %s is not liboclcompiler.so 1.12.3670983' % sys.argv[1])
    # the first LOAD segment maps the file at offset = address
    code, entry = cave_code()
    struct.pack_into('<%dI' % len(code), data, CAVE, *code)
    for pred in (FCMP_ONE, FCMP_UEQ, FCMP_UGT, FCMP_UGE, FCMP_ULT, FCMP_ULE):
        slot = JUMP_TABLE + 4 * (pred - 1)
        struct.pack_into('<I', data, slot, b(slot, entry[pred]))
    open(sys.argv[2], 'wb').write(data)


if __name__ == '__main__':
    main()
