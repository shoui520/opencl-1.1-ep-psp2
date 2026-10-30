#!/usr/bin/env python3
"""Read the USP-finalised hardware shaders usc_shim writes (prefix.N.hw).

    hwshader.py [--usedis PATH] file.hw
"""
import struct
import subprocess
import sys

FIELDS = ('flags', 'inst_count', 'prog_start', 'sa_update_inst_count', 'pa_regs',
          'reserved_pa_regs', 'temp_regs', 'sa_regs', 'sa_use_flags', 'memconst_base_reg',
          'memconst2_base_reg', 'scratch_base_reg', 'regconsts_first_reg', 'regconsts_count',
          'indexed_temp_base_reg', 'memconst_count', 'regconst_count', 'texture_count',
          'indexed_temp_size', 'scratch_size', 'sa_address_adjust', 'vs_input_pa_regs',
          'vs_inputs_used0', 'vs_inputs_used1', 'vs_inputs_used2', 'vs_inputs_used3',
          'output_count')

CONST_FMT = {0: 'f32', 1: 'f16', 2: 'c10', 3: 'u8', 4: 'static'}

# USP_HWSHADER_FLAGS_*
LABEL_AT_END = 0x2
ABS_BRANCHES_USED = 0x800
# USP_HW_SAREG_USE_FLAG_*
SA_MEMCONSTS = 0x1
SA_MEMCONSTS2 = 0x2
SA_SCRATCH = 0x4
SA_TEXSTATE = 0x8
SA_REGCONSTS = 0x10
SA_INDEXEDTEMPS = 0x20


def parse(data):
    if data[:4] != b'UHW2':
        raise ValueError('not a usc_shim hardware shader')
    pos = 4
    vals = struct.unpack_from('<%dI' % len(FIELDS), data, pos)
    pos += 4 * len(FIELDS)
    hw = dict(zip(FIELDS, vals))
    hw['sa_address_adjust'] = struct.unpack('<i', struct.pack('<I', hw['sa_address_adjust']))[0]
    hw['insts'] = [struct.unpack_from('<II', data, pos + 8 * i) for i in range(hw['inst_count'])]
    pos += 8 * hw['inst_count']

    def loads(count):
        nonlocal pos
        out = []
        for _ in range(count):
            src, src_shift, dest, dest_shift, fmt = struct.unpack_from('<5I', data, pos)
            out.append({'src': src, 'src_shift': src_shift, 'dest': dest,
                        'dest_shift': dest_shift, 'fmt': fmt})
            pos += 20
        return out

    hw['reg_consts'] = loads(hw['regconst_count'])
    hw['mem_consts'] = loads(hw['memconst_count'])
    count, = struct.unpack_from('<I', data, pos)
    hw['static_consts'] = dict(struct.unpack_from('<II', data, pos + 4 + 8 * i)
                               for i in range(count))
    return hw


def disassemble(usedis, insts, target='7:216'):
    text = ''.join('%08X %08X\n' % (hi, lo) for lo, hi in insts)
    return subprocess.run([usedis, '-t', target, '-'], input=text, capture_output=True,
                          text=True, check=True).stdout.splitlines()


def main():
    args = sys.argv[1:]
    usedis = None
    if args and args[0] == '--usedis':
        usedis, args = args[1], args[2:]
    hw = parse(open(args[0], 'rb').read())
    for k in FIELDS:
        print('  %-22s 0x%x' % (k, hw[k] & 0xffffffff))
    for name in ('reg_consts', 'mem_consts'):
        for l in hw[name]:
            if l['fmt'] == 4:
                print('%s: dest %d = static 0x%08x' % (name, l['dest'], l['src']))
            else:
                print('%s: dest %d <- c%d.%s %s shift %d/%d' % (
                    name, l['dest'], l['src'] // 4, 'xyzw'[l['src'] % 4],
                    CONST_FMT.get(l['fmt'], l['fmt']), l['src_shift'], l['dest_shift']))
    for ch, v in sorted(hw['static_consts'].items()):
        print('static c%d.%s = 0x%08x' % (ch // 4, 'xyzw'[ch % 4], v))
    lines = disassemble(usedis, hw['insts']) if usedis else \
        ['%08X %08X' % (hi, lo) for lo, hi in hw['insts']]
    for line in lines:
        print('   ' + line)


if __name__ == '__main__':
    main()
