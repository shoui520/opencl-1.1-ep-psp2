#!/usr/bin/env python3
"""Parse USP pre-compiled shader binaries ("PC shaders") as written by the USC.

Version 0x23 is PVR_PSP2's format (tools/intern/usc2/uspbin.c).  Version 0x25
is IMG DDK 1.12's; its differences were found by aligning the two encodings
of the same programs and are confined to the program descriptor.

    uspbin.py file.usp            dump the structure and every instruction
    uspbin.py --code file.usp     print only "hi lo" instruction words
"""
import struct
import sys

USP_PC_SHADER_ID = 0x55535020
BLOCK_NAMES = {1: 'PROGDESC', 2: 'HWCODE', 3: 'BRANCH', 4: 'LABEL', 5: 'SAMPLE',
               6: 'SAMPLEUNPACK', 7: 'TEXTURE_WRITE', 8: 'END'}

# Fixed-size part of the program descriptor, in write order.
PROGDESC_V23 = [
    ('I', 'uShaderType'), ('I', 'uFlags'), ('I', 'uUFFlags'), ('I', 'uHWFlags'),
    ('I', 'uTargetDev'), ('I', 'uTargetRev'), ('I', 'uBRNCount'), ('I', 'uSAUsageFlags'),
    ('H', 'uMemConstBaseAddrSAReg'), ('H', 'uMemConst2BaseAddrSAReg'),
    ('H', 'uScratchMemBaseAddrSAReg'), ('H', 'uTexStateFirstSAReg'),
    ('I', 'uTexStateMemOffset'),
    ('H', 'uRegConstsFirstSAReg'), ('H', 'uRegConstsMaxSARegCount'),
    ('H', 'uIndexedTempBaseAddrSAReg'), ('H', 'uExtraPARegs'), ('H', 'uPARegCount'),
    ('H', 'uTempRegCount'), ('H', 'uSecTempRegCount'),
    ('I', 'uIndexedTempTotalSize'),
    ('H', 'uPSInputCount'), ('H', 'uRegConstCount'), ('I', 'uMemConstCount'),
    ('H', 'uRegTexStateCount'),
    ('I', 'auNonAnisoTexStages'), ('I', 'auAnisoTexStages'),
    ('I', 'uScratchAreaSize'),
    ('h', 'iSAAddressAdjust'), ('H', 'uPSResultRegType'), ('H', 'uPSResultRegNum'),
    ('H', 'uPSResultRegFmt'), ('H', 'uPSResultRegCount'), ('H', 'uPSResultTempReg'),
    ('H', 'uPSResultPAReg'), ('H', 'uPSResultOutputReg'),
    ('H', 'uProgStartLabelID'), ('H', 'uPTPhase0EndLabelID'), ('H', 'uPTPhase1StartLabelID'),
    ('H', 'uPTSplitPhase1StartLabelID'), ('H', 'uSAUpdateInstCount'),
    ('H', 'uRegConstsSARegCount'), ('H', 'uVSInputPARegCount'), ('H', 'uShaderOutputsCount'),
]

# 1.12 additions (names are descriptive guesses; positions are what matter).
PROGDESC_V25 = list(PROGDESC_V23)
PROGDESC_V25.insert(PROGDESC_V25.index(('H', 'uProgStartLabelID')), ('H', 'v25_uLabelA'))
PROGDESC_V25.insert(PROGDESC_V25.index(('H', 'uProgStartLabelID')) + 1, ('H', 'v25_uLabelB'))
PROGDESC_V25.remove(('H', 'uSAUpdateInstCount'))
PROGDESC_V25 += [('H', 'uSAUpdateInstCount'), ('H', 'v25_unk1'), ('H', 'v25_unk2')]

USC_MAX_SHADER_OUTPUTS = 128


class Reader:
    def __init__(self, data, pos=0):
        self.data, self.pos = data, pos

    def take(self, fmt):
        v = struct.unpack_from('<' + fmt, self.data, self.pos)
        self.pos += struct.calcsize('<' + fmt)
        return v[0] if len(v) == 1 else v

    def inst(self):
        lo, hi = self.take('II')
        return hi, lo


def const_loads(r, count):
    loads = []
    for _ in range(count):
        fmt = r.take('H')
        if fmt == 5:  # USP_PC_CONST_FMT_STATIC
            val = r.take('I')
            dest, shift = r.take('HH')
            loads.append(('static', val, dest, shift))
        else:
            src, dest, shift = r.take('HHH')
            loads.append((fmt, src, dest, shift))
    return loads


def parse(data):
    r = Reader(data)
    ident, version, size = r.take('III')
    if ident != USP_PC_SHADER_ID:
        raise ValueError('not a USP PC shader')
    if len(data) != 12 + size:
        raise ValueError('size mismatch: header says %d, file has %d' % (12 + size, len(data)))
    layout = {0x23: PROGDESC_V23, 0x25: PROGDESC_V25}.get(version)
    if layout is None:
        raise ValueError('unsupported version 0x%x' % version)

    out = {'version': version, 'blocks': []}
    if r.take('I') != 1:
        raise ValueError('first block is not PROGDESC')
    pd = {}
    for fmt, name in layout:
        pd[name] = r.take(fmt)
    out['progdesc'] = pd
    out['brns'] = [r.take('I') for _ in range(pd['uBRNCount'])]
    out['psinputs'] = [r.take('HHHHHH') for _ in range(pd['uPSInputCount'])]
    out['memconsts'] = const_loads(r, pd['uMemConstCount'])
    out['regconsts'] = const_loads(r, pd['uRegConstCount'])
    out['sacode'] = [r.inst() for _ in range(pd['uSAUpdateInstCount'])]
    if pd['uShaderType'] == 1:
        out['vsinputs'] = [r.take('I') for _ in range((pd['uVSInputPARegCount'] + 31) // 32)]
    out['validoutputs'] = [r.take('I') for _ in range(USC_MAX_SHADER_OUTPUTS // 32)]

    while True:
        btype = r.take('I')
        name = BLOCK_NAMES.get(btype)
        if name is None:
            raise ValueError('unknown block type %d at 0x%x' % (btype, r.pos - 4))
        if name == 'END':
            break
        if name == 'HWCODE':
            count = r.take('H')
            moe = r.take('H' * 9)
            descs = [r.take('H') for _ in range(count)]
            insts = [r.inst() for _ in range(count)]
            out['blocks'].append(('HWCODE', {'moe': moe, 'descs': descs, 'insts': insts}))
        elif name == 'LABEL':
            out['blocks'].append(('LABEL', r.take('H')))
        elif name == 'BRANCH':
            inst = r.inst()
            out['blocks'].append(('BRANCH', {'inst': inst, 'dest': r.take('H')}))
        else:
            raise ValueError('block %s not handled (compute kernels should not use it)' % name)
    if r.pos != len(data):
        raise ValueError('END block at 0x%x but file is 0x%x bytes' % (r.pos, len(data)))
    return out


def code_words(p):
    """Every instruction in program order, as (kind, hi, lo)."""
    for hi, lo in p['sacode']:
        yield 'sa', hi, lo
    for kind, b in p['blocks']:
        if kind == 'HWCODE':
            for (hi, lo), desc in zip(b['insts'], b['descs']):
                yield 'code/%04x' % desc, hi, lo
        elif kind == 'BRANCH':
            hi, lo = b['inst']
            yield 'branch->L%d' % b['dest'], hi, lo
        elif kind == 'LABEL':
            yield 'label L%d' % b, None, None


def main():
    args = sys.argv[1:]
    code_only = args and args[0] == '--code'
    if code_only:
        args = args[1:]
    p = parse(open(args[0], 'rb').read())
    if code_only:
        for kind, hi, lo in code_words(p):
            if hi is None:
                print('#', kind)
            else:
                print('%08X %08X %s' % (hi, lo, kind))
        return
    print('version 0x%x' % p['version'])
    for k, v in p['progdesc'].items():
        print('  %-28s %s' % (k, v))
    for key in ('brns', 'psinputs', 'memconsts', 'regconsts', 'vsinputs', 'validoutputs'):
        if p.get(key):
            print('%s: %s' % (key, p[key]))
    for kind, hi, lo in code_words(p):
        print('  %-12s %s' % (kind, '' if hi is None else '%08X %08X' % (hi, lo)))


if __name__ == '__main__':
    main()
