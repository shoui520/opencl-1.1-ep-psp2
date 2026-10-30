#!/usr/bin/env python3
"""Dump a SceGxmProgram (GXP) binary.

Offsets in a GXP are relative to the field that holds them.

    gxpdump.py [--usedis PATH] file.gxp
"""
import struct
import subprocess
import sys

HEADER = [
    (0x00, 'I', 'magic'), (0x04, 'B', 'majorVersion'), (0x05, 'B', 'minorVersion'),
    (0x06, 'H', 'sdkVersion'), (0x08, 'I', 'size'), (0x0c, 'I', 'binaryGuid'),
    (0x10, 'I', 'sourceGuid'), (0x14, 'I', 'flags'), (0x18, 'I', 'bufferFlags'),
    (0x1c, 'I', 'texUnitFlags0'), (0x20, 'I', 'texUnitFlags1'),
    (0x24, 'I', 'parameterCount'), (0x28, 'i', 'parametersOffset'),
    (0x2c, 'i', 'varyingsOffset'), (0x30, 'H', 'primaryRegCount'),
    (0x32, 'H', 'secondaryRegCount'), (0x34, 'H', 'tempRegCount'), (0x36, 'H', 'reserved'),
    (0x38, 'H', 'tempRegCount2'), (0x3a, 'H', 'primaryPhaseCount'),
    (0x3c, 'I', 'primaryInstructionCount'), (0x40, 'i', 'primaryProgramOffset'),
    (0x44, 'I', 'secondaryInstructionCount'), (0x48, 'i', 'secondaryProgramOffset'),
    (0x4c, 'i', 'entryOffset'), (0x50, 'I', 'scratchBufferCount'),
    (0x54, 'I', 'threadBufferCount'), (0x58, 'I', 'literalBufferCount'),
    (0x5c, 'I', 'dataBufferCount'), (0x60, 'I', 'textureBufferCount'),
    (0x64, 'I', 'defaultUniformBufferCount'), (0x68, 'i', 'literalBufferOffset'),
    (0x6c, 'I', 'reserved2'), (0x70, 'I', 'literalCount'), (0x74, 'i', 'literalsOffset'),
    (0x78, 'I', 'bufferRefCount'), (0x7c, 'i', 'bufferRefsOffset'),
    (0x80, 'I', 'constantTextureCount'), (0x84, 'i', 'constantTexturesOffset'),
    (0x88, 'I', 'textureRefCount'), (0x8c, 'i', 'textureRefsOffset'),
    (0x90, 'I', 'secondaryPdsDataCount'), (0x94, 'i', 'secondaryPdsDataOffset'),
]
FIELD_POS = {name: off for off, _, name in HEADER}
CATEGORIES = {0: 'attribute', 1: 'uniform', 2: 'sampler', 3: 'aux surface', 4: 'uniform buffer'}


def rel(h, name):
    """Absolute position of the table a self-relative field points at."""
    return FIELD_POS[name] + h[name]


def cstr(data, pos):
    return data[pos:data.index(b'\0', pos)].decode('latin-1')


def parse(data):
    h = {name: struct.unpack_from('<' + fmt, data, off)[0] for off, fmt, name in HEADER}
    if h['magic'] != 0x00505847:
        raise ValueError('not a GXP')
    p = {'header': h}
    p['fragment'] = bool(h['flags'] & 1)

    params = []
    base = rel(h, 'parametersOffset')
    for i in range(h['parameterCount']):
        pos = base + i * 16
        name_off, info, array_size, res = struct.unpack_from('<iIII', data, pos)
        params.append({
            'name': cstr(data, pos + name_off), 'category': info & 0xf,
            'type': (info >> 4) & 0xf, 'components': (info >> 8) & 0xf,
            'container': (info >> 12) & 0xf, 'semantic': (info >> 16) & 0xff,
            'semanticIndex': (info >> 24) & 0xf, 'arraySize': array_size, 'resourceIndex': res,
            'info': info})
    p['parameters'] = params

    vpos = rel(h, 'varyingsOffset')
    if not p['fragment']:
        v = struct.unpack_from('<IIIIIIIHH', data, vpos)
        p['varyings'] = dict(zip(('inputMask0', 'inputMask1', 'regFormatMask0', 'regFormatMask1',
                                  'outputs', 'texCoordOutputs', 'outputs2', 'instanceData',
                                  'indexData'), v))
    else:
        out_reg, out_type, out_count, in_count, tex_count, in_off = \
            struct.unpack_from('<HBBHHi', data, vpos + 8)
        inputs = [struct.unpack_from('<IIII', data, vpos + 16 + in_off + 16 * i)
                  for i in range(in_count)]
        p['varyings'] = {'outputRegister': out_reg, 'outputRegisterType': out_type,
                         'outputComponentCount': out_count, 'inputCount': in_count,
                         'textureIndexCount': tex_count, 'inputs': inputs}

    def code(count_name, off_name):
        pos = rel(h, off_name)
        return [struct.unpack_from('<II', data, pos + 8 * i)[::-1] for i in range(h[count_name])]

    p['primary'] = code('primaryInstructionCount', 'primaryProgramOffset')
    p['secondary'] = code('secondaryInstructionCount', 'secondaryProgramOffset')
    p['entry'] = struct.unpack_from('<i', data, rel(h, 'entryOffset'))[0]

    def table(count_name, off_name, fmt):
        pos, size = rel(h, off_name), struct.calcsize('<' + fmt)
        return [struct.unpack_from('<' + fmt, data, pos + size * i) for i in range(h[count_name])]

    p['literalBuffer'] = [struct.unpack_from('<I', data, rel(h, 'literalBufferOffset') + 4 * i)[0]
                          for i in range(h['literalBufferCount'])]
    p['literals'] = table('literalCount', 'literalsOffset', 'HHI')
    p['bufferRefs'] = table('bufferRefCount', 'bufferRefsOffset', 'HHI')
    p['constantTextures'] = table('constantTextureCount', 'constantTexturesOffset', 'HH')
    p['textureRefs'] = table('textureRefCount', 'textureRefsOffset', 'HH')
    p['secondaryPdsData'] = table('secondaryPdsDataCount', 'secondaryPdsDataOffset', 'HHHH')
    return p


def disassemble(usedis, words):
    if not words:
        return []
    text = '\n'.join('%08X %08X' % w for w in words) + '\n'
    return subprocess.run([usedis, '-t', '7:216', '-'], input=text, capture_output=True,
                          text=True, check=True).stdout.splitlines()


def main():
    args = sys.argv[1:]
    usedis = None
    if args and args[0] == '--usedis':
        usedis, args = args[1], args[2:]
    p = parse(open(args[0], 'rb').read())
    for k, v in p['header'].items():
        print('  %-26s 0x%x' % (k, v & 0xffffffff))
    print('entry: instruction %d' % p['entry'])
    for prm in p['parameters']:
        print('param %-16s cat=%s type=%d comps=%d container=%d sem=%d/%d array=%d res=%d' % (
            prm['name'], CATEGORIES.get(prm['category'], prm['category']), prm['type'],
            prm['components'], prm['container'], prm['semantic'], prm['semanticIndex'],
            prm['arraySize'], prm['resourceIndex']))
    print('varyings:', {k: (hex(v) if isinstance(v, int) else v) for k, v in p['varyings'].items()})
    for key in ('literalBuffer', 'literals', 'bufferRefs', 'constantTextures', 'textureRefs',
                'secondaryPdsData'):
        if p[key]:
            print('%s: %s' % (key, p[key]))
    for name in ('primary', 'secondary'):
        print('%s code (%d):' % (name, len(p[name])))
        lines = disassemble(usedis, p[name]) if usedis else ['%08X %08X' % w for w in p[name]]
        for line in lines:
            print('   ' + line)


if __name__ == '__main__':
    main()
