#!/usr/bin/env python3
"""Compile OpenCL C for the PS Vita into a program binary, which the runtime
(runtime/cl) loads with clCreateProgramWithBinary.

    clbin.py [-o out.bin] [-O "build options"] [-w workdir] file.cl

The host compiler driver (tools/oclc/oclc.c, under qemu) is run as $OCLC
(default "oclc"); tools/oclc/setup.sh builds it as build/oclc/oclc.  Kernel-scope __local variables are moved into a hidden
argument first (locals.py), and each kernel's USP output is packaged as a
GXM vertex program (tools/gxp/mkgxp.py).  The format is described in
runtime/cl/program.c.
"""
import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'gxp'))
import hwshader  # noqa: E402
import locals as cl_locals  # noqa: E402
import mkgxp  # noqa: E402

MAGIC = b'VCLB'
VERSION = 1
KERNEL_WORDS = 20
ARG_WORDS = 8
ADDR_VALUE, ADDR_LOCAL = 0, 1

FIRST_ARG_REG = 20		# runtime/gpu/gxc_kernel.h GXC_C_ARGS
REQD = re.compile(r'reqd_work_group_size\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)')


class CompileError(Exception):
    pass


def reqd_sizes(src):
    """{kernel: (x, y, z)} for the kernels declared with reqd_work_group_size."""
    out = {}
    for m in re.finditer(r'\b(?:__)?kernel\b', src):
        # the attribute may come before or after the qualifier
        start = max(src.rfind(';', 0, m.start()), src.rfind('}', 0, m.start())) + 1
        name = re.compile(r'\bvoid\s+(?:__attribute__\s*\(\(.*?\)\)\s*)?(\w+)\s*\(', re.S)
        n = name.search(src, m.end())
        if not n:
            continue
        r = REQD.search(src, start, n.start())
        if r:
            out[n.group(1)] = tuple(int(x) for x in r.groups())
    return out


def compact_channels(hw, k):
    """The front end numbers the constant registers of the whole program, so a
    program of many kernels can pass the runtime's table: number the ones a
    kernel uses from c20 (GXC_C_ARGS, after the work-item constants) up, in
    order, so an argument of several registers stays in one piece."""
    # every kernel's static constants list the whole program's: the shader
    # reads constants only through its loads, so keep the ones it loads
    loaded = {l['src'] for l in hw['reg_consts'] if l['fmt'] == mkgxp.CONST_F32}
    hw['static_consts'] = {c: v for c, v in hw['static_consts'].items() if c in loaded}
    regs = {c // 4 for c in loaded}
    for space, creg, elem, comps, _ in k['args']:
        regs.update(range(creg, creg + (comps + 3) // 4))
    regs.update(c // 4 for _, c in k['private'])
    if k['hidden']:
        regs.add(k['hidden'][0])
    used = sorted(r for r in regs if r >= FIRST_ARG_REG)
    new = {r: FIRST_ARG_REG + i for i, r in enumerate(used)}

    def ch(c):
        return new[c // 4] * 4 + c % 4 if c // 4 in new else c
    for l in hw['reg_consts']:
        if l['fmt'] == mkgxp.CONST_F32:
            l['src'] = ch(l['src'])
    hw['static_consts'] = {ch(c): v for c, v in hw['static_consts'].items()}
    k['args'] = [(space, new.get(creg, creg), elem, comps, ref)
                 for space, creg, elem, comps, ref in k['args']]
    k['private'] = [(size, ch(c)) for size, c in k['private']]
    if k['hidden']:
        k['hidden'] = (new.get(k['hidden'][0], k['hidden'][0]), k['hidden'][1])


def compile_kernels(src, work, options='', base='program'):
    """Compile OpenCL C source text; returns the kernels in compile order, each
    {name, gxp, map, args: [(space, creg, elem, comps, by_ref)],
     private: [(bytes, channel)], hidden: (creg, size) or None, reqd}."""
    rewritten, local_info = cl_locals.rewrite(src)
    os.makedirs(work, exist_ok=True)
    cl_path = os.path.join(work, base + '.cl')
    with open(cl_path, 'w') as f:
        f.write(rewritten)
    prefix = os.path.join(work, base)
    meta_path = prefix + '.meta'
    if os.path.exists(meta_path):
        os.remove(meta_path)
    env = dict(os.environ, OCLC_GXM='1', OCLC_DUMP=prefix)
    cmd = os.environ.get('OCLC', 'oclc').split() + ['-m', meta_path]
    if options:
        cmd += ['-O', options]
    r = subprocess.run(cmd + [cl_path], env=env, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(meta_path):
        raise CompileError('compile failed:\n%s%s' % (r.stdout[-4000:], r.stderr[-4000:]))
    kernels, cur = [], None
    for line in open(meta_path):
        f = line.split()
        if f[0] == 'kernel':
            cur = {'name': f[1], 'program': int(f[3]), 'args': [], 'private': []}
            kernels.append(cur)
        elif f[0] == 'arg':
            w = [int(x) for x in f[2:]]
            space, creg, elem, comps, ptr = w[2], w[3], w[4], w[5], w[6]
            cur['args'].append((space, creg, elem, comps, int(space == ADDR_VALUE and ptr != 0)))
        elif f[0] == 'private':
            cur['private'].append((int(f[2]), int(f[3])))
    reqd = reqd_sizes(src)
    for k in kernels:
        if k['program'] < 0:
            raise CompileError('%s: no program for the kernel' % k['name'])
        hw = hwshader.parse(open('%s.%d.hw' % (prefix, k['program']), 'rb').read())
        # the front end lists every private array of the program for each
        # kernel: keep the ones whose base this kernel loads
        loaded = {l['src'] for l in hw['reg_consts'] if l['fmt'] == mkgxp.CONST_F32}
        k['private'] = [p for p in k['private'] if p[1] in loaded]
        k['hidden'] = None
        if local_info.get(k['name']):
            # locals.py's argument is the last one
            space, creg, elem, comps, _ = k['args'].pop()
            if space != ADDR_LOCAL:
                raise CompileError('%s: the hidden __local argument is missing' % k['name'])
            k['hidden'] = (creg, elem)
        compact_channels(hw, k)
        try:
            k['gxp'] = mkgxp.build(hw)
        except mkgxp.PackError as e:
            raise CompileError('%s: %s' % (k['name'], e))
        k['map'] = mkgxp.const_map(hw)
        k['reqd'] = reqd.get(k['name'], (0, 0, 0))
    return kernels


def align(n, a=16):
    return (n + a - 1) & ~(a - 1)


def pack(kernels):
    """The program binary for compile_kernels' output."""
    head = 16 + KERNEL_WORDS * 4 * len(kernels)
    blobs = bytearray()

    def put(data, a=16):
        nonlocal blobs
        blobs += bytes(align(head + len(blobs), a) - head - len(blobs))
        off = head + len(blobs)
        blobs += data
        return off

    records = []
    for k in kernels:
        name = put(k['name'].encode() + b'\0', 4)
        gxp = put(k['gxp'])
        kmap = put(k['map'], 4)
        args = put(b''.join(struct.pack('<8I', *a, 0, 0, 0) for a in k['args']), 4)
        priv = put(b''.join(struct.pack('<2I', *p) for p in k['private']), 4)
        hcreg, hsize = k['hidden'] or (0, 0)
        records.append(struct.pack('<20I', name, len(k['name'].encode()), gxp, len(k['gxp']),
                                   kmap, len(k['map']), len(k['args']), args,
                                   len(k['private']), priv, hcreg, hsize, *k['reqd'],
                                   0, 0, 0, 0, 0))
    size = align(head + len(blobs), 4)
    out = struct.pack('<4s3I', MAGIC, VERSION, len(kernels), size) + b''.join(records) + blobs
    return out + bytes(size - len(out))


def build(src, options='', work=None, base='program'):
    """Compile source text to a program binary."""
    if work:
        return pack(compile_kernels(src, work, options, base))
    with tempfile.TemporaryDirectory() as tmp:
        return pack(compile_kernels(src, tmp, options, base))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('source')
    ap.add_argument('-o', dest='output')
    ap.add_argument('-O', dest='options', default='')
    ap.add_argument('-w', dest='work', help='keep the intermediate files here')
    a = ap.parse_args()
    base = os.path.splitext(os.path.basename(a.source))[0]
    try:
        binary = build(open(a.source).read(), a.options, a.work, base)
    except CompileError as e:
        sys.exit('%s: %s' % (a.source, e))
    with open(a.output or base + '.clbin', 'wb') as f:
        f.write(binary)


if __name__ == '__main__':
    main()
