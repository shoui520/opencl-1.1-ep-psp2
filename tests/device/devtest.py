#!/usr/bin/env python3
"""Device tests through the OpenCL API, checked against a CPU OpenCL (pocl).

    devtest.py build WORKDIR [CASE ...]   write WORKDIR/tests.pak and the expected
                                          results (WORKDIR/expected/)
    devtest.py check WORKDIR RESULTS      compare the device's results.pak

A test is a program binary (tools/oclc/clbin.py, which runs the host compiler
driver as $OCLC) and a script of OpenCL calls.  The harness (tests/harness)
runs the script with the runtime (runtime/cl); pocl runs it here, through
pyopencl, from the source.  What the scripts read back and every call's
return code are compared.  Kernels run with work-groups of one item on both
sides, which is all the device supports.

Pack: "VCP2", count, offsets[count], then the tests, each:
    "VCT2", size, name[32], binary offset, binary size, op count, ops offset,
    data offset, data size, then the ops, OP_WORDS words each: opcode, wait
    (an event slot + 1, 0: none), event (the slot + 1 to keep the command's
    event in, 0: none), then the operands of OPS; "data" operands are
    offsets into the test's data.
Results: "VCR2", count, then per test: status, kernel time (us), guard
    errors, op count, byte count, every op's return code, then the outputs
    (the read-backs and statuses), in op order.
All words are 32-bit little-endian; offsets are from the test's start.
"""
import json
import os
import struct
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'oclc'))
import clbin  # noqa: E402

OP_WORDS = 24
NO_DATA = 0xffffffff
NULL_ID = 0xffffffff

# name: (opcode, operands)
OPS = {
    'buffer': (1, 'id flags size data'),		# a guarded buffer, data (all of it) or none
    'subbuffer': (2, 'id parent flags origin size'),
    'kernel': (3, 'kid name name_size'),
    'arg_mem': (4, 'kid index id'),			# id NULL_ID: a NULL buffer
    'arg_value': (5, 'kid index data size'),
    'arg_local': (6, 'kid index size'),
    'ndrange': (7, 'kid dims g0 g1 g2 o0 o1 o2 local has_offset'),	# local 1: ones, 0: NULL
    'read': (8, 'id offset size blocking'),			# output: size bytes
    'write': (9, 'id offset size data blocking'),
    'copy': (10, 'src dst src_offset dst_offset size'),
    'read_rect': (11, 'id bo0 bo1 bo2 ho0 ho1 ho2 r0 r1 r2 brow bslice hrow hslice '
                      'host_size data blocking'),		# output: host_size bytes
    'write_rect': (12, 'id bo0 bo1 bo2 ho0 ho1 ho2 r0 r1 r2 brow bslice hrow hslice '
                       'host_size data blocking'),
    'copy_rect': (13, 'src dst so0 so1 so2 do0 do1 do2 r0 r1 r2 srow sslice drow dslice'),
    'map': (14, 'id flags offset size data blocking'),	# output (CL_MAP_READ): size bytes
    'user_event': (15, ''),
    'set_status': (16, 'slot status'),
    'status': (17, 'slot'),					# output: 4 bytes
    'finish': (18, ''),
    'wait': (19, 'slot'),
    'release': (20, 'id'),
    'ep_api': (21, ''),					# output: failures, then EP_API_REPORT bytes of text
}
EP_API_REPORT = 4096

CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY = 1, 2, 4
CL_MEM_USE_HOST_PTR, CL_MEM_ALLOC_HOST_PTR = 8, 16
CL_MAP_READ, CL_MAP_WRITE = 1, 2


class Buf:
    """A __global or __constant buffer: count elements of dtype."""

    def __init__(self, dtype, count, init='rand', lo=None, hi=None, values=None):
        self.dtype = np.dtype(dtype)
        self.count = count
        self.init, self.lo, self.hi, self.values = init, lo, hi, values

    def make(self, rng):
        dt = self.dtype
        base = dt.base if dt.subdtype else dt
        shape = (self.count,) + (dt.shape if dt.subdtype else ())
        if self.values is not None:
            a = np.zeros(shape, base)
            v = np.asarray(self.values, base).reshape(-1)
            a.reshape(-1)[:len(v)] = v[:a.size]
            return a
        if callable(self.init):
            return np.asarray(self.init(rng, shape), base).reshape(shape)
        if self.init == 'zeros':
            return np.zeros(shape, base)
        if self.init == 'iota':
            return np.arange(np.prod(shape)).astype(base).reshape(shape)
        if base.kind == 'f':
            lo = -100.0 if self.lo is None else self.lo
            hi = 100.0 if self.hi is None else self.hi
            return rng.uniform(lo, hi, shape).astype(base)
        info = np.iinfo(base)
        lo = info.min if self.lo is None else self.lo
        hi = info.max if self.hi is None else self.hi
        return rng.integers(lo, hi, shape, endpoint=True, dtype=np.int64).astype(base)


class Val:
    """A by-value argument: a numpy scalar, vector or structured record."""

    def __init__(self, value):
        self.value = value

    def raw(self):
        return np.asarray(self.value).tobytes()


class Local:
    """A __local pointer argument of nbytes."""

    def __init__(self, nbytes):
        self.nbytes = nbytes


class Case:
    """A kernel run: the Buf arguments are read back and compared."""

    def __init__(self, name, src, kernel, global_size, args, offset=None, ulp=0, atol=0.0,
                 check=None, skip_compare=(), options=''):
        self.name, self.src, self.kernel = name, src, kernel
        self.global_size = tuple(global_size)
        self.offset = tuple(offset) if offset else None
        self.args = args
        self.ulp, self.atol = ulp, atol
        self.check = check		# f(inputs, ref, dev) -> [error strings]
        self.skip_compare = set(skip_compare)	# argument indices check() covers
        self.options = options


class ApiCase:
    """A script of OpenCL calls, script(s, rng) (src: its kernels, or None);
    outputs and return codes must match exactly."""

    def __init__(self, name, src, script, options='', check=None):
        self.name, self.src, self.script, self.options = name, src, script, options
        self.check = check		# f(outputs) -> [error strings], for outputs not compared


class Script:
    def __init__(self):
        self.ops = []		# (name, operands, wait, event, compare)

    def add(self, op, wait=None, event=None, compare=True, **kw):
        self.ops.append((op, kw, wait, event, compare))

    def buffer(self, id, data, flags=CL_MEM_READ_WRITE, **kw):
        self.add('buffer', id=id, flags=flags, size=len(data), data=bytes(data), **kw)

    def kernel(self, kid, name):
        self.add('kernel', kid=kid, name=name)

    def ndrange(self, kid, global_size, offset=None, local=1, **kw):
        g = list(global_size) + [1] * (3 - len(global_size))
        o = list(offset or ()) + [0] * (3 - len(offset or ()))
        self.add('ndrange', kid=kid, dims=len(global_size), g0=g[0], g1=g[1], g2=g[2],
                 o0=o[0], o1=o[1], o2=o[2], local=local, has_offset=int(offset is not None), **kw)

    def read(self, id, size, offset=0, blocking=1, **kw):
        self.add('read', id=id, offset=offset, size=size, blocking=blocking, **kw)


def output_size(name, kw):
    if name == 'read':
        return kw['size']
    if name == 'read_rect':
        return kw['host_size']
    if name == 'map':
        return kw['size'] if kw['flags'] & CL_MAP_READ else 0
    if name == 'status':
        return 4
    if name == 'ep_api':
        return 4 + EP_API_REPORT
    return 0


def encode(script):
    """(ops words, data) of a script."""
    data = bytearray()

    def put(b):
        off = len(data)
        data.extend(b)
        data.extend(bytes(-len(data) % 16))
        return off
    words = []
    for name, kw, wait, event, _ in script.ops:
        code, operands = OPS[name]
        w = [code, 0 if wait is None else wait + 1, 0 if event is None else event + 1]
        kw = dict(kw)
        if name == 'kernel':
            kw['name_size'] = len(kw['name'].encode())
            kw['name'] = put(kw['name'].encode() + b'\0')
        if 'data' in kw:
            kw['data'] = NO_DATA if kw['data'] is None else put(kw['data'])
        for o in operands.split():
            v = kw.pop(o, 0)
            w.append(v & 0xffffffff)
        if kw:
            raise ValueError('%s: unknown operands %s' % (name, list(kw)))
        words += w + [0] * (OP_WORDS - len(w))
    return words, bytes(data)


def align(n, a=16):
    return (n + a - 1) & ~(a - 1)


def bundle(name, binary, script):
    words, data = encode(script)
    head = 4 + 4 + 32 + 4 * 6
    ops_off = align(head)
    bin_off = align(ops_off + 4 * len(words))
    data_off = align(bin_off + len(binary))
    size = align(data_off + len(data))
    out = bytearray(size)
    struct.pack_into('<4sI32s6I', out, 0, b'VCT2', size, name.encode()[:31], bin_off,
                     len(binary), len(script.ops), ops_off, data_off, len(data))
    struct.pack_into('<%dI' % len(words), out, ops_off, *words)
    out[bin_off:bin_off + len(binary)] = binary
    out[data_off:data_off + len(data)] = data
    return bytes(out)


def run_reference(script, src, options):
    """Run a script on pocl; returns (codes, outputs)."""
    import pyopencl as cl
    plat = [p for p in cl.get_platforms() if 'Portable' in p.name or 'pocl' in p.name.lower()]
    ctx = cl.Context(plat[0].get_devices())
    q = cl.CommandQueue(ctx)
    prog = cl.Program(ctx, src).build(options='-cl-std=CL1.1 ' + options) if src else None
    bufs, kernels, events, keep = {}, {}, {}, []
    codes, outputs = [], []
    mf = cl.mem_flags

    def raw(b):
        return np.frombuffer(b, np.dtype((np.void, len(b))))[0]
    for name, kw, wait, event, _ in script.ops:
        wait_for = [events[wait]] if wait is not None else None
        out = None
        ev = None
        try:
            if name == 'buffer':
                host = np.frombuffer(kw['data'], np.uint8).copy() if kw['data'] else None
                flags = kw['flags']
                if host is not None and not flags & CL_MEM_USE_HOST_PTR:
                    flags |= mf.COPY_HOST_PTR
                keep.append(host)
                bufs[kw['id']] = cl.Buffer(ctx, flags, kw['size'], hostbuf=host)
            elif name == 'subbuffer':
                bufs[kw['id']] = bufs[kw['parent']].get_sub_region(kw['origin'], kw['size'],
                                                                   kw['flags'])
            elif name == 'kernel':
                kernels[kw['kid']] = cl.Kernel(prog, kw['name'])
            elif name == 'arg_mem':
                kernels[kw['kid']].set_arg(kw['index'], None if kw['id'] == NULL_ID
                                           else bufs[kw['id']])
            elif name == 'arg_value':
                kernels[kw['kid']].set_arg(kw['index'], raw(kw['data']))
            elif name == 'arg_local':
                kernels[kw['kid']].set_arg(kw['index'], cl.LocalMemory(kw['size']))
            elif name == 'ndrange':
                d = kw['dims']
                g = (kw['g0'], kw['g1'], kw['g2'])[:d]
                o = (kw['o0'], kw['o1'], kw['o2'])[:d] if kw['has_offset'] else None
                ev = cl.enqueue_nd_range_kernel(q, kernels[kw['kid']], g, (1,) * d,
                                                global_work_offset=o, wait_for=wait_for)
            elif name == 'read':
                out = np.zeros(kw['size'], np.uint8)
                ev = cl.enqueue_copy(q, out, bufs[kw['id']], src_offset=kw['offset'],
                                     is_blocking=bool(kw['blocking']), wait_for=wait_for)
            elif name == 'write':
                host = np.frombuffer(kw['data'], np.uint8)[:kw['size']].copy()
                keep.append(host)
                ev = cl.enqueue_copy(q, bufs[kw['id']], host, dst_offset=kw['offset'],
                                     is_blocking=bool(kw['blocking']), wait_for=wait_for)
            elif name in ('read_rect', 'write_rect'):
                host = (np.frombuffer(kw['data'], np.uint8).copy() if kw['data']
                        else np.zeros(kw['host_size'], np.uint8))
                rect = dict(buffer_origin=(kw['bo0'], kw['bo1'], kw['bo2']),
                            host_origin=(kw['ho0'], kw['ho1'], kw['ho2']),
                            region=(kw['r0'], kw['r1'], kw['r2']),
                            buffer_pitches=(kw['brow'], kw['bslice']),
                            host_pitches=(kw['hrow'], kw['hslice']),
                            is_blocking=bool(kw['blocking']), wait_for=wait_for)
                if name == 'read_rect':
                    out = host
                    ev = cl.enqueue_copy(q, host, bufs[kw['id']], **rect)
                else:
                    keep.append(host)
                    ev = cl.enqueue_copy(q, bufs[kw['id']], host, **rect)
            elif name == 'copy':
                ev = cl.enqueue_copy(q, bufs[kw['dst']], bufs[kw['src']], byte_count=kw['size'],
                                     src_offset=kw['src_offset'], dst_offset=kw['dst_offset'],
                                     wait_for=wait_for)
            elif name == 'copy_rect':
                ev = cl.enqueue_copy(q, bufs[kw['dst']], bufs[kw['src']],
                                     src_origin=(kw['so0'], kw['so1'], kw['so2']),
                                     dst_origin=(kw['do0'], kw['do1'], kw['do2']),
                                     region=(kw['r0'], kw['r1'], kw['r2']),
                                     src_pitches=(kw['srow'], kw['sslice']),
                                     dst_pitches=(kw['drow'], kw['dslice']), wait_for=wait_for)
            elif name == 'map':
                a, ev = cl.enqueue_map_buffer(q, bufs[kw['id']], kw['flags'], kw['offset'],
                                              (kw['size'],), np.uint8, is_blocking=True,
                                              wait_for=wait_for)
                if kw['flags'] & CL_MAP_READ:
                    out = a.copy()
                if kw['flags'] & CL_MAP_WRITE and kw['data']:
                    a[:] = np.frombuffer(kw['data'], np.uint8)[:kw['size']]
                a.base.release(q)
            elif name == 'user_event':
                ev = cl.UserEvent(ctx)
            elif name == 'set_status':
                events[kw['slot']].set_status(kw['status'])
            elif name == 'status':
                out = np.array([events[kw['slot']].command_execution_status], np.int32).view(
                    np.uint8)
            elif name == 'finish':
                q.finish()
            elif name == 'wait':
                cl.wait_for_events([events[kw['slot']]])
            elif name == 'release':
                bufs.pop(kw['id']).release()
            elif name == 'ep_api':
                pass			# the device checks itself
            codes.append(0)
        except cl.Error as e:
            codes.append(e.code)
        if event is not None and ev is not None:
            events[event] = ev
        if output_size(name, kw):
            outputs.append(out if out is not None else np.zeros(output_size(name, kw), np.uint8))
    q.finish()
    return codes, [o.tobytes() for o in outputs]


def case_script(case, meta_args, inputs):
    if len(meta_args) != len(case.args):
        raise RuntimeError('%s: %d arguments in the kernel, %d in the case' % (
            case.name, len(meta_args), len(case.args)))
    s = Script()
    for i, a in enumerate(case.args):
        if isinstance(a, Buf):
            s.buffer(i, inputs[i].tobytes())
    s.kernel(0, case.kernel)
    for i, a in enumerate(case.args):
        if isinstance(a, Buf):
            s.add('arg_mem', kid=0, index=i, id=i)
        elif isinstance(a, Val):
            s.add('arg_value', kid=0, index=i, data=a.raw(), size=len(a.raw()))
        elif isinstance(a, Local):
            s.add('arg_local', kid=0, index=i, size=a.nbytes)
    s.ndrange(0, case.global_size, case.offset)
    for i, a in enumerate(case.args):
        if isinstance(a, Buf):
            s.read(i, inputs[i].nbytes)
    return s


def build(work, names):
    from cases import CASES
    cases = [c for c in CASES if not names or c.name in names]
    binaries = {}
    bundles = []
    exp_dir = os.path.join(work, 'expected')
    os.makedirs(exp_dir, exist_ok=True)
    manifest = []
    for case in cases:
        src = open(os.path.join(ROOT, case.src)).read() if case.src else None
        key = (case.src, case.options)
        if src and key not in binaries:
            base = os.path.splitext(os.path.basename(case.src))[0]
            try:
                kernels = clbin.compile_kernels(src, os.path.join(work, 'build'), case.options,
                                                base)
                binaries[key] = (kernels, {k['name']: k for k in kernels})
            except clbin.CompileError as e:
                binaries[key] = e
        b = binaries.get(key, ([], {}))
        if isinstance(b, Exception):
            print('%-24s SKIP: %s' % (case.name, b))
            continue
        program, kernels = b
        binary = clbin.pack(program) if program else b''
        rng = np.random.default_rng(zlib.crc32(case.name.encode()))
        inputs = {}
        if isinstance(case, Case):
            if case.kernel not in kernels:
                print('%-24s SKIP: kernel not found' % case.name)
                continue
            # a kernel case only creates its kernel: its binary needs no others
            binary = clbin.pack([kernels[case.kernel]])
            inputs = {i: a.make(rng) for i, a in enumerate(case.args) if isinstance(a, Buf)}
            script = case_script(case, kernels[case.kernel]['args'], inputs)
        else:
            script = Script()
            case.script(script, rng)
        codes, outputs = run_reference(script, src, case.options)
        bundles.append(bundle(case.name, binary, script))
        np.savez(os.path.join(exp_dir, case.name + '.npz'),
                 codes=np.array(codes, np.int32),
                 compare=np.array([op[4] for op in script.ops], bool),
                 out_compare=np.array([op[4] for op in script.ops
                                       if output_size(op[0], op[1])], bool),
                 **{'in%d' % i: v for i, v in inputs.items()},
                 **{'out%d' % i: np.frombuffer(o, np.uint8) for i, o in enumerate(outputs)})
        manifest.append(case.name)
        print('%-24s built (%d bytes, %d ops)' % (case.name, len(bundles[-1]), len(script.ops)))
    offs, pos = [], align(8 + 4 * len(bundles))
    for b in bundles:
        offs.append(pos)
        pos = align(pos + len(b))
    pak = bytearray(pos)
    struct.pack_into('<4sI%dI' % len(offs), pak, 0, b'VCP2', len(offs), *offs)
    for o, b in zip(offs, bundles):
        pak[o:o + len(b)] = b
    open(os.path.join(work, 'tests.pak'), 'wb').write(pak)
    json.dump(manifest, open(os.path.join(work, 'manifest.json'), 'w'))
    print('%d cases, %d bytes' % (len(bundles), len(pak)))


def ulp_diff(a, b):
    """Distance in units in the last place between float32 arrays."""
    ia = a.view(np.int32).astype(np.int64)
    ib = b.view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, -(ia & 0x7fffffff), ia)
    ib = np.where(ib < 0, -(ib & 0x7fffffff), ib)
    return np.abs(ia - ib)


def compare(case, i, ref, dev):
    errs = []
    base = ref.dtype.base if ref.dtype.subdtype else ref.dtype
    r = ref.reshape(-1)
    d = dev.reshape(-1)
    if base.kind == 'f':
        with np.errstate(invalid='ignore'):
            both_nan = np.isnan(r) & np.isnan(d)
            close = np.abs(r.astype(np.float64) - d.astype(np.float64)) <= case.atol
        bad = ~(both_nan | (r == d) | close | (ulp_diff(r, d) <= case.ulp))
    else:
        bad = r != d
    for j in np.nonzero(bad)[0][:6]:
        errs.append('arg%d[%d]: device %r, expected %r' % (i, j, d[j], r[j]))
    if bad.any():
        errs.append('arg%d: %d of %d wrong' % (i, int(bad.sum()), len(r)))
    return errs


def check_case(case, exp, outs):
    """Compare a kernel case's read-backs, which are its Buf arguments in order."""
    errs = []
    dev, refs = {}, {}
    bufs = [i for i, a in enumerate(case.args) if isinstance(a, Buf)]
    inputs = {int(k[2:]): exp[k] for k in exp.files if k.startswith('in')}
    for n, i in enumerate(bufs):
        like = inputs[i]
        dt = like.dtype.base if like.dtype.subdtype else like.dtype
        dev[i] = np.frombuffer(outs[n].tobytes(), dt).reshape(like.shape)
        refs[i] = np.frombuffer(exp['out%d' % n].tobytes(), dt).reshape(like.shape)
        if i not in case.skip_compare:
            errs += compare(case, i, refs[i], dev[i])
    if case.check:
        errs += case.check(inputs, refs, dev)
    return errs


def check(work, results_path):
    from cases import CASES
    by_name = {c.name: c for c in CASES}
    manifest = json.load(open(os.path.join(work, 'manifest.json')))
    data = open(results_path, 'rb').read()
    magic, count = struct.unpack_from('<4sI', data, 0)
    if magic != b'VCR2':
        sys.exit('not a results pack')
    pos = 8
    failed = 0
    for n in range(count):
        status, time_us, guard, nops, nbytes = struct.unpack_from('<iIIII', data, pos)
        pos += 20
        codes = np.frombuffer(data, np.int32, nops, pos)
        pos += 4 * nops
        blob = data[pos:pos + nbytes]
        pos += nbytes
        name = manifest[n]
        case = by_name[name]
        exp = np.load(os.path.join(work, 'expected', name + '.npz'))
        errs = []
        if guard:
            errs.append('%d guard bytes overwritten' % guard)
        if status:
            errs.append('status 0x%08x' % (status & 0xffffffff))
        want = exp['codes']
        for j in np.nonzero((codes != want) & exp['compare'])[0][:6]:
            errs.append('op %d returned %d, pocl %d' % (j, codes[j], want[j]))
        outs, off = [], 0
        for j in range(len(exp['out_compare'])):
            size = exp['out%d' % j].size
            outs.append(np.frombuffer(blob[off:off + size], np.uint8))
            off += size
        if off != len(blob):
            errs.append('%d output bytes, want %d' % (len(blob), off))
        elif isinstance(case, Case):
            errs += check_case(case, exp, outs)
        else:
            for j, cmp in enumerate(exp['out_compare']):
                if cmp and outs[j].tobytes() != exp['out%d' % j].tobytes():
                    bad = np.nonzero(outs[j] != exp['out%d' % j])[0]
                    errs.append('output %d: %d bytes differ, first at %d (device %d, pocl %d)' % (
                        j, len(bad), bad[0], outs[j][bad[0]], exp['out%d' % j][bad[0]]))
            if case.check:
                errs += case.check(outs)
        print('%-24s %s %8.3f ms' % (name, 'FAIL' if errs else 'ok  ', time_us / 1000.0))
        for e in errs:
            print('    ' + e)
        failed += bool(errs)
    print('%d of %d failed' % (failed, count))
    if count < len(manifest):
        print('%d of %d did not run: %s...' % (len(manifest) - count, len(manifest),
                                              ', '.join(manifest[count:count + 4])))
        failed += len(manifest) - count
    return failed


def main():
    # cases.py imports the classes from "devtest": make that this module
    sys.modules.setdefault('devtest', sys.modules[__name__])
    sys.path.insert(0, HERE)
    if len(sys.argv) >= 3 and sys.argv[1] == 'build':
        build(sys.argv[2], sys.argv[3:])
    elif len(sys.argv) == 4 and sys.argv[1] == 'check':
        sys.exit(1 if check(sys.argv[2], sys.argv[3]) else 0)
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
