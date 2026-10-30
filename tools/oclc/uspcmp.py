#!/usr/bin/env python3
"""Compare two compiled versions of the same kernel (e.g. IMG's USC and
PVR_PSP2's USC) and flag structural differences that suggest a miscompile:

  - constants the reference loads but the candidate never reads (literals
    excepted: either compiler may turn them into immediates)
  - secondary (SA update) code present in only one of them
  - large differences in the instruction mix

    uspcmp.py [--usedis PATH] [-t id:rev] reference.usp candidate.usp
"""
import collections
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uspbin  # noqa: E402

# Opcodes whose count may legitimately differ between compilers.
DIS_LINE = re.compile(r'^\s*\d+: [0-9A-F]{8} [0-9A-F]{8}\s+(.*)$')

NOISY = {'mov', 'vmov', 'or', 'nop', 'wdf', 'idf', 'br', 'ima32', 'imae', 'pcksh', 'unpckvec', 'vpck'}


def disassemble(usedis, target, p):
    lines = ['%08X %08X' % (hi, lo) for _, hi, lo in uspbin.code_words(p) if hi is not None]
    out = subprocess.run([usedis, '-t', target, '-'], input='\n'.join(lines) + '\n',
                         capture_output=True, text=True, check=True).stdout
    ops = []
    for line in out.splitlines():
        m = DIS_LINE.match(line)
        words = m.group(1).split() if m else []
        if not words:
            continue
        op = words[0]
        if op in ('p0', 'p1', 'p2', 'p3', '!p0', '!p1', '!p2', '!p3') and len(words) > 1:
            op = words[1]
        ops.append(op.split('.')[0])
    return ops


def loaded_constants(p):
    """Source constants (by UniFlex channel index or static value) moved to SA registers."""
    out = set()
    for load in p['regconsts'] + p['memconsts']:
        if load[0] == 'static':
            out.add(('static', load[1]))
        else:
            out.add(('c%d.%d' % (load[1] // 4, load[1] % 4)))
    return out


def static_channels(usp_path):
    """Literal constant channels listed in the decoded UniFlex dump that
    oclc's OCLC_DUMP writes next to the program (prefix.N.uf)."""
    uf = usp_path[:-len('.usp')] + '.uf'
    out = set()
    if os.path.exists(uf):
        for m in re.finditer(r'^;\s+(c\d+\.\d+) = ', open(uf).read(), re.M):
            out.add(m.group(1))
    return out


def main():
    args = sys.argv[1:]
    usedis, target = 'usedis', '8:115'
    while args and args[0].startswith('-'):
        if args[0] == '--usedis':
            usedis = args[1]
            args = args[2:]
        elif args[0] == '-t':
            target = args[1]
            args = args[2:]
        else:
            break
    ref = uspbin.parse(open(args[0], 'rb').read())
    cand = uspbin.parse(open(args[1], 'rb').read())
    flags = []

    rc, cc = loaded_constants(ref), loaded_constants(cand)
    # A static constant in one may be an immediate in the other; only
    # buffer-backed constants (kernel args, literals) must match.
    # Literals may also be folded into immediates or hardware constants.
    literals = static_channels(args[0])
    missing = sorted(c for c in rc - cc if not isinstance(c, tuple) and c not in literals)
    if missing:
        flags.append('constants read by reference but not candidate: %s' % ', '.join(missing))
    if bool(ref['sacode']) != bool(cand['sacode']):
        flags.append('SA update code: reference %d insts, candidate %d' % (len(ref['sacode']), len(cand['sacode'])))

    ro = collections.Counter(disassemble(usedis, target, ref))
    co = collections.Counter(disassemble(usedis, target, cand))
    mix = []
    for op in sorted(set(ro) | set(co)):
        if op in NOISY or op.startswith('#'):
            continue
        if ro[op] != co[op]:
            mix.append('%s %d->%d' % (op, ro[op], co[op]))
            if (ro[op] == 0) != (co[op] == 0):
                flags.append('opcode %s only in %s' % (op, 'reference' if co[op] == 0 else 'candidate'))
    print('insts %d->%d; mix: %s' % (sum(ro.values()), sum(co.values()), ', '.join(mix) or 'same'))
    for f in flags:
        print('RED FLAG: ' + f)
    return 1 if flags else 0


if __name__ == '__main__':
    sys.exit(main())
