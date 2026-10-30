#!/usr/bin/env python3
"""Move kernel-scope __local variables into a hidden kernel argument.

IMG's front end miscompiles __local variables declared in a kernel body (each
kernel of a program gets address code for every such variable, chained
through registers that other kernels define, and a scalar whose address is
taken shares a register with its value), while __local pointer arguments
work.  So for each kernel that declares any, this rewrites

    __kernel void k(__global int *p) { __local float t[16]; ... }

to

    struct __vita_locals_k { float t[16]; };
    __kernel void k(__global int *p, __local struct __vita_locals_k *__vita_locals)
    { #define t (__vita_locals->t) ... } #undef t

and the runtime passes the struct as a local argument.  Line numbers are
kept with #line.  OpenCL C 1.1 only allows these variables at kernel function
scope and without initialisers, which is what this looks for.

    locals.py file.cl > rewritten.cl
"""
import re
import sys

TOKEN = re.compile(r'''
    (?P<ws>[ \t\r\f\v]+|\\\n)
  | (?P<nl>\n)
  | (?P<comment>//[^\n]*|/\*.*?\*/)
  | (?P<str>"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')
  | (?P<id>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<num>\.?[0-9](?:[eEpP][+-]|[A-Za-z_0-9.])*)
  | (?P<punct>->|\+\+|--|<<=|>>=|<<|>>|<=|>=|==|!=|&&|\|\||[-+*/%&|^]=|\#\#|.)
''', re.S | re.X)

LOCAL = ('__local', 'local')
KERNEL = ('__kernel', 'kernel')
HIDDEN_ARG = '__vita_locals'


class Tok:
    __slots__ = ('kind', 'text', 'pos', 'line')

    def __init__(self, kind, text, pos, line):
        self.kind, self.text, self.pos, self.line = kind, text, pos, line


def tokenize(src):
    """Tokens of src outside preprocessor directives, with 1-based lines."""
    toks = []
    pos, line, bol = 0, 1, True
    while pos < len(src):
        if bol:
            m = re.match(r'[ \t]*#', src[pos:pos + 200])
            if m:
                # skip the directive, including continuation lines
                end = pos
                while True:
                    nl = src.find('\n', end)
                    if nl < 0:
                        end = len(src)
                        break
                    if src[nl - 1] == '\\':
                        end = nl + 1
                        continue
                    end = nl
                    break
                line += src.count('\n', pos, end)
                pos = end
                continue
        m = TOKEN.match(src, pos)
        kind = m.lastgroup
        text = m.group()
        if kind not in ('ws', 'nl', 'comment'):
            toks.append(Tok(kind, text, pos, line))
            bol = False
        elif kind == 'nl':
            bol = True
        line += text.count('\n')
        pos = m.end()
    return toks


def match_close(toks, i):
    """Index of the bracket closing toks[i]."""
    pairs = {'(': ')', '[': ']', '{': '}'}
    want = [pairs[toks[i].text]]
    for j in range(i + 1, len(toks)):
        t = toks[j].text
        if t in pairs:
            want.append(pairs[t])
        elif want and t == want[-1]:
            want.pop()
            if not want:
                return j
    raise SyntaxError('unbalanced %r at line %d' % (toks[i].text, toks[i].line))


def split_top(toks, sep):
    """Split a token list at top-level separators."""
    parts, cur, depth = [], [], 0
    for t in toks:
        if t.text in '([{':
            depth += 1
        elif t.text in ')]}':
            depth -= 1
        if depth == 0 and t.text == sep:
            parts.append(cur)
            cur = []
        else:
            cur.append(t)
    parts.append(cur)
    return parts


class Kernel:
    def __init__(self):
        self.name = None
        self.start = None	# first token of the definition
        self.params = None	# (open, close) token indices
        self.body = None	# (open, close) token indices
        self.locals = []	# (decl token range, [(specifier text, name, dims text)])


def text_of(src, toks):
    return src[toks[0].pos:toks[-1].pos + len(toks[-1].text)] if toks else ''


def find_kernels(src, toks):
    kernels = []
    i, depth = 0, 0
    while i < len(toks):
        t = toks[i]
        if t.text == '{':
            depth += 1
        elif t.text == '}':
            depth -= 1
        elif depth == 0 and t.text in KERNEL:
            k = Kernel()
            k.start = i
            j = i + 1
            while j < len(toks):
                if toks[j].text == '__attribute__':
                    j = match_close(toks, j + 1) + 1
                    continue
                if toks[j].kind == 'id' and toks[j + 1].text == '(':
                    break
                j += 1
            k.name = toks[j].text
            k.params = (j + 1, match_close(toks, j + 1))
            j = k.params[1] + 1
            while toks[j].text == '__attribute__':
                j = match_close(toks, j + 1) + 1
            if toks[j].text == '{':
                k.body = (j, match_close(toks, j))
                find_locals(src, toks, k)
                kernels.append(k)
                i = k.body[1] + 1
                continue
            i = j
            continue
        i += 1
    return kernels


def find_locals(src, toks, k):
    """Kernel-scope declarations of __local variables (not pointers to them)."""
    i = k.body[0] + 1
    end = k.body[1]
    while i < end:
        if toks[i].text == '{':
            i = match_close(toks, i) + 1
            continue
        # a statement at kernel scope: up to ';' or a nested block
        j = i
        while j < end and toks[j].text not in (';', '{'):
            if toks[j].text in ('(', '['):
                j = match_close(toks, j)
            j += 1
        if j < end and toks[j].text == ';':
            stmt = toks[i:j]
            if any(t.text in LOCAL for t in stmt[:4]) and stmt[0].text not in ('return',):
                decls = parse_local_decl(src, stmt)
                if decls:
                    k.locals.append(((i, j), decls))
            i = j + 1
        else:
            i = j
    return k


def parse_local_decl(src, stmt):
    """[(specifiers, name, dims)] for a __local variable declaration, or None
    if it declares pointers (private variables pointing to local memory)."""
    chunks = split_top(stmt, ',')
    for c in chunks:
        if any(t.text == '*' for t in top_level(c)):
            if len(chunks) > 1:
                raise SyntaxError('line %d: mixed __local variable and pointer declarations '
                                  'are not supported' % stmt[0].line)
            return None
        if any(t.text == '=' for t in top_level(c)):
            raise SyntaxError('line %d: __local variables cannot be initialised' % stmt[0].line)
    decls = []
    specs = None
    for n, c in enumerate(chunks):
        # the declarator name is the last identifier before the first '['
        b = next((x for x, t in enumerate(c) if t.text == '['), len(c))
        name_at = max(x for x in range(b) if c[x].kind == 'id')
        if n == 0:
            specs = ' '.join(t.text for t in c[:name_at] if t.text not in LOCAL)
        elif name_at != 0:
            raise SyntaxError('line %d: unsupported __local declaration' % stmt[0].line)
        decls.append((specs, c[name_at].text, text_of(src, c[name_at + 1:])))
    return decls


def top_level(toks):
    depth = 0
    for t in toks:
        if t.text in '([{':
            depth += 1
        elif t.text in ')]}':
            depth -= 1
        elif depth == 0:
            yield t


def rewrite(src):
    """Returns (new source, {kernel: [local variable names]})."""
    toks = tokenize(src)
    kernels = [k for k in find_kernels(src, toks) if k.locals]
    edits = []	# (start offset, end offset, replacement)
    info = {}
    for k in kernels:
        struct = '__vita_locals_' + k.name
        members = []
        names = []
        for (a, b), decls in k.locals:
            for specs, name, dims in decls:
                members.append('%s %s%s;' % (specs, name, dims))
                names.append(name)
            # blank the declaration, keeping its newlines, and define the names
            first, last = toks[a], toks[b]
            span = src[first.pos:last.pos + 1]
            defines = ''.join('#define %s (%s->%s)\n' % (name, HIDDEN_ARG, name) for _, name, _ in decls)
            edits.append((first.pos, last.pos + 1,
                          re.sub(r'[^\n]', ' ', span) + '\n' + defines + '#line %d\n' % last.line))
        info[k.name] = names
        start = toks[k.start]
        edits.append((start.pos, start.pos, 'struct %s { %s }; ' % (struct, ' '.join(members))))
        po, pc = k.params
        param = '__local struct %s *%s' % (struct, HIDDEN_ARG)
        inner = toks[po + 1:pc]
        if not inner or (len(inner) == 1 and inner[0].text == 'void'):
            edits.append((toks[po].pos + 1, toks[pc].pos, param))
        else:
            edits.append((toks[pc].pos, toks[pc].pos, ', ' + param))
        close = toks[k.body[1]]
        edits.append((close.pos + 1, close.pos + 1,
                      '\n' + ''.join('#undef %s\n' % n for n in names) + '#line %d\n' % close.line))
    out = []
    pos = 0
    for a, b, text in sorted(edits, key=lambda e: (e[0], e[1])):
        out.append(src[pos:a])
        out.append(text)
        pos = b
    out.append(src[pos:])
    return ''.join(out), info


def main():
    src = open(sys.argv[1]).read()
    new, info = rewrite(src)
    sys.stdout.write(new)
    for k, names in info.items():
        sys.stderr.write('%s: %s\n' % (k, ', '.join(names)))


if __name__ == '__main__':
    main()
