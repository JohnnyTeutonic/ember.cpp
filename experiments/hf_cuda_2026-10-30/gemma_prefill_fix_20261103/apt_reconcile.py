"""Make one APT source entry, ours, the only active entry for NVIDIA's CUDA repository (AMENDMENT.md, toolkit gate).

APT refuses two entries for the same repository with different Signed-By values, and an image may already configure
NVIDIA's repository in its own way. This script:
  1. sets aside every active entry whose URI is exactly the target repository (trailing slash ignored):
     one-line entries (sources.list, *.list) are commented out; in deb822 files (*.sources) only the matching
     stanza is disabled with "Enabled: no", or, when that stanza lists other URIs too, only the target URI is
     removed from it. Other entries, stanzas and files are left as they are. An http:// or longer-path variant is a
     different APT source and is not touched.
  2. keeps a copy of every file it changes in --backup;
  3. writes --entry, a one-line entry for the target signed by --keyring;
  4. checks that exactly one active entry for the target remains, and that it is --entry; exits 1 otherwise.
It prints a JSON record of what it changed. Running it again changes nothing further.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

MARK = '# set aside by vm_gemma.sh: '


def same(uri, target):
    return uri.rstrip('/') == target.rstrip('/')


def one_line_uri(line):
    """The URI of an active one-line entry, or None for comments, blanks and malformed lines."""
    s = line.strip()
    if not s or s.startswith('#'):
        return None
    tokens = s.split()
    if tokens[0] not in ('deb', 'deb-src'):
        return None
    rest = s[len(tokens[0]):].strip()
    if rest.startswith('['):  # options may contain spaces: [arch=amd64 signed-by=/x.gpg]
        end = rest.find(']')
        if end < 0:
            return None
        rest = rest[end + 1:].strip()
    parts = rest.split()
    return parts[0] if parts else None


def fix_one_line_file(text, target):
    out, changed = [], 0
    for line in text.splitlines(keepends=True):
        uri = one_line_uri(line)
        if uri is not None and same(uri, target):
            out.append(MARK + line.lstrip())
            changed += 1
        else:
            out.append(line)
    return ''.join(out), changed


def stanzas(text):
    """Split deb822 text into blocks of lines, keeping the blank-line separators as their own blocks."""
    blocks, cur = [], []
    for line in text.splitlines(keepends=True):
        if line.strip() == '':
            if cur:
                blocks.append(cur)
                cur = []
            blocks.append([line])
        else:
            cur.append(line)
    if cur:
        blocks.append(cur)
    return blocks


def fields(block):
    """Field name -> (index of its first line, value) for a deb822 stanza; continuation lines are joined."""
    found, name = {}, None
    for i, line in enumerate(block):
        if line.startswith('#'):
            continue
        if line[:1] in (' ', '\t') and name:
            idx, val = found[name]
            found[name] = (idx, val + ' ' + line.strip())
        elif ':' in line:
            name, val = line.split(':', 1)
            name = name.strip().lower()
            found[name] = (i, val.strip())
    return found


def fix_deb822_file(text, target):
    out, changed = [], 0
    for block in stanzas(text):
        f = fields(block)
        if 'uris' not in f or f.get('enabled', (0, 'yes'))[1].lower() == 'no':
            out.extend(block)
            continue
        uris = f['uris'][1].split()
        if not any(same(u, target) for u in uris):
            out.extend(block)
            continue
        changed += 1
        keep = [u for u in uris if not same(u, target)]
        idx = f['uris'][0]
        # drop the URIs field's continuation lines; rewrite it on one line
        end = idx + 1
        while end < len(block) and block[end][:1] in (' ', '\t'):
            end += 1
        if keep:
            block = block[:idx] + ['URIs: ' + ' '.join(keep) + '\n'] + block[end:]
        else:
            if 'enabled' in f:
                block = [l for j, l in enumerate(block) if j != f['enabled'][0]]
            if block and not block[-1].endswith('\n'):
                block[-1] += '\n'
            block = block + ['Enabled: no\n']
        out.extend(block)
    return ''.join(out), changed


def active_entries(apt_etc, target):
    found = []
    candidates = [apt_etc / 'sources.list'] + sorted((apt_etc / 'sources.list.d').glob('*.list'))
    for p in candidates:
        if p.is_file():
            for line in p.read_text().splitlines():
                uri = one_line_uri(line)
                if uri is not None and same(uri, target):
                    found.append(str(p))
    for p in sorted((apt_etc / 'sources.list.d').glob('*.sources')):
        for block in stanzas(p.read_text()):
            f = fields(block)
            if 'uris' in f and f.get('enabled', (0, 'yes'))[1].lower() != 'no' \
                    and any(same(u, target) for u in f['uris'][1].split()):
                found.append(str(p))
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apt-etc', required=True, type=Path)
    ap.add_argument('--uri', required=True)
    ap.add_argument('--keyring', required=True)
    ap.add_argument('--entry', required=True, type=Path)
    ap.add_argument('--backup', required=True, type=Path)
    a = ap.parse_args(argv)
    a.backup.mkdir(parents=True, exist_ok=True)
    record = {'target': a.uri, 'changed': {}}
    files = [a.apt_etc / 'sources.list'] + sorted((a.apt_etc / 'sources.list.d').glob('*.list')) \
        + sorted((a.apt_etc / 'sources.list.d').glob('*.sources'))
    for p in files:
        if not p.is_file() or p == a.entry:
            continue
        text = p.read_text()
        new, n = (fix_deb822_file if p.suffix == '.sources' else fix_one_line_file)(text, a.uri)
        if n:
            shutil.copy2(p, a.backup / p.name)
            p.write_text(new)
            record['changed'][str(p)] = n
    a.entry.write_text(f'deb [signed-by={a.keyring}] {a.uri.rstrip("/")}/ /\n')
    record['entry'] = str(a.entry)
    record['active_after'] = active_entries(a.apt_etc, a.uri)
    print(json.dumps(record, indent=2))
    if record['active_after'] != [str(a.entry)]:
        print('expected exactly one active entry for the target, ours', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
