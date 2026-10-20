"""Doxygen input filter for Markdown pages (FILTER_PATTERNS in the Doxyfile).

The README is written for GitHub, so two things in it do not survive Doxygen as written:

- ```mermaid blocks: Doxygen has no Mermaid support, so they become raw HTML that mermaid.js
  (loaded from jsDelivr, once per page) draws in the browser.
- links to repository files (other .md pages, folders, images, videos) that are not part of the
  generated site: they become absolute GitHub URLs, images through raw.githubusercontent.com.

Code blocks and inline code are left untouched.

    python3 docs/doxygen/md_filter.py https://github.com/OWNER/REPO BRANCH FILE.md
"""
import html
import os
import re
import sys

MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs"
LINK = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)")


def main(repo, branch, path):
    owner_repo = repo.rstrip("/").split("github.com/", 1)[1]
    raw = f"https://raw.githubusercontent.com/{owner_repo}/{branch}"
    here = os.path.relpath(os.path.dirname(os.path.abspath(path)), os.getcwd())

    def absolute(m):
        bang, text, url = m.groups()
        if re.match(r"^([a-z]+:|#|/)", url):
            return m.group(0)
        target, _, anchor = url.partition("#")
        rel = os.path.normpath(os.path.join(here, target)).replace(os.sep, "/")
        if bang:
            new = f"{raw}/{rel}"
        else:
            kind = "tree" if target.endswith("/") or os.path.isdir(rel) else "blob"
            new = f"{repo}/{kind}/{branch}/{rel}" + (f"#{anchor}" if anchor else "")
        return f"{bang}[{text}]({new})"

    out, fence, block, mermaid = [], None, [], False
    for line in open(path, encoding="utf-8").read().split("\n"):
        stripped = line.strip()
        if fence is None and stripped.startswith("```"):
            fence = stripped[3:].strip().lower() or "text"
            block = []
            if fence != "mermaid":
                out.append(line)
            continue
        if fence is not None:
            if stripped.startswith("```"):
                if fence == "mermaid":
                    mermaid = True
                    out += ["\\htmlonly[block]", '<div class="mermaid">',
                            html.escape("\n".join(block)), "</div>", "\\endhtmlonly"]
                else:
                    out.append(line)
                fence = None
            elif fence == "mermaid":
                block.append(line)
            else:
                out.append(line)
            continue
        parts = re.split(r"(`[^`]*`)", line)
        out.append("".join(p if i % 2 else LINK.sub(absolute, p) for i, p in enumerate(parts)))
    if mermaid:
        out += ["", "\\htmlonly[block]", '<script type="module">',
                f'import mermaid from "{MERMAID}";',
                'const dark = document.documentElement.classList.contains("dark-mode") ||',
                '  matchMedia("(prefers-color-scheme: dark)").matches;',
                'mermaid.initialize({ startOnLoad: true, theme: dark ? "dark" : "neutral" });',
                "</script>", "\\endhtmlonly"]
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main(*sys.argv[1:4])
