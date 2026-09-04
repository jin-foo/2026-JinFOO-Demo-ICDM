# Leafmates: Together by Isolation

Demo paper for the [IEEE ICDM 2026 Demo Track](https://icdm2026.neu.edu.cn/CallforDemos/list.htm).

**Authors:** Jin Foo, Amin Beheshti, and Xuyun Zhang (School of Computing, Macquarie University)

Isolation-kernel retrieval as an inspectable neighbourhood: a namespaced leaf set, path predicates, LSH candidates, and the exact neighbours banding missed, on public simulated spend tables ([Sparkov](https://github.com/namebrandon/Sparkov_Data_Generation), [TabFormer](https://github.com/IBM/TabFormer)).

**Status:** submitted. This repository is public for ICDM 2026 reviewers. It is the author LaTeX plus the packed booth explorer. It is not an IEEE-published version.

Clone the repo, compile `main.tex`, or download [`demo/leafmates.html`](demo/leafmates.html) and open it in a browser (GitHub will not run the file).

## Compile

```bash
latexmk -pdf main.tex
```

Root file: `main.tex`. IEEE 2-column conference format, **4 pages including references**, single-blind.

## Layout

```
main.tex            # paper
references.bib
figures/            # Fig. 1 TikZ path, Fig. 2 schematic (+ generator), UI snapshot
demo/leafmates.html # packed booth explorer (open locally; GitHub will not run it)
```

Packed explorer: [`demo/leafmates.html`](https://github.com/jin-foo/2026-JinFOO-Demo-ICDM/blob/main/demo/leafmates.html). Download and open in a browser.

## Licence

Copyright © 2026 the authors. Submitted to IEEE ICDM 2026. All rights reserved pending IEEE copyright transfer if the paper is accepted.
