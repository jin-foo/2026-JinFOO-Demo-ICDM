# Leafmates: Together by Isolation

Demo paper for the [IEEE ICDM 2026 Demo Track](https://icdm2026.neu.edu.cn/CallforDemos/list.htm).

**Authors:** Jin Foo, Amin Beheshti, and Xuyun Zhang (School of Computing, Macquarie University)

Isolation-kernel retrieval as an inspectable neighbourhood: a namespaced leaf set, path predicates, LSH candidates, and the exact neighbours banding missed, on public simulated spend tables ([Sparkov](https://github.com/namebrandon/Sparkov_Data_Generation), [TabFormer](https://github.com/IBM/TabFormer)).

**Status:** submitted. This repository is the author LaTeX. It is not an IEEE-published version.

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
```

The packed booth explorer is a separate, network-free HTML file. Open it locally; GitHub will not execute it. It is not stored in this repository.

## Licence

Copyright © 2026 the authors. Submitted to IEEE ICDM 2026. All rights reserved pending IEEE copyright transfer if the paper is accepted.
