#!/usr/bin/env python3
"""
FST-based 2D pattern generator (Pattern 3: vertical constraints via a Mealy transducer)

Each new row is produced from the row above by a Mealy machine (finite-state
transducer) that scans the row left-to-right. The machine has a finite state,
so the output at column j can depend on the entire prefix up to j-1 as well as
the current input symbol at j. The state at the end of a row is carried into
the next row, giving vertical memory with finite-state capacity.

Presets implemented:
  - chevrons_k3:    3-symbol alphabet, adds 0/1 mod 3 depending on a toggle state; toggle on symbol '2'
  - stripes_k4:     4-symbol alphabet, constant +1 mod 4 mapping (no state change): diagonal stripes
  - brick_k2:       2-symbol alphabet, output = x XOR phase, and 'phase' advances as a 2-cycle each column
                    producing a brick/staggered look when carried across rows
  - wave_k5:        5-symbol alphabet, state is a 5-cycle that increments every time we see input==0,
                    making quasi-waves and interference when base row is random.

Usage:
  python fst_patterns.py --pattern chevrons_k3 --width 256 --height 128 --seed 7 --out out.png

"""

from dataclasses import dataclass
from typing import Dict, Tuple, Set, List, Iterable, Any, Optional
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# ---------- Mealy transducer ----------

@dataclass(frozen=True)
class FST:
    states: Set[Any]
    in_alphabet: List[int]
    out_alphabet: List[int]
    transition: Dict[Tuple[Any, int], Tuple[Any, int]]  # (state, input) -> (next_state, output)
    start: Any

    def run(self, inp: Iterable[int], start_state: Optional[Any] = None) -> Tuple[List[int], Any]:
        st = self.start if start_state is None else start_state
        out: List[int] = []
        for x in inp:
            st, y = self.transition[(st, x)]
            out.append(y)
        return out, st

# ---------- Core grid builder ----------

def make_grid_via_transducer(base_row: List[int], rows: int, fst: FST) -> np.ndarray:
    W = len(base_row)
    grid = np.zeros((rows, W), dtype=int)
    grid[0, :] = base_row
    st = None  # carry state across rows
    for r in range(1, rows):
        out, st = fst.run(grid[r-1, :], st)
        grid[r, :] = out
    return grid

# ---------- Base row helpers ----------

def base_row_random(W: int, k: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    return [rng.randrange(0, k) for _ in range(W)]

def base_row_periodic(W: int, k: int, period: int = 8) -> List[int]:
    row = [(i % period) % k for i in range(W)]
    return row

# ---------- Preset FSTs ----------

def fst_chevrons_k3() -> FST:
    """
    3-symbol alphabet {0,1,2}. Output = (x + add) mod 3, where 'add' is 0 in state q0 and 1 in state q1.
    Toggle q0<->q1 whenever input==2. This produces chevrons/diagonal motifs.
    """
    states = {'q0', 'q1'}
    in_alphabet = [0, 1, 2]
    out_alphabet = [0, 1, 2]
    T = {}
    for st in states:
        add = 0 if st == 'q0' else 1
        for x in in_alphabet:
            y = (x + add) % 3
            if x == 2:
                ns = 'q1' if st == 'q0' else 'q0'
            else:
                ns = st
            T[(st, x)] = (ns, y)
    return FST(set(states), in_alphabet, out_alphabet, T, 'q0')

def fst_stripes_k4() -> FST:
    """
    4-symbol alphabet {0,1,2,3}. Stateless shift: y = (x + 1) mod 4, state remains 'q'.
    This produces clean diagonal stripes.
    """
    states = {'q'}
    in_alphabet = [0,1,2,3]
    out_alphabet = [0,1,2,3]
    T = {('q', x): ('q', (x+1) % 4) for x in in_alphabet}
    return FST(set(states), in_alphabet, out_alphabet, T, 'q')

def fst_brick_k2() -> FST:
    """
    2-symbol alphabet {0,1}. Output = x XOR phase. Phase alternates each column (q0->q1->q0).
    Carried state across rows yields a staggered/brick pattern.
    """
    states = {'q0', 'q1'}
    in_alphabet = [0,1]
    out_alphabet = [0,1]
    T = {}
    for st in states:
        phase = 0 if st == 'q0' else 1
        for x in in_alphabet:
            y = x ^ phase
            ns = 'q1' if st == 'q0' else 'q0'  # toggle every column
            T[(st, x)] = (ns, y)
    return FST(set(states), in_alphabet, out_alphabet, T, 'q0')

def fst_wave_k5() -> FST:
    """
    5-symbol alphabet {0,1,2,3,4}. States q0..q4 represent a running 'phase' p.
    If input==0, advance p=(p+1) mod 5; output y = (x + p) mod 5.
    """
    states = [f'q{i}' for i in range(5)]
    in_alphabet = [0,1,2,3,4]
    out_alphabet = [0,1,2,3,4]
    T = {}
    for i, st in enumerate(states):
        p = i
        for x in in_alphabet:
            y = (x + p) % 5
            if x == 0:
                ns = states[(i+1) % 5]
            else:
                ns = st
            T[(st, x)] = (ns, y)
    return FST(set(states), in_alphabet, out_alphabet, T, 'q0')

# ---------- Render ----------

def save_grid(grid: np.ndarray, title: str, fname: str) -> str:
    out = Path(fname)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.imshow(grid, aspect='auto', interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.savefig(out, bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close()
    return str(out)

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Pattern 3: FST-based vertical constraints")
    ap.add_argument("--pattern", type=str, default="chevrons_k3",
                    choices=["chevrons_k3", "stripes_k4", "brick_k2", "wave_k5"],
                    help="which preset Mealy transducer to use")
    ap.add_argument("--width", type=int, default=256, help="number of columns")
    ap.add_argument("--height", type=int, default=128, help="number of rows")
    ap.add_argument("--seed", type=int, default=7, help="seed for base row randomness")
    ap.add_argument("--base", type=str, default="random", choices=["random","periodic"],
                    help="how to generate the first row")
    ap.add_argument("--out", type=str, default="fst_pattern.png", help="output PNG filename")
    args = ap.parse_args()

    # Pick FST and alphabet size
    if args.pattern == "chevrons_k3":
        fst = fst_chevrons_k3(); k = 3
    elif args.pattern == "stripes_k4":
        fst = fst_stripes_k4(); k = 4
    elif args.pattern == "brick_k2":
        fst = fst_brick_k2(); k = 2
    elif args.pattern == "wave_k5":
        fst = fst_wave_k5(); k = 5
    else:
        raise ValueError("Unknown pattern")

    # Base row
    if args.base == "random":
        row0 = base_row_random(args.width, k, args.seed)
    else:
        row0 = base_row_periodic(args.width, k)

    grid = make_grid_via_transducer(row0, args.height, fst)
    title = f"Pattern: {args.pattern}  |  base={args.base}  |  k={k}"
    path = save_grid(grid, title, args.out)
    print(f"Saved: {Path(path).resolve()}")

if __name__ == "__main__":
    main()
