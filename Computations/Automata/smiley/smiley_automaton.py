#!/usr/bin/env python3
"""
Smiley-face self-assembly via a Mealy automaton (finite-state transducer).

- The target 2D smiley image is compiled into a Mealy machine over a one-letter
  input alphabet. Each input "tick" outputs one tile code and advances the
  automaton's state to the next raster position (left-to-right, top-to-bottom).
- Feed exactly W*H ticks to assemble the full image.

Usage
-----
python smiley_automaton.py --width 128 --height 128 --out smiley.png

Tile codes (output alphabet):
  0 = background, 1 = face, 2 = eye, 3 = mouth
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Set, List, Iterable, Any, Optional
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Mealy transducer ----------

@dataclass(frozen=True)
class FST:
    states: Set[Any]
    in_alphabet: List[Any]
    out_alphabet: List[Any]
    transition: Dict[Tuple[Any, Any], Tuple[Any, Any]]  # (state, input) -> (next_state, output)
    start: Any

    def run(self, inp: Iterable[Any], start_state: Optional[Any] = None) -> Tuple[List[Any], Any]:
        st = self.start if start_state is None else start_state
        out: List[Any] = []
        for x in inp:
            st, y = self.transition[(st, x)]
            out.append(y)
        return out, st

# ---------- Smiley geometry ----------

def build_smiley_grid(W: int, H: int) -> np.ndarray:
    """
    Returns an integer grid with codes:
      0 = background, 1 = face, 2 = eye, 3 = mouth
    """
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    r_face = 0.45 * min(W, H)

    # Face disk
    face = (xx - cx) ** 2 + (yy - cy) ** 2 <= r_face ** 2

    # Eyes (two disks)
    eye_r = 0.08 * r_face
    lx = cx - 0.4 * r_face
    rx = cx + 0.4 * r_face
    ey = cy - 0.25 * r_face
    left_eye  = (xx - lx) ** 2 + (yy - ey) ** 2 <= eye_r ** 2
    right_eye = (xx - rx) ** 2 + (yy - ey) ** 2 <= eye_r ** 2

    # Mouth: thick circular arc below center
    mouth_cx = cx
    mouth_cy = cy + 0.18 * r_face
    mouth_r  = 0.62 * r_face
    thickness = 0.05 * r_face
    dist = np.sqrt((xx - mouth_cx) ** 2 + (yy - mouth_cy) ** 2)
    ring = (np.abs(dist - mouth_r) <= thickness)
    lower_half = yy >= cy
    mouth = ring & lower_half

    # Compose layers (painter's order: background -> face -> mouth -> eyes)
    grid = np.zeros((H, W), dtype=int)
    grid[face] = 1
    grid[mouth] = 3
    grid[left_eye | right_eye] = 2
    return grid

# ---------- Compile grid into Mealy ----------

def compile_grid_to_mealy(grid: np.ndarray) -> FST:
    H, W = grid.shape
    states: Set[Any] = set()
    T: Dict[Tuple[Any, Any], Tuple[Any, Any]] = {}

    def sid(r: int, c: int) -> Tuple[int, int]:
        return (r, c)

    alphabet = ['•']  # one-letter input alphabet

    # Create states for each cell + HALT
    for r in range(H):
        for c in range(W):
            states.add(sid(r, c))
    states.add('HALT')

    # Transitions: on '•', output tile code at (r,c), then advance in raster order
    for r in range(H):
        for c in range(W):
            s = sid(r, c)
            out = int(grid[r, c])
            if c + 1 < W:
                ns = sid(r, c + 1)
            elif r + 1 < H:
                ns = sid(r + 1, 0)
            else:
                ns = 'HALT'
            T[(s, '•')] = (ns, out)

    # HALT self-loop (ignored if you feed exactly W*H ticks)
    T[('HALT', '•')] = ('HALT', 0)

    return FST(states=states, in_alphabet=alphabet, out_alphabet=[0,1,2,3], transition=T, start=(0,0))

# ---------- Run and render ----------

def assemble_and_save(fst: FST, W: int, H: int, out_path: Path) -> None:
    ticks = ['•'] * (W * H)
    out, _ = fst.run(ticks)
    arr = np.array(out, dtype=int).reshape((H, W))

    plt.figure(figsize=(6, 6))
    plt.imshow(arr, interpolation='nearest')
    plt.title("Smiley via Mealy Automaton (0=bg,1=face,2=eye,3=mouth)")
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1, dpi=240)
    plt.close()

# ---------- CLI ----------

def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate a smiley face via a Mealy automaton.")
    p.add_argument("--width",  type=int, default=96, help="canvas width in pixels")
    p.add_argument("--height", type=int, default=96, help="canvas height in pixels")
    p.add_argument("--out", type=str, default="smiley.png", help="output image filename")
    args = p.parse_args()

    W, H = args.width, args.height
    grid = build_smiley_grid(W, H)
    fst  = compile_grid_to_mealy(grid)
    out_path = Path(args.out)
    assemble_and_save(fst, W, H, out_path)
    print(f"Saved to {out_path.resolve()}")

if __name__ == "__main__":
    main()
