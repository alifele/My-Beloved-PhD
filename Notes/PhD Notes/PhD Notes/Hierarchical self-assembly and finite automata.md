
The idea of application of automata theory: Every finite automata, partitions the set of all possible inputs into the strings that it accepts and the ones that it does not accept. With this concept, one can design automata that can detect certain patterns in the string. Like the automata that only accepts the string abc or the one that only accepts strings with even number of a-s or any other pattern. Even one can design automata that can detect hierarchical patterns with having sub-automata diagrams that is connected by larger automata. This similarity sparks the potential of using automata in hierarchical self assembly in the sense that encode the interactions between tiles into a code and then the desired self-assembly will be corresponding to an automata that accepts only that string


Update: I was able to prototype my idea in ChatGPT and design some basic automata that can generate some patterns, like Serpinksly trangle from Winfree's PhD thesis, etc

```python
# Automata Pattern Suite — full, runnable code
# ------------------------------------------------------------
# This script demonstrates several "automata-to-pattern" constructions:
#  1) Checkerboard via DFA (row) + Mealy transducer (vertical)
#  2) 2D Thue–Morse via DFAO (deterministic automaton with output)
#  3) Sierpiński triangle via 1D cellular automaton (Rule 90)
#  4) de Bruijn row via Eulerian walk on a de Bruijn DFA, then vertical shifts
#
# All images are saved in /mnt/data and previewed inline.
#
# Notes
#  - Uses matplotlib only (no seaborn), one figure per output, default colors.
#  - Everything is deterministic given the RNG seed.
#
# ------------------------------------------------------------

from dataclasses import dataclass
from typing import Dict, Tuple, Set, List, Iterable, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# ---------- Generic DFA (acceptor) ----------

@dataclass(frozen=True)
class DFA:
    states: Set[Any]
    alphabet: List[Any]
    transition: Dict[Tuple[Any, Any], Any]
    start: Any
    accept: Set[Any]

    def step(self, state: Any, symbol: Any) -> Any:
        return self.transition[(state, symbol)]

    def accepts(self, s: Iterable[Any]) -> bool:
        st = self.start
        for ch in s:
            st = self.step(st, ch)
        return st in self.accept

def dfa_generate_fixed_length(dfa: DFA, L: int, rng: random.Random) -> List[Any]:
    """
    Generate a length-L string in dfa.alphabet that is guaranteed to be accepted.
    Uses backward dynamic programming to enforce feasibility at each step.
    """
    # can_finish[k][state] = can we read k symbols from 'state' and end in accept?
    can_finish = [{s: False for s in dfa.states} for _ in range(L + 1)]
    for s in dfa.states:
        can_finish[0][s] = s in dfa.accept

    for k in range(1, L + 1):
        for s in dfa.states:
            can_finish[k][s] = any(can_finish[k - 1][dfa.step(s, a)] for a in dfa.alphabet)

    if not can_finish[L][dfa.start]:
        raise ValueError("Requested length has no accepted string from start state for this DFA.")

    word: List[Any] = []
    st = dfa.start
    remaining = L
    while remaining > 0:
        choices = [a for a in dfa.alphabet if can_finish[remaining - 1][dfa.step(st, a)]]
        a = rng.choice(choices)
        word.append(a)
        st = dfa.step(st, a)
        remaining -= 1
    assert st in dfa.accept
    return word

# ---------- DFAO (automaton with output at end state) ----------

@dataclass(frozen=True)
class DFAO:
    states: Set[Any]
    alphabet: List[Any]
    transition: Dict[Tuple[Any, Any], Any]
    start: Any
    output: Dict[Any, Any]  # output per final state

    def run(self, s: Iterable[Any]) -> Any:
        st = self.start
        for ch in s:
            st = self.transition[(st, ch)]
        return self.output[st]

# ---------- Mealy transducer (per-symbol output) ----------

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

# ---------- Cellular Automaton (1D, radius=1) ----------

class CA1D:
    """
    Simple 1D binary cellular automaton with radius 1 (3-neighborhood rule).
    Rule is a mapping from (left,center,right) in {0,1}^3 to {0,1}.
    """
    def __init__(self, rule_table: Dict[Tuple[int, int, int], int]):
        self.rule = rule_table

    @classmethod
    def from_wolfram(cls, rule_number: int):
        """
        Constructs a radius-1 CA using Wolfram numbering (0..255).
        Neighborhoods in order: 111,110,101,100,011,010,001,000.
        """
        bits = [(rule_number >> i) & 1 for i in range(8)]
        neighborhoods = [
            (1,1,1), (1,1,0), (1,0,1), (1,0,0),
            (0,1,1), (0,1,0), (0,0,1), (0,0,0)
        ]
        rule = {nh: bits[7 - idx] for idx, nh in enumerate(neighborhoods)}
        return cls(rule)

    def step(self, row: np.ndarray) -> np.ndarray:
        n = len(row)
        left  = np.roll(row,  1)
        right = np.roll(row, -1)
        out = np.zeros_like(row)
        # Build key indexes 0..7 for each neighborhood, then map via rule
        idx = (left << 2) | (row << 1) | right
        # Precompute lookup
        lut = np.zeros(8, dtype=int)
        for k, nh in enumerate([(i>>2 &1, i>>1 &1, i&1) for i in range(8)]):
            lut[k] = self.rule[nh]
        out[:] = lut[idx]
        return out

# ---------- Useful specific automata ----------

def dfa_alternating_01() -> DFA:
    """Two-state DFA that generates strings of alternating 0 and 1 (even length acceptance)."""
    states = {'s0', 's1'}
    alphabet = [0, 1]
    T = {
        ('s0', 0): 's1', ('s0', 1): 'sX',
        ('s1', 1): 's0', ('s1', 0): 'sX',
    }
    # Complete with sink
    states.add('sX')
    for st in list(states):
        for a in alphabet:
            if (st, a) not in T:
                T[(st, a)] = 'sX'
    return DFA(states, alphabet, T, 's0', {'s0'})  # accept even-length alternating strings

def fst_row_flip() -> FST:
    """Mealy machine that flips every symbol (0->1, 1->0)."""
    states = {'q'}
    in_alphabet = [0, 1]
    out_alphabet = [0, 1]
    T = {('q', 0): ('q', 1), ('q', 1): ('q', 0)}
    return FST(states, in_alphabet, out_alphabet, T, 'q')

def dfao_thue_morse() -> DFAO:
    """
    Two-state DFAO that outputs parity of 1-bits of the binary input.
    Reading MSB->LSB of n gives t(n) (Thue–Morse).
    """
    states = {'E', 'O'}  # even/odd parity
    alphabet = [0, 1]
    T = {
        ('E', 0): 'E', ('E', 1): 'O',
        ('O', 0): 'O', ('O', 1): 'E',
    }
    start = 'E'
    output = {'E': 0, 'O': 1}
    return DFAO(set(states), alphabet, T, start, output)

def de_bruijn_binary(k: int, n: int) -> List[int]:
    """
    Generate a binary de Bruijn sequence B(k=2, n) using a standard recursive algorithm.
    Returns a list of 0/1 of length 2**n.
    """
    # This implementation is specialized to k=2 for simplicity
    a = [0] * (2 * n)
    seq = []

    def db(t: int, p: int):
        if t > n:
            if n % p == 0:
                seq.extend(a[1:p+1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, 2):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return seq

# ---------- Pattern builders ----------

def pattern_checkerboard(W: int, H: int, rng: random.Random) -> np.ndarray:
    """
    Build a checkerboard: a DFA emits alternating 0,1 row; an FST flips every other row.
    """
    row_dfa = dfa_alternating_01()
    base_row = dfa_generate_fixed_length(row_dfa, W, rng)
    flip = fst_row_flip()

    grid = np.zeros((H, W), dtype=int)
    grid[0, :] = base_row
    st = None
    for r in range(1, H):
        # flip previous row to alternate
        out, st = flip.run(grid[r-1, :], st)
        grid[r, :] = out
    return grid

def pattern_thue_morse_2d(W: int, H: int) -> np.ndarray:
    """
    2D Thue–Morse: value(i,j) = tm(i) XOR tm(j), with tm computed by DFAO.
    """
    tm = dfao_thue_morse()

    def tm_value(n: int) -> int:
        if n == 0:
            return tm.run([0])  # read a single 0
        bits = list(map(int, bin(n)[2:]))
        return tm.run(bits)

    row = np.array([tm_value(j) for j in range(W)], dtype=int)
    col = np.array([tm_value(i) for i in range(H)], dtype=int)[:, None]
    grid = (row ^ col).astype(int)
    return grid

def pattern_sierpinski_rule90(W: int, H: int) -> np.ndarray:
    """
    Sierpiński triangle from Rule 90 CA, seeded with a single 1 at center.
    """
    ca = CA1D.from_wolfram(90)
    row = np.zeros(W, dtype=int)
    row[W // 2] = 1
    grid = np.zeros((H, W), dtype=int)
    grid[0, :] = row
    for r in range(1, H):
        row = ca.step(row)
        grid[r, :] = row
    return grid

def pattern_debruijn_shift(W: int, H: int, n: int) -> np.ndarray:
    """
    Use a binary de Bruijn sequence of order n as the first row;
    subsequent rows are cyclic shifts by +1 to create diagonal "covering" of all n-grams.
    """
    seq = de_bruijn_binary(2, n)
    # ensure row length W by repeating
    s = (seq * ((W // len(seq)) + 2))[:W]
    grid = np.zeros((H, W), dtype=int)
    row = np.array(s, dtype=int)
    for r in range(H):
        grid[r, :] = np.roll(row, r)
    return grid

# ---------- Render helpers ----------

def save_grid(grid: np.ndarray, title: str, fname: str):
    out = Path('/mnt/data') / fname
    plt.figure(figsize=(8, 3))
    plt.imshow(grid, aspect='auto', interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.savefig(out, bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close()
    return str(out)

# ---------- Generate all patterns ----------

W, H = 256, 128
rng = random.Random(7)

g1 = pattern_checkerboard(W, H, rng)
p1 = save_grid(g1, "Checkerboard (DFA row + Mealy vertical flip)", "automata_checkerboard.png")

g2 = pattern_thue_morse_2d(W, H)
p2 = save_grid(g2, "2D Thue–Morse (DFAO parity XOR)", "automata_thue_morse_2d.png")

g3 = pattern_sierpinski_rule90(W, H)
p3 = save_grid(g3, "Sierpiński via Rule 90 CA", "automata_sierpinski_rule90.png")

g4 = pattern_debruijn_shift(W, H, n=8)
p4 = save_grid(g4, "de Bruijn row (order 8) with vertical cyclic shifts", "automata_debruijn_shift.png")

# Preview inline
for path in [p1, p2, p3, p4]:
    img = plt.imread(path)
    plt.figure(figsize=(8,3))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

[p1, p2, p3, p4]

```


This can generate the patterns like:

![[Pasted image 20250824073110.png]]

![[Pasted image 20250824073118.png]]

![[Pasted image 20250824073136.png]]

![[Pasted image 20250824073146.png]]












## From Some of the older runs
![[Pasted image 20250824073240.png]]

```python
# Recreate and save Patterns 1 and 3 to accompany Pattern 2

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

rng = random.Random(42)
W, H = 120, 60

# Pattern 1: DFA rows (even number of 'a')
grid1 = make_grid_via_row_dfa(make_even_a_dfa(), W, H, rng)
p1 = show_and_save(grid1, "Pattern 1 — DFA rows (even number of 'a')", "automata_pattern_1.png")

# Pattern 3: Vertical constraint via transducer
row_for_fst = [rng.randrange(0,3) for _ in range(W)]
fst = make_vertical_transducer()
grid3 = make_grid_via_transducer(row_for_fst, H, fst)
p3 = show_and_save(grid3, "Pattern 3 — Vertical constraint via transducer", "automata_pattern_3.png")

# Inline previews
for path in [p1, p3]:
    img = plt.imread(path)
    plt.figure(figsize=(8,3))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

[p1, p3]

```

```python
# Automata-driven pattern generator
# - DFA-based row patterns
# - Hierarchical (macro DFA + sub-DFAs) expansion
# - Finite-state transducer to enforce vertical constraints between rows
#
# This script generates three images and saves them to /mnt/data

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Set, Tuple, List, Iterable, Any, Optional
import random
from pathlib import Path

# ---------- Core DFA ----------

@dataclass(frozen=True)
class DFA:
    states: Set[Any]
    alphabet: List[str]
    transition: Dict[Tuple[Any, str], Any]
    start: Any
    accept: Set[Any]

    def step(self, state: Any, symbol: str) -> Any:
        return self.transition[(state, symbol)]

    def accepts(self, s: Iterable[str]) -> bool:
        st = self.start
        for ch in s:
            st = self.step(st, ch)
        return st in self.accept

def generate_string_of_length(dfa: DFA, L: int, rng: random.Random) -> List[str]:
    """
    Generate a length-L string over dfa.alphabet that is accepted by the DFA.
    Uses backward DP to ensure completion is possible from each prefix.
    """
    # Backward DP: can_finish[k][state] = True if from state we can read k symbols and end in accept
    can_finish = [ {s: False for s in dfa.states} for _ in range(L+1) ]
    for s in dfa.states:
        can_finish[0][s] = (s in dfa.accept)
    for k in range(1, L+1):
        for s in dfa.states:
            # there exists a symbol a s.t. transition leads to state s' with can_finish[k-1][s'] True
            can_finish[k][s] = any( can_finish[k-1][ dfa.step(s, a) ] for a in dfa.alphabet )

    if not any(can_finish[L][s] for s in dfa.states if s == dfa.start):
        raise ValueError("No string of requested length is accepted by the DFA from the start state.")

    out = []
    state = dfa.start
    remaining = L
    while remaining > 0:
        # choose among symbols that keep completion possible
        valid = [a for a in dfa.alphabet if can_finish[remaining-1][ dfa.step(state, a) ]]
        if not valid:
            # should not happen if DP is correct
            raise RuntimeError("Dead end while generating string")
        sym = rng.choice(valid)
        out.append(sym)
        state = dfa.step(state, sym)
        remaining -= 1
    # ensure acceptance
    assert state in dfa.accept
    return out

# ---------- Finite-State Transducer (Mealy) ----------

@dataclass(frozen=True)
class FST:
    states: Set[Any]
    in_alphabet: List[int]
    out_alphabet: List[int]
    transition: Dict[Tuple[Any, int], Tuple[Any, int]]  # (state, inp) -> (next_state, out)
    start: Any

    def run(self, inp: Iterable[int], start_state: Optional[Any] = None) -> Tuple[List[int], Any]:
        st = self.start if start_state is None else start_state
        out = []
        for x in inp:
            st, y = self.transition[(st, x)]
            out.append(y)
        return out, st

# ---------- Example DFAs ----------

def make_even_a_dfa() -> DFA:
    # Alphabet {a,b}; states q0 (even a), q1 (odd a)
    states = {'q0','q1'}
    alphabet = ['a','b']
    trans = {}
    trans[('q0','a')] = 'q1'
    trans[('q0','b')] = 'q0'
    trans[('q1','a')] = 'q0'
    trans[('q1','b')] = 'q1'
    return DFA(states, alphabet, trans, 'q0', {'q0'})

def make_no_bb_even_len_dfa() -> DFA:
    # Alphabet {a,b}; language: no consecutive 'b' and even length
    # States track (parity, last_was_b)
    states = {('even', False), ('even', True), ('odd', False), ('odd', True)}
    alphabet = ['a','b']
    trans = {}
    for parity in ['even','odd']:
        for lastb in [False, True]:
            s = (parity, lastb)
            # on 'a': parity flips, lastb=False
            np = 'odd' if parity=='even' else 'even'
            trans[(s,'a')] = (np, False)
            # on 'b': only allowed if lastb==False; parity flips; lastb=True
            if not lastb:
                trans[(s,'b')] = (np, True)
            else:
                # To keep DFA total, if invalid, transition to a sink state
                pass
    sink = ('sink', False)
    # complete transitions
    states.add(sink)
    for a in ['a','b']:
        trans[(sink,a)] = sink
    for parity in ['even','odd']:
        for lastb in [False, True]:
            s = (parity,lastb)
            if (s,'b') not in trans:
                trans[(s,'b')] = sink
    # accepts: even length regardless of lastb
    accept = {('even', False), ('even', True)}
    return DFA(states, alphabet, trans, ('even', False), accept)

def make_exact_abc_dfa() -> DFA:
    # Accept exactly "abc"
    states = {'s','sa','sab','sink'}
    alphabet = ['a','b','c']
    T = {}
    # from start 's'
    T[('s','a')] = 'sa'
    T[('s','b')] = 'sink'
    T[('s','c')] = 'sink'
    # from 'sa'
    T[('sa','a')] = 'sink'
    T[('sa','b')] = 'sab'
    T[('sa','c')] = 'sink'
    # from 'sab'
    T[('sab','a')] = 'sink'
    T[('sab','b')] = 'sink'
    T[('sab','c')] = 'ACC'  # we'll treat 'ACC' as a state
    # sink + ACC
    states.add('ACC')
    for st in list(states):
        for ch in alphabet:
            if (st,ch) not in T:
                T[(st,ch)] = 'sink'
    return DFA(states, alphabet, T, 's', {'ACC'})

# ---------- Macro (hierarchical) assembly ----------

@dataclass
class MacroSpec:
    name: str
    dfa: DFA
    length_range: Tuple[int,int]  # inclusive

@dataclass
class MacroDFA:
    dfa: DFA               # DFA over macro-symbol alphabet (e.g., ['A','B'])
    macros: Dict[str, MacroSpec]  # map macro symbol -> spec

def generate_macro_string(mdfa: MacroDFA, macro_count: int, rng: random.Random) -> List[str]:
    return generate_string_of_length(mdfa.dfa, macro_count, rng)

def expand_macros(macro_word: List[str], mdfa: MacroDFA, rng: random.Random) -> List[str]:
    out: List[str] = []
    for M in macro_word:
        spec = mdfa.macros[M]
        L = rng.randint(spec.length_range[0], spec.length_range[1])
        out.extend(generate_string_of_length(spec.dfa, L, rng))
    return out

# ---------- Build Example Macro System ----------

def make_macro_system() -> MacroDFA:
    # Macro alphabet: A,B
    # Macro DFA: accepts (AB)^* A  (alternation ending with A)
    states = {'S','sA','sB'}
    alphabet = ['A','B']
    T = {
        ('S','A'):'sA', ('S','B'):'sB',
        ('sA','A'):'sA', ('sA','B'):'sB',
        ('sB','A'):'sA', ('sB','B'):'sB',
    }
    macro_accept = {'sA'}
    macro_dfa = DFA(states, alphabet, T, 'S', macro_accept)

    # Sub-DFAs:
    #  A: no 'bb' and even length
    A = MacroSpec('A', make_no_bb_even_len_dfa(), (6, 14))
    #  B: even number of 'a'
    B = MacroSpec('B', make_even_a_dfa(), (5, 12))

    return MacroDFA(macro_dfa, {'A':A,'B':B})

# ---------- Transducer Example ----------

def make_vertical_transducer() -> FST:
    # Inputs/outputs in {0,1,2}; two states toggle some rule to create diagonals/chevrons
    states = {'q0','q1'}
    in_alphabet = [0,1,2]
    out_alphabet = [0,1,2]
    T = {}
    # Rule: out = (inp + (0 if state==q0 else 1)) % 3
    # next state toggles when inp==2
    for st in states:
        for x in in_alphabet:
            add = 0 if st=='q0' else 1
            y = (x + add) % 3
            ns = 'q1' if (x==2 and st=='q0') else ('q0' if (x==2 and st=='q1') else st)
            T[(st,x)] = (ns, y)
    return FST(set(states), in_alphabet, out_alphabet, T, 'q0')

# ---------- Pattern assembly ----------

def make_grid_via_row_dfa(dfa: DFA, width: int, height: int, rng: random.Random) -> np.ndarray:
    # map alphabet symbols to integers for visualization
    sym_to_int = {s:i for i,s in enumerate(dfa.alphabet)}
    grid = np.zeros((height, width), dtype=int)
    for r in range(height):
        row = generate_string_of_length(dfa, width, rng)
        grid[r,:] = [sym_to_int[x] for x in row]
    return grid

def make_grid_via_macro(mdfa: MacroDFA, width: int, height: int, rng: random.Random) -> np.ndarray:
    sym_to_int = {'a':0, 'b':1}  # both sub-DFAs use these letters
    grid = np.zeros((height, width), dtype=int)
    # make a long string and wrap lines
    macro_word = generate_macro_string(mdfa, max(2, width//10), rng)
    micro = expand_macros(macro_word, mdfa, rng)
    # ensure we have enough symbols
    need = width*height
    while len(micro) < need:
        macro_word = generate_macro_string(mdfa, max(2, width//10), rng)
        micro.extend(expand_macros(macro_word, mdfa, rng))
    arr = [sym_to_int[ch] for ch in micro[:need]]
    grid[:,:] = np.array(arr, dtype=int).reshape((height,width))
    return grid

def make_grid_via_transducer(base_row: List[int], rows: int, fst: FST) -> np.ndarray:
    W = len(base_row)
    grid = np.zeros((rows, W), dtype=int)
    grid[0,:] = base_row
    st = None
    for r in range(1, rows):
        out, st = fst.run(grid[r-1,:], st)
        grid[r,:] = out
    return grid

# ---------- Generate and save patterns ----------

rng = random.Random(42)

W, H = 120, 60

# (1) Pure DFA rows (even number of 'a')
grid1 = make_grid_via_row_dfa(make_even_a_dfa(), W, H, rng)

# (2) Hierarchical macro system
mdfa = make_macro_system()
grid2 = make_grid_via_macro(mdfa, W, H, rng)

# (3) Vertical constraint via FST
# Build a base row using DFA over alphabet {0,1,2} by mapping from a,b
row_for_fst = [rng.randrange(0,3) for _ in range(W)]
fst = make_vertical_transducer()
grid3 = make_grid_via_transducer(row_for_fst, H, fst)

# ---------- Visualize ----------

figsize = (8, 9)
figs = []

def show_and_save(grid: np.ndarray, title: str, filename: str):
    plt.figure(figsize=(8,3))
    plt.imshow(grid, aspect='auto', interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    outpath = Path('/mnt/data') / filename
    plt.savefig(outpath, bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close()
    return str(outpath)

p1 = show_and_save(grid1, "Pattern 1 — DFA rows (even number of 'a')", "automata_pattern_1.png")
p2 = show_and_save(grid2, "Pattern 2 — Hierarchical macros (macro DFA + sub-DFAs)", "automata_pattern_2.png")
p3 = show_and_save(grid3, "Pattern 3 — Vertical constraint via transducer", "automata_pattern_3.png")

# Display inline previews
for path in [p1, p2, p3]:
    img = plt.imread(path)
    plt.figure(figsize=(8,3))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

paths = [p1, p2, p3]
paths

```