# Chevron motif (matches the demo we discussed)
python fst_patterns.py --pattern chevrons_k3 --width 256 --height 128 --seed 7 --out chevrons.png

# Diagonal stripes (stateless +1 mod 4)
python fst_patterns.py --pattern stripes_k4 --width 256 --height 128 --out stripes.png

# Brick/staggered look (XOR with 2-phase state)
python fst_patterns.py --pattern brick_k2 --width 256 --height 128 --out brick.png

# Wave/interference (phase increments on zeros)
python fst_patterns.py --pattern wave_k5 --width 256 --height 128 --out wave.png

