import random
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import copy
import yaml



class Tile:
    def __init__(self, north, east, south, west, name='T'):
        self.glues = {'N': north, 'E': east, 'S': south, 'W': west}
        self.name = name

    def get_glue(self, direction):
        return self.glues[direction]



class AssemblyWithGlueInteractions:
    def __init__(self, tile_set):
        self.asmbDict = defaultdict(lambda: None)
        self.boundaryTiles = set()
        self.tile_set = tile_set
        self.seed_pos_List = [(0, 0)]
        for seed_pos in self.seed_pos_List:
            self.asmbDict[seed_pos] = tile_set[0]
            self.update_boundary(seed_pos)

        self.history = []

    def neighbor_positions(self, x, y):
        return {
            'N': (x, y + 1),
            'E': (x + 1, y),
            'S': (x, y - 1),
            'W': (x - 1, y)
        }

    def update_boundary(self, pos):
        x, y = pos
        for direction, (nx, ny) in self.neighbor_positions(x, y).items():
            if self.asmbDict[(nx, ny)] is None:
                self.boundaryTiles.add((nx, ny))
        if self.asmbDict[pos]:
            self.boundaryTiles.discard(pos)

    def glue_energy(self, g1, g2):
        return GLUE_INTERACTION[GLUE_INDEX[g1]][GLUE_INDEX[g2]]

    def local_energy(self, tile, x, y):
        energy = 0
        for direction, (nx, ny) in self.neighbor_positions(x, y).items():
            neighbor = self.asmbDict[(nx, ny)]
            g1 = tile.get_glue(direction)
            if neighbor:
                g2 = neighbor.get_glue({'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}[direction])
                energy += self.glue_energy(g1, g2)
            else:
                energy += GLUE_BINDING_TO_FREE.get(g1, 0)
        return energy

    def grow_step(self):
        if not self.boundaryTiles:
            return

        (x, y) = random.choice(list(self.boundaryTiles))

        current_tile = self.asmbDict[(x, y)]

        if current_tile:
            energy_connected = self.local_energy(current_tile, x, y)
            mu = chem_potentials[current_tile.name]
            deltaF = -energy_connected + mu
            if deltaF < 0 or random.random() < math.exp(-deltaF / TEMPERATURE):
                self.asmbDict[(x, y)] = None
                self.update_boundary((x, y))
            return  # Only detachment or attachment per step

        # --- Attachment step ---
        candidates = []
        weights = []
        for tile in self.tile_set:
            if tile.name == "Seed":  # prevent placing seeds after initialization
                continue
            mu = chem_potentials[tile.name]
            energy = self.local_energy(tile, x, y)
            deltaF = energy - mu  # Free energy
            weight = math.exp(-deltaF / TEMPERATURE)
            candidates.append(tile)
            weights.append(weight)

        candidates.append(None)
        deltaF = 0
        weights.append(np.exp(-deltaF / TEMPERATURE))

        total_weight = sum(weights)
        if total_weight == 0:
            return

        chosen_tile = random.choices(candidates, weights=weights, k=1)[0]
        self.asmbDict[(x, y)] = chosen_tile
        self.update_boundary((x, y)) if chosen_tile else None


    def grow(self, steps):
        for _ in range(steps):
            self.grow_step()
            self.history.append(copy.deepcopy(self.asmbDict))




def plot_assembly(assembly, tile_size=1.0):
    fig, ax = plt.subplots(figsize=(4, 4))
    for (x, y), tile in assembly.asmbDict.items():
        if tile:
            rect = patches.Rectangle((x * tile_size, y * tile_size),
                                     tile_size, tile_size,
                                     linewidth=1,
                                     edgecolor='black',
                                     facecolor=colorMap.get(tile.name, 'gray'))
            ax.add_patch(rect)
    ax.set_aspect('equal')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_axis_off()
    # plt.grid(True)
    plt.show()






def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config('configs/vertical_horizontal.yaml')

    TEMPERATURE = config['TEMPERATURE']
    GLUE_TYPES = config['GLUE_TYPES']
    GLUE_INDEX = {g: i for i, g in enumerate(GLUE_TYPES)}
    GLUE_INTERACTION = config['GLUE_INTERACTION']
    GLUE_BINDING_TO_FREE = config['GLUE_BINDING_TO_FREE']
    chem_potentials = config['chem_potentials']
    colorMap = config['colorMap']

    tiles = [
        Tile(**tile) for tile in config['tiles']
    ]


    glue_based_assembly = AssemblyWithGlueInteractions(tiles)
    glue_based_assembly.grow(steps=1000)

    # history = glue_based_assembly.history
    # structureSize = [len(history[i].items()) for i in range(len(history))]
    # plt.plot(structureSize)
    # plt.show()
    plot_assembly(glue_based_assembly, tile_size=1.0)
