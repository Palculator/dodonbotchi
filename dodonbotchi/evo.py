import copy
import json
import math
import os
import random

from pathlib import Path
from time import sleep

import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw, ImageFilter

from deap import base
from deap import creator
from deap import tools

from .config import CFG as cfg
from .mame import Ddonpach, get_action_str
from .util import ensure_directories

sns.set()

DIRECTIONS = []

for vert in range(3):
    for hori in range(3):
        if not vert and not hori:
            continue
        DIRECTIONS.append((vert, hori))

assert len(DIRECTIONS) == 8

WINDOW_SIZE = 64
CXPB, MUTPB = 0.5, 0.2
GENS = 25


def setup_plot(plot, title):
    plot.clear()
    plot.cla()
    plot.set_title(title, fontsize=10)
    plot.set_xlabel('@Signaltonsalat', fontsize=4)
    plot.set_ylabel('')
    plot.set_xticks([])
    plot.set_yticks([])


class EvoPachi:
    size = WINDOW_SIZE

    def __init__(self, cwd):
        self.rng = random.Random()

        cwd = Path(cwd)
        self.inp = cwd / 'inp'
        self.fxd = cwd / 'fxd'
        self.rnd = cwd / 'rnd'
        snp = cwd / 'snp'
        ensure_directories(*[str(p) for p in [self.inp, self.rnd, snp]])

        if not self.fxd.exists():
            with open(self.fxd, 'w') as fixed:
                pass

        self.ddonpach = Ddonpach('rc')
        self.ddonpach.inp_dir = str(self.inp)
        self.ddonpach.snp_dir = str(snp)
        self.ddonpach.start_mame()
        # self.ddonpach.read_observation()

        self.fitness = None
        self.individual = None
        self.toolbox = None

        self.fig = None
        self.game = None
        self.game_img = None
        self.current_input = None
        self.current_input_img = None
        self.current_combo = None
        self.current_score = None
        self.success_rate = None
        self.best_combo = None
        self.best_score = None

        self.current_deaths = 0
        self.current_success = 0

        self.fixed_steps = 0

        self.frame = 1

        self.pop = None

        self.reset_plots()
        self.setup_deap()

    def reset_plots(self):
        self.fig = plt.figure(1, figsize=(8, 4))
        self.game = plt.subplot2grid((4, 8), (0, 0), colspan=3, rowspan=4)
        self.game_img = None
        self.success_rate = plt.subplot2grid(
            (4, 8), (0, 3), colspan=2, rowspan=2)
        self.best_score = plt.subplot2grid((4, 8), (0, 5), colspan=2)
        self.best_combo = plt.subplot2grid((4, 8), (1, 5), colspan=2)
        self.current_input = plt.subplot2grid(
            (4, 8), (2, 3), colspan=2, rowspan=2)
        self.current_score = plt.subplot2grid((4, 8), (2, 5), colspan=2)
        self.current_combo = plt.subplot2grid((4, 8), (3, 5), colspan=2)

        setup_plot(self.game, 'Game')
        self.reset_best()
        self.reset_current()

    def reset_current(self):
        self.current_input.clear()
        self.current_input_img = None
        self.current_score.clear()
        self.current_combo.clear()
        self.current_input.cla()
        self.current_score.cla()
        self.current_combo.cla()

        self.current_input.set_title('')
        self.current_input.set_xlabel('Input', fontsize=10)
        self.current_input.set_ylabel('')
        self.current_input.set_xticks([])
        self.current_input.set_yticks([])

        self.current_score.set_title('')
        self.current_score.set_xlabel('')
        self.current_score.set_ylabel('')
        self.current_score.set_xticks([])
        self.current_score.set_yticks([])

        self.current_combo.set_title('')
        self.current_combo.set_xlabel('Score & Combo', fontsize=10)
        self.current_combo.set_ylabel('')
        self.current_combo.set_xticks([])
        self.current_combo.set_yticks([])

        self.current_combo.set_xlim(-1, WINDOW_SIZE)
        self.current_score.set_xlim(-1, WINDOW_SIZE)

    def reset_best(self):
        self.success_rate.clear()
        self.best_score.clear()
        self.best_combo.clear()
        self.success_rate.cla()
        self.best_score.cla()
        self.best_combo.cla()

        self.success_rate.set_title('Success/Death', fontsize=10)
        self.success_rate.set_xlabel('')
        self.success_rate.set_ylabel('')
        self.success_rate.set_xticks([])
        self.success_rate.set_yticks([])

        self.best_score.set_title('Best Score & Combo / Gen', fontsize=10)
        self.best_score.set_xlabel('')
        self.best_score.set_ylabel('')
        self.best_score.set_xticks([])
        self.best_score.set_yticks([])

        self.best_combo.set_title('')
        self.best_combo.set_xlabel('')
        self.best_combo.set_ylabel('')
        self.best_combo.set_xticks([])
        self.best_combo.set_yticks([])

        self.best_combo.set_xlim(-1, GENS)
        self.best_score.set_xlim(-1, GENS)

    def setup_deap(self):
        self.fitness = creator.create('FitnessMax', base.Fitness,
                                      weights=(1.0, 1.0))

        creator.create('Individual', list, fitness=creator.FitnessMax)
        self.individual = creator.Individual

        self.toolbox = base.Toolbox()
        self.toolbox.register('individual',
                              self.generate_candidate, EvoPachi.size)
        self.toolbox.register('population', tools.initRepeat, list,
                              self.toolbox.individual)
        self.toolbox.register('evaluate', self.evaluate)
        self.toolbox.register('mate', tools.cxTwoPoint)
        self.toolbox.register('mutate', self.mutate)
        self.toolbox.register('select', tools.selTournament, tournsize=3)

    def count_fixed_steps(self):
        self.fixed_steps = 0
        with open(self.fxd, 'r') as fixed:
            for action in fixed:
                self.fixed_steps += 1

    def replay(self):
        with open(self.fxd, 'r') as fixed:
            for action in fixed:
                self.ddonpach.send_action(action)
                observation = self.ddonpach.read_observation()

    def sample_action(self):
        vert, hori = self.rng.choice(DIRECTIONS)
        shot = 1
        return get_action_str(vert=vert, hori=hori, shot=shot)

    def generate_candidate(self, size):
        candidate = [self.sample_action() for _ in range(size)]
        candidate = self.individual(candidate)
        return candidate

    def generate_candidates(self, count):
        candidates = []
        for i in range(count):
            candidate = self.generate_candidate(EvoPachi.size)
            candidates.append(candidate)
        return candidates

    def mutate(self, individual):
        spot = self.rng.randint(0, len(individual) - 1)
        individual[spot] = self.sample_action()
        return individual,

    def draw_inputs(self, cursor, candidate):
        num = math.floor(math.sqrt(WINDOW_SIZE))
        dim = 7 * num
        img = Image.new('RGB', (dim, dim), '#FFFFFF')
        drw = ImageDraw.Draw(img)

        for idx, action in enumerate(candidate):
            background = '#AAAAAA'
            if idx < cursor:
                background = '#FFFFFF'

            x = idx % num
            y = math.floor(idx / num)

            rectangle = (x * 7 + 1, y * 7 + 1, (x + 1)
                         * 7 - 2, (y + 1) * 7 - 2)
            drw.rectangle(rectangle, fill=background)

            rectangle = (x * 7 + 3, y * 7 + 2, x * 7 + 3, y * 7 + 2)
            colour = '#222222'
            if action[0] == '2':
                colour = '#FF0000'

            drw.rectangle(rectangle, fill=colour)

            rectangle = (x * 7 + 3, y * 7 + 4, x * 7 + 3, y * 7 + 4)
            colour = '#222222'
            if action[0] == '1':
                colour = '#FF0000'

            drw.rectangle(rectangle, fill=colour)

            rectangle = (x * 7 + 2, y * 7 + 3, x * 7 + 2, y * 7 + 3)
            colour = '#222222'
            if action[1] == '1':
                colour = '#FF0000'

            drw.rectangle(rectangle, fill=colour)

            rectangle = (x * 7 + 4, y * 7 + 3, x * 7 + 4, y * 7 + 3)
            colour = '#222222'
            if action[1] == '2':
                colour = '#FF0000'

            drw.rectangle(rectangle, fill=colour)

        return img

    def evaluate(self, candidate):
        self.ddonpach.send_load_state(cfg.save_state)
        self.replay()
        self.reset_current()
        trace = []
        combos = []
        scores = []
        for idx, action in enumerate(candidate):
            self.ddonpach.send_action(action)
            observation = self.ddonpach.read_observation()

            score = observation['score']
            combo = observation['combo']

            snap = self.ddonpach.get_snap()
            if self.game_img:
                self.game_img.set_data(snap)
            else:
                self.game_img = self.game.imshow(snap)

            inputs = self.draw_inputs(idx, candidate)
            if self.current_input_img:
                self.current_input_img.set_data(inputs)
            else:
                self.current_input_img = self.current_input.imshow(inputs)
            self.current_score.plot(idx, score, 'ro', markersize=1)
            self.current_combo.plot(idx, combo, 'bo', markersize=1)

            if observation['death']:
                img = Image.open('death.png')
                self.current_input.imshow(img)
                self.current_deaths += 1
                self.plot_success_rate()
                plt.savefig(
                    str(self.rnd / '{:09}.png'.format(self.frame)), dpi=150)
                self.frame += 1
                return -100, -100
            else:
                plt.savefig(
                    str(self.rnd / '{:09}.png'.format(self.frame)), dpi=150)
                self.frame += 1

            scores.append(score)
            combos.append(combo)
            trace.append(action)

        self.current_success += 1
        self.plot_success_rate()

        return np.average(combos), score

    def plot_success_rate(self):
        total = self.current_success + self.current_deaths
        rate = [self.current_success / total, self.current_deaths / total]
        self.success_rate.pie(rate, colors=['g', 'r'])

    def get_success_rate(self, last_success):
        total = [1 for ind in self.pop if ind.fitness.valid]
        total = len(total) + 1
        success = [1 for ind in self.pop
                   if ind.fitness.valid and ind.fitness.values[1] >= 0]
        failure = [1 for ind in self.pop
                   if ind.fitness.valid and ind.fitness.values[1] < 0]

        if last_success:
            success.append(1)
        else:
            failure.append(1)

        sizes = [len(success) / total, len(failure) / total]

        return sizes

    def evolution_step(self):
        self.reset_best()
        self.current_deaths = 0
        self.current_success = 0
        g = 0
        title = '{} steps fixed, generation {}/{}'
        title = title.format(self.fixed_steps, g, GENS)
        setup_plot(self.game, title)
        self.game_img = None
        self.pop = self.toolbox.population(n=10)
        fitnesses = list(map(self.toolbox.evaluate, self.pop))
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit
        fits = [ind.fitness.values[0] for ind in self.pop]
        sortpop = sorted(self.pop, key=lambda p: p.fitness.values[1])
        self.best_score.plot(g, sortpop[-1].fitness.values[1],
                             'ro', markersize=1)
        self.best_combo.plot(g, sortpop[-1].fitness.values[0],
                             'bo', markersize=1)

        known_best = sortpop[-1]

        while g < GENS:
            g += 1
            title = '{} steps fixed, generation {}/{}'
            title = title.format(self.fixed_steps, g, GENS)
            setup_plot(self.game, title)
            self.game_img = None
            offspring = self.toolbox.select(self.pop, len(self.pop))
            offspring = list(map(self.toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if self.rng.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if self.rng.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            sortpop = sorted(self.pop, key=lambda p: p.fitness.values[1])
            if sortpop[-1].fitness.values > known_best.fitness.values:
                known_best = sortpop[-1]

            self.best_score.plot(g, known_best.fitness.values[1],
                                 'ro', markersize=1)
            self.best_combo.plot(g, known_best.fitness.values[0],
                                 'bo', markersize=1)

            self.pop[:] = offspring

        return known_best, known_best.fitness.values[1]

    def backtrack(self):
        with open(self.fxd, 'r') as fixed:
            lines = fixed.readlines()

        if len(lines) >= EvoPachi.size:
            lines = lines[:len(lines) - EvoPachi.size]
        else:
            lines = []

        with open(self.fxd, 'w') as fixed:
            for line in lines:
                fixed.write('{}\n'.format(line))

    def progression(self):
        while True:
            self.count_fixed_steps()
            best, score = self.evolution_step()
            if score >= 0:
                with open(self.fxd, 'a') as fixed:
                    for line in best:
                        fixed.write('{}\n'.format(line))
            else:
                self.backtrack()


def evolve(cwd):
    e = EvoPachi(cwd)
    e.progression()


def replay(cwd):
    e = EvoPachi(cwd)
    e.replay()
