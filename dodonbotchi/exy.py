import copy
import json
import logging as log
import math
import os
import queue
import random
import threading

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
from .util import ensure_directories, get_now_string

sns.set()

DIRECTIONS = []

for vert in range(3):
    for hori in range(3):
        if not vert and not hori:
            continue
        DIRECTIONS.append((vert, hori))

assert len(DIRECTIONS) == 8

WINDOW_SIZE = 36
CXPB, MUTPB = 0.5, 0.2
POP = 8
GENS = 5

FONT_SIZE = 10
WATERMARK = '@Signaltonsalat'
WATERMARK_SIZE = 8

save_queue = queue.Queue()
finished = False


def saver():
    global save_queue
    while save_queue.full() or not finished:
        img, path = save_queue.get()
        img.save(path)
        save_queue.task_done()


def clear_labels_ticks(*plots):
    for plot in plots:
        plot.clear()
        plot.cla()

        plot.set_title('')
        plot.set_xlabel('')
        plot.set_ylabel('')
        plot.set_xticks([])
        plot.set_yticks([])


def draw_inputs(cursor, candidate):
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


def get_best_individual(pop):
    sortpop = sorted(pop, key=lambda p: p.fitness.values[0])
    return sortpop[-1]


class DdonpachSyncError(Exception):
    pass


class Exy:
    size = WINDOW_SIZE

    def __init__(self, cwd):
        self.rng = random.Random()

        cwd = Path(cwd)
        self.inp = cwd / 'inp'
        self.fxd = cwd / 'fxd'
        self.rnd = cwd / 'rnd'
        self.snp = cwd / 'snp'
        ensure_directories(*[str(p) for p in [self.inp, self.rnd, self.snp]])

        if not self.fxd.exists():
            with open(self.fxd, 'w') as fixed:
                pass

        self.fitness = None
        self.individual = None
        self.toolbox = None

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
        self.saved = 1

        self.reset_plots()
        self.setup_deap()

    def reset_plots(self):
        plt.figure(1, figsize=(8, 4))

        grid = (4, 8)

        self.game = plt.subplot2grid(grid, (0, 0), colspan=3, rowspan=4)
        self.game_img = None

        self.success_rate = plt.subplot2grid(grid, (0, 3),
                                             colspan=2, rowspan=2)

        self.best_score = plt.subplot2grid(grid, (0, 5), colspan=2)
        self.best_combo = plt.subplot2grid(grid, (1, 5), colspan=2)

        self.current_input = plt.subplot2grid(grid, (2, 3),
                                              colspan=2, rowspan=2)

        self.current_score = plt.subplot2grid(grid, (2, 5), colspan=2)
        self.current_combo = plt.subplot2grid(grid, (3, 5), colspan=2)

        self.reset_game_plot('Game')
        self.reset_best()
        self.reset_current()

    def reset_game_plot(self, title):
        clear_labels_ticks(self.game)
        self.game.set_title(title, fontsize=FONT_SIZE)
        self.game.set_xlabel(WATERMARK, fontsize=WATERMARK_SIZE)
        self.game_img = None

    def reset_current(self):
        clear_labels_ticks(self.current_input, self.current_score,
                           self.current_combo)

        self.current_input_img = None

        self.current_input.set_xlabel('Input', fontsize=FONT_SIZE)
        self.current_combo.set_xlabel('Score & Combo', fontsize=FONT_SIZE)

        self.current_combo.set_xlim(-1, WINDOW_SIZE)
        self.current_score.set_xlim(-1, WINDOW_SIZE)

    def reset_best(self):
        clear_labels_ticks(self.success_rate, self.best_score, self.best_combo)

        self.success_rate.set_title('Success/Death', fontsize=FONT_SIZE)

        self.best_score.set_title('Best Score & Combo / Gen',
                                  fontsize=FONT_SIZE)

        self.best_combo.set_xlim(-1, GENS)
        self.best_score.set_xlim(-1, GENS)

    def setup_deap(self):
        self.fitness = creator.create('FitnessMax', base.Fitness,
                                      weights=(1.0, 1.0))

        creator.create('Individual', list, fitness=creator.FitnessMax)
        self.individual = creator.Individual

        self.toolbox = base.Toolbox()
        self.toolbox.register('individual',
                              self.generate_candidate, Exy.size)
        self.toolbox.register('population', tools.initRepeat, list,
                              self.toolbox.individual)
        self.toolbox.register('evaluate', self.evaluate)
        self.toolbox.register('mate', tools.cxOnePoint)
        self.toolbox.register('mutate', self.mutate)
        self.toolbox.register('select', tools.selTournament, tournsize=3)

    def open_ddonpach(self, recording):
        ddonpach = Ddonpach(recording)
        ddonpach.inp_dir = str(self.inp)
        ddonpach.snp_dir = str(self.snp)
        return ddonpach

    def count_fixed_steps(self):
        self.fixed_steps = 0
        with open(self.fxd, 'r') as fixed:
            for action in fixed:
                self.fixed_steps += 1

    def replay(self, ddonpach):
        ddonpach.send_command(command='wait', frames=480)
        state = ddonpach.read_gamestate()
        with open(self.fxd, 'r') as fixed:
            for line in fixed:
                action, score = line.split(';')
                ddonpach.send_action(action)
                state = ddonpach.read_gamestate()
                if int(score) != state['score']:
                    raise DdonpachSyncError('Score out of sync during replay.')
            return state['score']
        return -1

    def sample_action(self, count=1):
        vert, hori = self.rng.choice(DIRECTIONS)
        shot = 1
        return get_action_str(vert=vert, hori=hori, shot=shot)

    def generate_candidate(self, size):
        candidate = [self.sample_action(count=i) for i in range(size)]
        candidate = self.individual(candidate)
        return candidate

    def generate_candidates(self, count):
        candidates = []
        for i in range(count):
            candidate = self.generate_candidate(Exy.size)
            candidates.append(candidate)
        return candidates

    def mutate(self, individual):
        spot = self.rng.randint(0, len(individual) - 1)
        individual[spot] = self.sample_action()
        return individual,

    def render_snap(self, snap):
        if self.game_img:
            self.game_img.set_data(snap)
        else:
            self.game_img = self.game.imshow(snap)

    def render_inputs(self, inputs):
        if self.current_input_img:
            self.current_input_img.set_data(inputs)
        else:
            self.current_input_img = self.current_input.imshow(inputs)

    def enqueue_plot(self, path):
        global save_queue
        plt.gcf().set_dpi(300)
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        img = Image.frombytes('RGB', canvas.get_width_height(),
                              canvas.tostring_rgb())
        save_queue.put((img, path))

    def evaluate(self, candidate):
        global save_queue

        recording = get_now_string()
        starting_score = -1
        for _ in range(16):
            with self.open_ddonpach(recording) as ddonpach:
                try:
                    starting_score = self.replay(ddonpach)
                except DdonpachSyncError as err:
                    log.error('Desync!')
                    log.exception(err)
                    continue

                save_queue.join()

                self.reset_current()

                combos = []

                for idx, action in enumerate(candidate):
                    ddonpach.send_action(action)
                    observation = ddonpach.read_gamestate()

                    score = observation['score']
                    combo = observation['combo']

                    snap = ddonpach.get_snap()
                    self.render_snap(snap)

                    inputs = draw_inputs(idx, candidate)
                    self.render_inputs(inputs)

                    self.current_score.plot(idx, score, 'ro', markersize=1)
                    self.current_combo.plot(idx, combo, 'bo', markersize=1)

                    out_file = '{:09}.png'.format(int(self.saved))
                    out_file = str(self.rnd / out_file)
                    self.frame += 1
                    if observation['death']:
                        img = Image.open('death.png')
                        self.current_input.imshow(img)
                        self.current_deaths += 1
                        self.plot_success_rate()

                        self.enqueue_plot(out_file)
                        self.saved += 1
                        return -100 / (idx + 1), -100 / (idx + 1)
                    else:
                        if self.frame % 8 == 0:
                            self.enqueue_plot(out_file)
                            self.saved += 1

                    combos.append(combo)

                    action = action.split(';')[0]
                    candidate[idx] = '{};{}'.format(action, score)

                self.current_success += 1
                self.plot_success_rate()

                increase = score - starting_score
                increase //= 5000
                if combos[-1] > 0:
                    return increase, int(np.average(combos))
                else:
                    return increase, -1

        # If we reach this, the replay desynced 16 times.
        return -10000, -10000

    def plot_success_rate(self):
        total = self.current_success + self.current_deaths
        rate = [self.current_success / total, self.current_deaths / total]
        self.success_rate.pie(rate, colors=['g', 'r'])

    def evaluate_population(self, pop):
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

    def mate_population(self, pop):
        offspring = self.toolbox.select(pop, len(pop))
        offspring = list(map(self.toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if self.rng.random() < CXPB:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        return offspring

    def mutate_offspring(self, offspring):
        for mutant in offspring:
            if self.rng.random() < MUTPB:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

    def evolution_step(self):
        self.reset_best()

        self.current_deaths = 0
        self.current_success = 0

        gen = 0

        title = '{} steps fixed, generation {}/{}'
        title = title.format(self.fixed_steps, gen, GENS)
        self.reset_game_plot(title)

        pop = self.toolbox.population(n=POP)

        self.evaluate_population(pop)

        best_ind = get_best_individual(pop)

        self.best_score.plot(gen, best_ind.fitness.values[0],
                             'ro', markersize=1)
        self.best_combo.plot(gen, best_ind.fitness.values[1],
                             'bo', markersize=1)

        known_best = best_ind

        while gen < GENS:
            gen += 1

            title = '{} steps fixed, generation {}/{}'
            title = title.format(self.fixed_steps, gen, GENS)
            self.reset_game_plot(title)

            offspring = self.mate_population(pop)
            self.mutate_offspring(offspring)
            if random.random() < 0.25:
                print('Introducing random candidate.')
                intro = self.generate_candidate(Exy.size)
                offspring.append(intro)
                offspring = offspring[1:]

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            self.evaluate_population(invalid_ind)

            best_ind = get_best_individual(pop)
            if best_ind.fitness.values > known_best.fitness.values:
                known_best = best_ind

            score = known_best.fitness.values[0]
            combo = known_best.fitness.values[1]

            self.best_score.plot(gen, score, 'ro', markersize=1)
            self.best_combo.plot(gen, combo, 'bo', markersize=1)

            pop[:] = offspring

        return known_best

    def backtrack(self):
        with open(self.fxd, 'r') as fixed:
            lines = fixed.readlines()

        if len(lines) >= Exy.size:
            lines = lines[:len(lines) - Exy.size]
        else:
            lines = []

        with open(self.fxd, 'w') as fixed:
            for line in lines:
                fixed.write('{}'.format(line))

    def progression(self):
        while True:
            self.count_fixed_steps()
            best = self.evolution_step()
            score = best.fitness.values[0]
            if score >= 0:
                with open(self.fxd, 'a') as fixed:
                    for line in best:
                        fixed.write('{}\n'.format(line))
            else:
                self.backtrack()


def evolve(cwd):
    global finished

    thread = threading.Thread(target=saver)
    thread.start()

    e = Exy(cwd)
    e.progression()
    finished = True


def replay(cwd):
    e = Exy(cwd)
    e.replay()
