import numpy as np
import random

def RNDprobab():
    return random.random()

def Scale(val, min_val_in, max_val_in, min_val_out, max_val_out):
    if max_val_in == min_val_in:
        return min_val_out
    scaled_val = ((val - min_val_in) / (max_val_in - min_val_in)) * (max_val_out - min_val_out) + min_val_out
    return int(max(min_val_out, min(max_val_out, scaled_val)))

def SeInDiSp(val, range_min, range_max, range_step):
    val = max(range_min, min(range_max, val))
    if range_step > 0 and range_step < (range_max - range_min):
        val = range_min + round((val - range_min) / range_step) * range_step
    return val

def GaussDistribution(mean, min_val, max_val, scale=1.0):
    std_dev = 0.1 * scale
    new_val = np.random.normal(loc=mean, scale=std_dev)
    return max(min_val, min(max_val, new_val))

class TETA_Agent:
    def __init__(self, num_coords):
        self.c = np.zeros(num_coords)
        self.f = -np.inf
        self.cB = np.zeros(num_coords)
        self.fB = -np.inf

class TETA_Optimizer:
    def __init__(self, num_coords, popSize=50):
        self.popSize = popSize
        self.coords = num_coords
        self.population = []
        self.revision = False
        self.cB = np.zeros(num_coords)
        self.fB = -np.inf
        self.rangeMin = np.zeros(num_coords)
        self.rangeMax = np.zeros(num_coords)
        self.rangeStep = np.zeros(num_coords)

    def Init(self, rangeMinP, rangeMaxP, rangeStepP):
        if len(rangeMinP) != self.coords:
            print("Error: Range arrays size mismatch.")
            return False
        self.rangeMin = np.array(rangeMinP)
        self.rangeMax = np.array(rangeMaxP)
        self.rangeStep = np.array(rangeStepP)
        self.population = [TETA_Agent(self.coords) for _ in range(self.popSize)]
        return True

    def initialize_anchors(self):
        for i in range(self.popSize):
            agent = self.population[i]
            for c in range(self.coords):
                val = random.uniform(self.rangeMin[c], self.rangeMax[c])
                agent.c[c] = SeInDiSp(val, self.rangeMin[c], self.rangeMax[c], self.rangeStep[c])
            agent.cB[:] = agent.c[:]

    def Moving(self):
        if not self.revision:
            self.initialize_anchors()
            self.revision = True
            return
        for i in range(self.popSize):
            current_agent = self.population[i]
            for c in range(self.coords):
                rnd = RNDprobab()
                rnd *= rnd
                pair = Scale(rnd, 0.0, 1.0, 0, self.popSize - 1)
                new_value = 0.0
                if i != pair:
                    selected_agent = self.population[pair]
                    if i < pair:
                        new_value = current_agent.c[c] + rnd * (selected_agent.cB[c] - current_agent.cB[c])
                    else:
                        if RNDprobab() > rnd:
                            new_value = current_agent.cB[c] + (1.0 - rnd) * (selected_agent.cB[c] - current_agent.cB[c])
                        else:
                            new_value = selected_agent.cB[c]
                else:
                    new_value = GaussDistribution(self.cB[c], self.rangeMin[c], self.rangeMax[c], scale=1.0)
                current_agent.c[c] = SeInDiSp(new_value, self.rangeMin[c], self.rangeMax[c], self.rangeStep[c])

    def Revision(self):
        for i in range(self.popSize):
            agent = self.population[i]
            if agent.f > self.fB:
                self.fB = agent.f
                self.cB[:] = agent.c[:]
            if agent.f > agent.fB:
                agent.fB = agent.f
                agent.cB[:] = agent.c[:]
        self.population.sort(key=lambda agent: agent.fB, reverse=True)

    def optimize(self, fitness_function, max_iterations, rangeMinP, rangeMaxP, rangeStepP):
        if not self.Init(rangeMinP, rangeMaxP, rangeStepP):
            return "Initialization Failed"
        self.Moving()
        for agent in self.population:
            agent.f = fitness_function(agent.c)
        self.Revision()
        for epoch in range(max_iterations):
            self.Moving()
            for agent in self.population:
                agent.f = fitness_function(agent.c)
            self.Revision()
        return self.cB, self.fB
