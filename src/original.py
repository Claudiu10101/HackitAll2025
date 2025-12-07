import math
import numpy as np
import random
from numba import jit
import time
import optuna

PLANE_UNSATISFY = 1000
PLANE_OVERCAPACITY = 100
AIRPORT_ZERO = 5000
AIRPORT_MUCHO = 1000
WEIGHTS = [1, 2, 3, 4]


class Airport:
    def __init__(self, id, cap, stock, load_cost, unload_cost):
        self.cap = cap
        self.id = id
        self.stock = stock
        self.load_cost = load_cost
        self.unload_cost = unload_cost


class Plane:
    def __init__(self, cap, fuel):
        self.cap = cap
        self.fuel = fuel


class Flight:
    def __init__(
        self, timestamp, distance, source, destination, plane, demands, resources
    ):
        self.destination = destination
        self.source = source
        self.timestamp = timestamp
        self.plane = plane
        self.demands = demands
        self.resources = resources
        self.distance = distance

    def calc_cost(self, state):
        """this is cost simulator function that applies the flight to the airports states and returns the cost incurred during flights"""
        cost = 0
        for i in range(len(self.resources)):
            cost += (
                self.destination.load_cost[i] * self.resources[i]
                + self.source.unload_cost[i] * self.resources[i]
                + self.plane.fuel * self.distance * self.resources[i] * WEIGHTS[i]
            )
            cost += (
                PLANE_UNSATISFY
                * self.distance
                * max(0, self.demands[i] - self.resources[i])
            )
            cost += (
                PLANE_OVERCAPACITY
                * self.distance
                * max(0, self.resources[i] - self.plane.cap[i])
            )

            state[self.timestamp][self.source.id][i] -= self.resources[i]
            state[self.timestamp][self.destination.id][i] += self.resources[i]
        return cost

    def undo(self, state):
        """undo the flight from the state"""
        for i in range(len(self.resources)):
            state[self.timestamp][self.source.id][i] += self.resources[i]
            state[self.timestamp][self.destination.id][i] -= self.resources[i]

    def gen_neighbour(self, delta, offset, selector):
        """generate a neighbour flight by perturbing the resources"""
        neigh = Flight(
            self.timestamp,
            self.distance,
            self.source,
            self.destination,
            self.plane,
            self.demands,
            self.resources,
        )
        for res in range(4):
            shape = self.resources[res].shape
            neigh.resources[res] = neigh.resources[res] * np.random.uniform(
                1 - delta, 1 + delta, size=shape
            ) + np.random.uniform(-offset, offset, size=shape)
        return neigh


STATE = [[[0] * 4 for _ in range(161)] for _ in range(720)]
AIRPORTS = [
    Airport(i, [10, 10, 12, 20], [10, 10, 12, 20], [10, 10, 12, 20], [10, 10, 12, 20])
    for i in range(161)
]
PLANES = [Plane([100, 100, 100, 100], 10) for _ in range(4)]


def cost_function(flights, state, timestamp):
    """Calculate total cost that you will have incurred during and after a round of flights"""

    cost = 0
    airports = set()

    for f in flights:
        cost += f.calc_cost(state)
        airports.add(f.destination.id)
        airports.add(f.source.id)

    # this is where we must compute the penalties for airports in the next 24h so that they are taken into account
    # TODO this should be for 24h using the forecast_pentalties function
    for i in airports:
        for res in range(4):
            cost += AIRPORT_ZERO * max(0, -state[timestamp][i][res])
            cost += AIRPORT_MUCHO * max(
                0, state[timestamp][i][res] - AIRPORTS[i].cap[res]
            )

    for f in flights:
        f.undo(state)

    return cost


def get_neighbor(flights, delta, offset, selector):
    """Generate a neighbouring solution by perturbing each flight"""
    new_flights = []
    for f in flights:
        new_flights.append(f.gen_neighbour(delta, offset, selector))

    return new_flights


def acceptance_probability(old_cost, new_cost, temperature):
    if new_cost < old_cost:
        return 1.0
    return math.exp((old_cost - new_cost) / temperature)


# Placeholders for initial data must be plugged in
POSSIBLE_STOCKS = [
    np.array(
        [
            random.randint(0, 100),
            random.randint(0, 100),
            random.randint(0, 100),
            random.randint(0, 100),
        ]
    )
    for _ in range(10)
]
INITIAL_FLIGHTS = [
    Flight(
        0,
        100,
        AIRPORTS[random.randint(0, 80)],
        AIRPORTS[random.randint(81, 160)],
        PLANES[random.randint(0, 3)],
        POSSIBLE_STOCKS[i],
        POSSIBLE_STOCKS[i],
    )
    for i in range(10)
]


def run_simulated_annealing(
    initial_temp,
    min_temp,
    cooling_rate,
    markov_len,
    neigh_delta,
    neigh_offset,
    neigh_selector,
):
    current_temp = initial_temp

    # take all the flights in the current round (we consider that resources = demands at start)
    current_solution = INITIAL_FLIGHTS[:]
    current_cost = cost_function(current_solution, STATE, 0)

    best_solution = current_solution
    best_cost = current_cost

    while current_temp > min_temp:
        for i in range(markov_len):
            new_solution = get_neighbor(
                current_solution, neigh_delta, neigh_offset, neigh_selector
            )
            new_cost = cost_function(new_solution, STATE, 0)

            if (
                acceptance_probability(current_cost, new_cost, current_temp)
                > random.random()
            ):
                current_solution = new_solution
                current_cost = new_cost

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = current_solution

        current_temp *= cooling_rate

    return best_cost, best_solution


# --- 3. The Brain: Bayesian Optimization (Optuna) ---
def objective(trial):
    """
    This function is what Optuna optimizes.
    It suggests params, runs the worker, and returns the result.
    """

    t_start = 100
    alpha = trial.suggest_float("cooling_rate", 0.8, 0.99)
    m_len = trial.suggest_int("markov_chain_length", 10, 50)
    t_min = trial.suggest_float("min_temp", 0.01, 1)
    delta = trial.suggest_float("neighbour_delta", 0.01, 0.99)
    offset = trial.suggest_float("neighbour_offset", 0.1, 10)
    selector = trial.suggest_float("neighbour_selector", 0, 1)

    # B. Run the actual algorithm
    # Note: Since SA is random, it's good practice to run it 3 times
    # and take the average to give Optuna a stable signal.
    scores = []
    for _ in range(5):
        score, _ = run_simulated_annealing(
            t_start, t_min, alpha, m_len, delta, offset, selector
        )
        scores.append(score)

    average_best_cost = min(scores)

    # C. Return the result to Optuna
    return average_best_cost


# --- 4. Execution ---
if __name__ == "__main__":
        t_start = 100
    alpha = trial.suggest_float("cooling_rate", 0.8, 0.99)
    m_len = trial.suggest_int("markov_chain_length", 10, 50)
    t_min = trial.suggest_float("min_temp", 0.01, 1)
    delta = trial.suggest_float("neighbour_delta", 0.01, 0.99)
    offset = trial.suggest_float("neighbour_offset", 0.1, 10)
    selector = trial.suggest_float("neighbour_selector", 0, 1)

	run_simulated_annealing(t_start, t_min,alpha,m_len,delta,offset,selector)
