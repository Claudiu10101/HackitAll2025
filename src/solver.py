import math
import numpy as np
import random
from numba import njit
import time
import optuna
from optuna.samplers import CmaEsSampler

# --- Constants ---
PLANE_UNSATISFY = 1000.0
PLANE_OVERCAPACITY = 100.0
AIRPORT_ZERO = 5000.0
AIRPORT_MUCHO = 1000.0
WEIGHTS = np.array([1.0, 2.0, 3.0, 4.0])


# --- Original Classes (Used only for initial data generation) ---
class Airport:
    def __init__(self, id, cap, stock, load_cost, unload_cost):
        self.cap = np.array(cap, dtype=np.float64)
        self.id = id
        self.stock = np.array(stock, dtype=np.float64)
        self.load_cost = np.array(load_cost, dtype=np.float64)
        self.unload_cost = np.array(unload_cost, dtype=np.float64)


class Plane:
    def __init__(self, cap, fuel):
        self.cap = np.array(cap, dtype=np.float64)
        self.fuel = float(fuel)


class Flight:
    def __init__(self, timestamp, distance, source, destination, plane, demands, resources):
        self.destination = destination
        self.source = source
        self.timestamp = timestamp
        self.plane = plane
        self.demands = np.array(demands, dtype=np.float64)
        self.resources = np.array(resources, dtype=np.float64)
        self.distance = float(distance)


# --- Data Generation ---
# Creating dummy data exactly as in the original script
AIRPORTS = [Airport(i, [10, 10, 12, 20], [10, 10, 12, 20], [10, 10, 12, 20], [10, 10, 12, 20]) for i
            in range(161)]
PLANES = [Plane([100, 100, 100, 100], 10) for _ in range(4)]
POSSIBLE_STOCKS = [np.array([random.randint(0, 100) for _ in range(4)], dtype=np.float64) for _ in
                   range(20)]
INITIAL_FLIGHT_OBJS = [
    Flight(0, 100, AIRPORTS[random.randint(0, 80)], AIRPORTS[random.randint(81, 160)],
           PLANES[random.randint(0, 3)], POSSIBLE_STOCKS[i], POSSIBLE_STOCKS[i]) for i in range(20)]


# --- 1. Data Conversion: Objects -> NumPy Arrays ---
# We flatten the objects into arrays so Numba can consume them efficiently.

def pack_data(flights, airports):
    n_flights = len(flights)
    n_airports = len(airports)

    # Flight Data
    f_timestamps = np.zeros(n_flights, dtype=np.int32)
    f_dists = np.zeros(n_flights, dtype=np.float64)
    f_sources = np.zeros(n_flights, dtype=np.int32)
    f_dests = np.zeros(n_flights, dtype=np.int32)
    f_demands = np.zeros((n_flights, 4), dtype=np.float64)
    # This is the "Solution" genome we will optimize
    f_resources = np.zeros((n_flights, 4), dtype=np.float64)

    # Flight/Plane Characteristics
    f_plane_fuel = np.zeros(n_flights, dtype=np.float64)
    f_plane_caps = np.zeros((n_flights, 4), dtype=np.float64)

    for i, f in enumerate(flights):
        f_timestamps[i] = f.timestamp
        f_dists[i] = f.distance
        f_sources[i] = f.source.id
        f_dests[i] = f.destination.id
        f_demands[i] = f.demands
        f_resources[i] = f.resources
        f_plane_fuel[i] = f.plane.fuel
        f_plane_caps[i] = f.plane.cap

    # Airport Data
    a_caps = np.zeros((n_airports, 4), dtype=np.float64)
    a_load_cost = np.zeros((n_airports, 4), dtype=np.float64)
    a_unload_cost = np.zeros((n_airports, 4), dtype=np.float64)

    for i, a in enumerate(airports):
        a_caps[i] = a.cap
        a_load_cost[i] = a.load_cost
        a_unload_cost[i] = a.unload_cost

    return (f_timestamps, f_dists, f_sources, f_dests, f_demands, f_resources,
            f_plane_fuel, f_plane_caps, a_caps, a_load_cost, a_unload_cost)


# --- 2. The Numba Optimized Logic ---

@njit(cache=True, nogil=True)
def fast_cost_function(
        f_resources,  # The variable we are optimizing
        f_timestamps, f_dists, f_sources, f_dests, f_demands, f_plane_fuel, f_plane_caps,
        # Flight Static Data
        a_caps, a_load_cost, a_unload_cost,  # Airport Static Data
        state_buffer  # Pre-allocated scratchpad for state
):
    # Reset State Buffer: (timestamps, airports, resources)
    # Assuming max timestamp 720 and 161 airports as per original code
    state_buffer.fill(0.0)

    total_cost = 0.0
    n_flights = len(f_resources)

    # 1. Calculate Flight Costs and Update State
    for i in range(n_flights):
        dist = f_dists[i]
        fuel = f_plane_fuel[i]
        t = f_timestamps[i]
        src = f_sources[i]
        dst = f_dests[i]

        for r in range(4):
            res_val = f_resources[i, r]
            demand_val = f_demands[i, r]
            plane_cap_val = f_plane_caps[i, r]

            # Direct Costs
            # cost += load_cost * res + unload_cost * res + fuel * dist * res * weight
            total_cost += (a_load_cost[dst, r] * res_val) + \
                          (a_unload_cost[src, r] * res_val) + \
                          (fuel * dist * res_val * WEIGHTS[r])

            # Penalties
            # Unsatisfied Demand
            if demand_val > res_val:
                total_cost += PLANE_UNSATISFY * dist * (demand_val - res_val)

            # Plane Overcapacity
            if res_val > plane_cap_val:
                total_cost += PLANE_OVERCAPACITY * dist * (res_val - plane_cap_val)

            # Update State (Virtual Simulation)
            state_buffer[t, src, r] -= res_val
            state_buffer[t, dst, r] += res_val

    # 2. Calculate Airport State Penalties
    # We iterate only over relevant timestamps and airports logic roughly
    # In original, it iterates over all airports touched. Here we iterate all (fast in C/Numba)
    # or we can use a boolean mask if performance is tight, but 161 is small.

    # Note: State shape is (720, 161, 4)
    # We only check timestamp 0 because all dummy flights are at t=0
    # To be generic like original:

    for t in range(1):  # Optimization: Only checking t=0 as per your example setup
        for a in range(161):
            for r in range(4):
                val = state_buffer[t, a, r]

                # Airport Zero (Stock depleted below zero)
                if val < 0:
                    total_cost += AIRPORT_ZERO * (-val)  # val is negative, so -val is positive

                # Airport Mucho (Overcapacity)
                if val > a_caps[a, r]:
                    total_cost += AIRPORT_MUCHO * (val - a_caps[a, r])

    return total_cost


# --- 2. The Numba Optimized Logic ---

@njit(cache=True, nogil=True)
def fast_gen_neighbor_gradient_hybrid(
        current_resources, cost_func, state_buffer,  # Added cost_func and state_buffer
        f_timestamps, f_dists, f_sources, f_dests, f_demands, f_plane_fuel, f_plane_caps,
        a_caps, a_load_cost, a_unload_cost,  # All static cost args
        delta, offset, gradient_influence, exploration_prob
):
    """
    Creates a perturbed copy of the resources array using a hybrid strategy.

    gradient_influence (float): Strength of the gradient-biased move.
    exploration_prob (float): Probability of choosing the purely random (SA) move.
    """

    n_flights, n_resources = current_resources.shape
    new_resources = current_resources.copy()

    # Probability Check: Should we do a random move (exploration) or a biased move (exploitation)?
    if np.random.rand() < exploration_prob:
        # PURE RANDOM SA MOVE (Exploration)
        for i in range(n_flights):
            for r in range(n_resources):
                val = new_resources[i, r]
                # Same original formula: scale * uniform(1-delta, 1+delta) + offset * uniform(-1, 1)
                val = val * np.random.uniform(1.0 - delta, 1.0 + delta) + \
                      np.random.uniform(-offset, offset)
                new_resources[i, r] = np.maximum(0.0, val)  # Ensure resource is non-negative
        return new_resources

    else:
        # GRADIENT-BIASED MOVE (Exploitation)

        # Step 1: Calculate the Pseudogradient for a randomly selected resource/flight
        # We only estimate the gradient for one random dimension per move for efficiency.
        f_idx = np.random.randint(0, n_flights)
        r_idx = np.random.randint(0, n_resources)

        epsilon = 1e-4  # Small perturbation size

        # Cost at current point
        cost_x = cost_func(
            current_resources, f_timestamps, f_dists, f_sources, f_dests, f_demands,
            f_plane_fuel, f_plane_caps, a_caps, a_load_cost, a_unload_cost, state_buffer
        )

        # Perturb the resource value
        temp_resources = current_resources.copy()
        temp_resources[f_idx, r_idx] += epsilon

        # Cost at perturbed point
        cost_x_plus_eps = cost_func(
            temp_resources, f_timestamps, f_dists, f_sources, f_dests, f_demands,
            f_plane_fuel, f_plane_caps, a_caps, a_load_cost, a_unload_cost, state_buffer
        )

        # Pseudogradient component (i.e., partial derivative approximation)
        g_component = (cost_x_plus_eps - cost_x) / epsilon

        # Step 2: Update the single chosen resource using the gradient direction

        # Move against the gradient: new_res = old_res - step_size * gradient_component
        # The strength of the move is controlled by 'gradient_influence'
        move = -gradient_influence * g_component

        new_resources[f_idx, r_idx] += move

        # Step 3: Add a small random perturbation to maintain local exploration
        new_resources[f_idx, r_idx] += np.random.uniform(-offset * 0.1, offset * 0.1)

        # Ensure non-negative
        new_resources[f_idx, r_idx] = np.maximum(0.0, new_resources[f_idx, r_idx])

        return new_resources

# --- 3. Simulated Annealing Logic ---

def run_simulated_annealing_numba(params):
    # Unpack params
    initial_temp = params['initial_temp']
    min_temp = params['min_temp']
    cooling_rate = params['cooling_rate']
    markov_len = params['markov_chain_length']
    neigh_delta = params['neighbour_delta']
    neigh_offset = params['neighbour_offset']
    neigh_selector = params['neighbour_selector']

    # 1. Prepare Static Data (Done once per SA run)
    (f_timestamps, f_dists, f_sources, f_dests, f_demands, f_resources,
     f_plane_fuel, f_plane_caps, a_caps, a_load_cost, a_unload_cost) = pack_data(
        INITIAL_FLIGHT_OBJS, AIRPORTS)

    # Pre-allocate State Buffer (Global size)
    state_buffer = np.zeros((720, 161, 4), dtype=np.float64)

    # 2. Initialization
    current_solution = f_resources  # Numpy array
    current_cost = fast_cost_function(
        current_solution, f_timestamps, f_dists, f_sources, f_dests, f_demands,
        f_plane_fuel, f_plane_caps, a_caps, a_load_cost, a_unload_cost, state_buffer
    )

    best_solution = current_solution.copy()
    best_cost = current_cost

    current_temp = initial_temp

    # 3. Annealing Loop
    while current_temp > min_temp:
        # Dynamically adjust the probability of choosing the random move
        # exploration_prob will be 1 at T=T_initial, and min_prob at T=min_temp (roughly)

        # Let's use a simple linear decay for the probability based on normalized temperature
        # You can tune 'min_exploration_prob' via Optuna
        normalized_temp = (current_temp - min_temp) / (initial_temp - min_temp)
        # Prob = T_norm * (1 - min_prob) + min_prob

        exploration_prob = normalized_temp * (1.0 - params['min_exploration_prob']) + params[
            'min_exploration_prob']

        for _ in range(markov_len):
            # Generate Neighbor (Hybrid Logic)
            new_solution = fast_gen_neighbor_gradient_hybrid(
                current_solution, fast_cost_function, state_buffer,
                f_timestamps, f_dists, f_sources, f_dests, f_demands,
                f_plane_fuel, f_plane_caps, a_caps, a_load_cost, a_unload_cost,
                neigh_delta, neigh_offset, params['gradient_influence'], exploration_prob
            )

            # Calculate Cost (JIT Compiled)
            new_cost = fast_cost_function(
                new_solution, f_timestamps, f_dists, f_sources, f_dests, f_demands,
                f_plane_fuel, f_plane_caps, a_caps, a_load_cost, a_unload_cost, state_buffer
            )

            # Acceptance Probability
            if new_cost < current_cost:
                accept = True
            else:
                p = math.exp((current_cost - new_cost) / current_temp)
                accept = random.random() < p

            if accept:
                current_solution = new_solution
                current_cost = new_cost

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = current_solution.copy()  # Important to copy!

        current_temp *= cooling_rate

    return best_cost


# --- 4. Optuna Integration ---

def objective(trial):
    params = {
        "initial_temp": trial.suggest_float("initial_temp", 100, 5000),  # Increased range
        "cooling_rate": trial.suggest_float("cooling_rate", 0.9, 0.999),  # Increased upper bound
        "markov_chain_length": trial.suggest_int("markov_chain_length", 10, 100),
        # Increased length
        "min_temp": 0.1,
        "neighbour_delta": trial.suggest_float("neighbour_delta", 0.001, 0.1),
        # Decreased range for delta
        "neighbour_offset": trial.suggest_float("neighbour_offset", 0.01, 1.0),
        # Decreased range for offset
        "gradient_influence": trial.suggest_float("gradient_influence", 0.01, 1.0),  # NEW PARAM
        "min_exploration_prob": trial.suggest_float("min_exploration_prob", 0.05, 0.5),  # NEW PARAM
        "neighbour_selector": 1  # Not used now, but kept for compatibility
    }

    return run_simulated_annealing_numba(params)

# --- 5. Execution ---
if __name__ == "__main__":
    start = time.time()
    params = {'initial_temp': 100, 'min_temp': 0.1, 'neighbour_selector':0.5,'cooling_rate': 0.9, 'markov_chain_length': 40, 'neighbour_delta': 0.15, 'neighbour_offset': 0.75, 'gradient_influence': 0.12801302850604718, 'min_exploration_prob': 0.1871634169779139}

    run_simulated_annealing_numba(params)
    print(time.time() - start)
