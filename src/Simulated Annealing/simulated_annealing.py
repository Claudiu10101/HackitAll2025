import math
import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from numba import njit

# Constants from original.py
NEGATIVE_INVENTORY = 534200.0
OVER_CAPACITY_STOCK = 777.0
PLANE_UNSATISFY = 0.006
PLANE_OVERCAPACITY = 5.0
WEIGHTS = np.array([5.0, 3.0, 2.5, 1.5])

ORDER_TIME = np.array([48.0, 36.0, 24.0, 12.0])
ORDER_COST = np.array([200.0, 150.0, 100.0, 50.0])


@njit(cache=True, nogil=True)
def acceptance_probability_jit(old_cost, new_cost, temperature):
    """JIT-compiled acceptance probability for simulated annealing."""
    if new_cost < old_cost:
        return 1.0
    return math.exp((old_cost - new_cost) / temperature)


@njit(cache=True, nogil=True)
def calculate_airport_penalties_single_jit(
    state, target_timestamp, airport_caps, n_airports
):
    """JIT-compiled calculation of airport-level penalties for single kit type.
    Checks entire time horizon for negative inventory penalties."""
    cost = 0.0

    horizon = state.shape[0]

    for t in range(horizon):
        for i in range(n_airports):
            current_stock = state[t, i]

            # Negative inventory penalty (across all timestamps)
            neg_inv = max(0.0, -current_stock)
            cost += NEGATIVE_INVENTORY * neg_inv

            # Over capacity penalty (at target_timestamp only)
            if t == target_timestamp:
                over_cap = max(0.0, current_stock - airport_caps[i])
                cost += OVER_CAPACITY_STOCK * over_cap

    return cost


@njit(cache=True, nogil=True)
def calculate_total_flight_costs_single_jit(
    resources,
    demands,
    distances,
    plane_caps,
    plane_fuels,
    source_ids,
    dest_ids,
    airport_load_costs,
    airport_processing_costs,
    n_flights,
    weight,
):
    """JIT-compiled calculation of total flight costs (1D resources, single kit type)."""
    cost = 0.0

    for i in range(n_flights):
        res = resources[i]
        dist = distances[i]
        src = source_ids[i]
        dst = dest_ids[i]

        # Cost breakdown
        loading = airport_load_costs[src] * res
        movement = plane_fuels[i] * dist * res * weight
        processing = airport_processing_costs[dst] * res
        cost += loading + movement + processing

        # Penalties
        unsatisfied = max(0.0, demands[i] - res)
        cost += PLANE_UNSATISFY * dist * unsatisfied

        overcap = max(0.0, res - plane_caps[i])
        cost += PLANE_OVERCAPACITY * dist * overcap

    return cost


@njit(cache=True, nogil=True)
def apply_flights_to_state_single_jit(
    state, resources, timestamps, source_ids, dest_ids, n_flights
):
    """JIT-compiled batch application of flights to state (1D resources, single kit type)."""
    for i in range(n_flights):
        t = timestamps[i]
        src = source_ids[i]
        dst = dest_ids[i]
        state[t, src] -= resources[i]
        state[t, dst] += resources[i]


@njit(cache=True, nogil=True)
def undo_flights_from_state_single_jit(
    state, resources, timestamps, source_ids, dest_ids, n_flights
):
    """JIT-compiled batch undo of flights from state (1D resources, single kit type)."""
    for i in range(n_flights):
        t = timestamps[i]
        src = source_ids[i]
        dst = dest_ids[i]
        state[t, src] += resources[i]
        state[t, dst] -= resources[i]


@dataclass
class SAAirport:
    id: int
    code: str
    cap: np.ndarray
    stock: np.ndarray
    load_cost: np.ndarray
    processing_cost: np.ndarray


@dataclass
class SAPlane:
    cap: np.ndarray
    fuel: float


class SAFlight:
    def __init__(
        self,
        flight_id: str,
        timestamp: int,
        distance: float,
        source: SAAirport,
        destination: SAAirport,
        plane: SAPlane,
        demands: np.ndarray,
        resources: np.ndarray,
    ):
        self.flight_id = flight_id
        self.timestamp = timestamp
        self.distance = distance
        self.source = source
        self.destination = destination
        self.plane = plane
        self.demands = demands
        self.resources = resources

    def calc_cost(self, state: np.ndarray) -> float:
        """
        Applies the flight to the airports states and returns the cost incurred during flights.
        Uses cost_simulator methodology: loading at origin, movement, processing at destination.
        state shape: (timestamps, num_airports, 4)
        """
        cost = 0.0
        # Vectorized calculation for the 4 resource types
        # resources: (4,)

        # Cost breakdown matching cost_simulator:
        # 1. Loading cost (at origin/source)
        loading = self.source.load_cost * self.resources
        # 2. Movement cost (fuel * distance * kit_weight)
        movement = self.plane.fuel * self.distance * self.resources * WEIGHTS
        # 3. Processing cost (at destination)
        processing = self.destination.processing_cost * self.resources

        cost += np.sum(loading + movement + processing)

        # Unsatisfied demand penalty
        unsatisfied = np.maximum(0, self.demands - self.resources)
        cost += np.sum(PLANE_UNSATISFY * self.distance * unsatisfied)

        # Overcapacity penalty (plane)
        overcap = np.maximum(0, self.resources - self.plane.cap)
        cost += np.sum(PLANE_OVERCAPACITY * self.distance * overcap)

        # Update state
        state[self.timestamp, self.source.id] -= self.resources
        state[self.timestamp, self.destination.id] += self.resources

        return cost

    def undo(self, state: np.ndarray):
        """Undo the flight from the state"""
        state[self.timestamp, self.source.id] += self.resources
        state[self.timestamp, self.destination.id] -= self.resources

    def gen_neighbour(self, delta: float, offset: float, selector: float) -> "SAFlight":
        """Generate a neighbour flight by perturbing the resources"""
        new_resources = self.resources.copy().astype(float)

        noise_mult = np.random.uniform(1 - delta, 1 + delta, size=4)
        noise_add = np.random.uniform(-offset, offset, size=4)
        new_resources = new_resources * noise_mult + noise_add

        new_resources = np.maximum(0, np.round(new_resources)).astype(int)

        return SAFlight(
            self.flight_id,
            self.timestamp,
            self.distance,
            self.source,
            self.destination,
            self.plane,
            self.demands,
            new_resources,
        )


@njit(cache=True, nogil=True)
def calculate_cost_numba_single(
    resources,  # (n_flights,)
    demands,  # (n_flights,)
    distances,  # (n_flights,)
    timestamps,  # (n_flights,)
    source_ids,  # (n_flights,)
    dest_ids,  # (n_flights,)
    plane_caps,  # (n_flights,)
    plane_fuels,  # (n_flights,)
    airport_load_costs,  # (n_airports,)
    airport_processing_costs,  # (n_airports,)
    airport_caps,  # (n_airports,)
    state,  # (horizon, n_airports) - modified in place!
    target_timestamp,
    weight,  # float
):
    cost = 0.0
    n_flights = len(resources)

    # Calculate all flight costs using optimized helper
    cost += calculate_total_flight_costs_single_jit(
        resources,
        demands,
        distances,
        plane_caps,
        plane_fuels,
        source_ids,
        dest_ids,
        airport_load_costs,
        airport_processing_costs,
        n_flights,
        weight,
    )

    # Apply flights to state
    apply_flights_to_state_single_jit(
        state, resources, timestamps, source_ids, dest_ids, n_flights
    )

    # Calculate airport penalties
    n_airports = state.shape[1]
    cost += calculate_airport_penalties_single_jit(
        state, target_timestamp, airport_caps, n_airports
    )

    # Undo state changes
    undo_flights_from_state_single_jit(
        state, resources, timestamps, source_ids, dest_ids, n_flights
    )

    return cost


@njit(cache=True, nogil=True)
def solve_numba_single(
    initial_resources,  # (n_flights,)
    demands,
    distances,
    timestamps,
    source_ids,
    dest_ids,
    plane_caps,
    plane_fuels,
    airport_load_costs,
    airport_processing_costs,
    airport_caps,
    initial_state,  # (horizon, n_airports)
    target_timestamp,
    weight,
    initial_temp,
    min_temp,
    cooling_rate,
    markov_len,
    neigh_delta,
    neigh_offset,
    neigh_selector,
):
    working_state = initial_state.copy()

    current_resources = initial_resources.copy()
    current_cost = calculate_cost_numba_single(
        current_resources,
        demands,
        distances,
        timestamps,
        source_ids,
        dest_ids,
        plane_caps,
        plane_fuels,
        airport_load_costs,
        airport_processing_costs,
        airport_caps,
        working_state,
        target_timestamp,
        weight,
    )

    best_resources = current_resources.copy()
    best_cost = current_cost

    current_temp = initial_temp

    n_flights = len(current_resources)

    while current_temp > min_temp:
        for _ in range(markov_len):
            new_resources = np.empty_like(current_resources)

            for i in range(n_flights):
                if np.random.random() > neigh_selector:
                    # Pseudo Gradient
                    grad = 0.0
                    val = current_resources[i]

                    # Transport and operation costs (increasing val increases these)
                    src = source_ids[i]
                    dst = dest_ids[i]

                    grad -= airport_load_costs[src]
                    grad -= plane_fuels[i] * distances[i] * weight
                    grad -= airport_processing_costs[dst]

                    # Demand
                    if val < demands[i]:
                        grad += PLANE_UNSATISFY * distances[i]

                    # Plane Cap
                    if val > plane_caps[i]:
                        grad -= PLANE_OVERCAPACITY * distances[i]

                    # Airport Stocks
                    stock_src = working_state[target_timestamp, src]
                    stock_dst = working_state[target_timestamp, dst]

                    if stock_src - val < 0:
                        grad -= NEGATIVE_INVENTORY

                    if stock_dst + val < 0:
                        grad += NEGATIVE_INVENTORY
                    elif stock_dst + val > airport_caps[dst]:
                        grad -= OVER_CAPACITY_STOCK

                    step = neigh_offset * random.random() * (1.0 if grad > 0 else -1.0)
                    if grad == 0:
                        step = 0

                    new_val = val + step
                    new_resources[i] = max(0, round(new_val))
                else:
                    val = current_resources[i]
                    noise_mult = np.random.uniform(1 - neigh_delta, 1 + neigh_delta)
                    noise_add = np.random.uniform(-neigh_offset, neigh_offset)
                    new_val = val * noise_mult + noise_add
                    new_resources[i] = max(0, round(new_val))

            new_cost = calculate_cost_numba_single(
                new_resources,
                demands,
                distances,
                timestamps,
                source_ids,
                dest_ids,
                plane_caps,
                plane_fuels,
                airport_load_costs,
                airport_processing_costs,
                airport_caps,
                working_state,
                target_timestamp,
                weight,
            )

            if (
                acceptance_probability_jit(current_cost, new_cost, current_temp)
                > np.random.random()
            ):
                current_resources = new_resources
                current_cost = new_cost

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_resources = current_resources.copy()

        current_temp *= cooling_rate

    return best_cost, best_resources


class SimulatedAnnealing:
    def __init__(self, num_airports: int, horizon: int):
        self.num_airports = num_airports
        self.horizon = horizon

    def run_single(
        self,
        kit_index: int,
        initial_flights: List[SAFlight],
        initial_state: np.ndarray,
        airports_map: Dict[int, SAAirport],
        initial_temp: float,
        min_temp: float,
        cooling_rate: float,
        markov_len: int,
        neigh_delta: float,
        neigh_offset: float,
        neigh_selector: float,
    ) -> Tuple[float, np.ndarray]:
        """
        Runs optimization for a single kit type (0, 1, 2, or 3).
        Returns (best_cost, best_resources_1d_array).
        """
        if not initial_flights:
            return 0.0, np.array([])

        n_flights = len(initial_flights)
        n_airports = self.num_airports

        # Prepare arrays for Numba (sliced for kit_index)
        initial_resources = np.zeros(n_flights, dtype=np.float64)
        demands = np.zeros(n_flights, dtype=np.float64)
        distances = np.zeros(n_flights, dtype=np.float64)
        timestamps = np.zeros(n_flights, dtype=np.int64)
        source_ids = np.zeros(n_flights, dtype=np.int64)
        dest_ids = np.zeros(n_flights, dtype=np.int64)
        plane_caps = np.zeros(n_flights, dtype=np.float64)
        plane_fuels = np.zeros(n_flights, dtype=np.float64)

        airport_load_costs = np.zeros(n_airports, dtype=np.float64)
        airport_processing_costs = np.zeros(n_airports, dtype=np.float64)
        airport_caps = np.zeros(n_airports, dtype=np.float64)

        for i, airport in airports_map.items():
            if i < n_airports:
                airport_load_costs[i] = airport.load_cost[kit_index]
                airport_processing_costs[i] = airport.processing_cost[kit_index]
                airport_caps[i] = airport.cap[kit_index]

        for i, f in enumerate(initial_flights):
            initial_resources[i] = f.resources[kit_index]
            demands[i] = f.demands[kit_index]
            distances[i] = f.distance
            timestamps[i] = f.timestamp
            source_ids[i] = f.source.id
            dest_ids[i] = f.destination.id
            plane_caps[i] = f.plane.cap[kit_index]
            plane_fuels[i] = f.plane.fuel

        target_timestamp = initial_flights[0].timestamp
        weight = WEIGHTS[kit_index]

        # Slice initial state: (horizon, n_airports, 4) -> (horizon, n_airports)
        initial_state_slice = initial_state[:, :, kit_index].astype(np.float64)

        best_cost, best_resources = solve_numba_single(
            initial_resources,
            demands,
            distances,
            timestamps,
            source_ids,
            dest_ids,
            plane_caps,
            plane_fuels,
            airport_load_costs,
            airport_processing_costs,
            airport_caps,
            initial_state_slice,
            target_timestamp,
            weight,
            initial_temp,
            min_temp,
            cooling_rate,
            markov_len,
            neigh_delta,
            neigh_offset,
            neigh_selector,
        )

        return best_cost, best_resources
