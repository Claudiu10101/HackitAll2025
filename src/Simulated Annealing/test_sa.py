import numpy as np
from simulated_annealing import solve_numba_single


def test_solve_numba_single():
    n_flights = 10
    n_airports = 5
    horizon = 20

    initial_resources = np.zeros(n_flights, dtype=np.float64)
    demands = np.ones(n_flights, dtype=np.float64) * 10
    distances = np.ones(n_flights, dtype=np.float64) * 100
    timestamps = np.zeros(n_flights, dtype=np.int64)
    source_ids = np.zeros(n_flights, dtype=np.int64)
    dest_ids = np.ones(n_flights, dtype=np.int64)
    plane_caps = np.ones(n_flights, dtype=np.float64) * 100
    plane_fuels = np.ones(n_flights, dtype=np.float64) * 0.1

    airport_load_costs = np.ones(n_airports, dtype=np.float64)
    airport_unload_costs = np.ones(n_airports, dtype=np.float64)
    airport_caps = np.ones(n_airports, dtype=np.float64) * 1000

    initial_state = np.zeros((horizon, n_airports), dtype=np.float64)
    target_timestamp = 0
    weight = 1.0

    initial_temp = 100.0
    min_temp = 1.0
    cooling_rate = 0.9
    markov_len = 10
    neigh_delta = 0.1
    neigh_offset = 5.0
    neigh_selector = 0.5

    cost, resources = solve_numba_single(
        initial_resources,
        demands,
        distances,
        timestamps,
        source_ids,
        dest_ids,
        plane_caps,
        plane_fuels,
        airport_load_costs,
        airport_unload_costs,
        airport_caps,
        initial_state,
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

    print("Cost:", cost)
    print("Resources:", resources)


if __name__ == "__main__":
    test_solve_numba_single()
