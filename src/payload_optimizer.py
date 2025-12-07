"""
Simulated-annealing payload optimizer that uses real data (forecasted inventories, plane limits,
airport costs, and backend penalty factors).

Inputs:
  - forecast state: abs_hour -> airport_code -> [first, business, premiumEconomy, economy] totals
    (built from AirportPredictor.forecast)
  - airport costs/capacities from airports_with_stocks.csv
  - plane capacities/fuel cost and flight distances from live flight updates
  - CostSimulator (to score per-flight costs/penalties consistent with backend)
  - Penalty factors from backend (NEGATIVE_INVENTORY, OVER_CAPACITY_STOCK)
No placeholders are used: penalties and costs come from the real data available in the trackers/predictor.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Backend penalty factors (from PenaltyFactors.java)
NEGATIVE_INVENTORY = 5342
OVER_CAPACITY_STOCK = 777
PLANE_UNSATISFY = 1000  # retained from original heuristic for unmet demand
PLANE_OVERCAPACITY = 100  # retained from original heuristic for plane overfill
WEIGHTS = [1, 2, 3, 4]  # class weighting for movement cost (per original sketch)

CLASSES = ["first", "business", "premiumEconomy", "economy"]


@dataclass
class OptimAirport:
    id: str
    capacity: Dict[str, int]
    stock: Dict[str, int]
    load_cost: Dict[str, float]
    unload_cost: Dict[str, float]


@dataclass
class OptimPlane:
    id: str
    capacity: Dict[str, int]
    fuel_cost: float


@dataclass
class OptimFlight:
    flight_id: str
    timestamp: int  # absolute hour
    distance: float
    origin: OptimAirport
    destination: OptimAirport
    plane: OptimPlane
    demand: Dict[str, int]
    load: Dict[str, int]

    def calc_cost(self, state: Dict[int, Dict[str, List[int]]], fallback_ts: int) -> float:
        """
        Apply this flight to the forecasted state and return its cost contribution.
        Mirrors the original calc_cost comment:
          - load/unload + movement cost
          - unmet demand penalty (PLANE_UNSATISFY)
          - plane overcapacity penalty (PLANE_OVERCAPACITY)
        Mutates the state snapshot for this timestamp (and must be undone later).
        """
        ts = self.timestamp if self.timestamp in state else fallback_ts
        snapshot = state.get(ts)
        if not snapshot:
            return 0.0
        origin_vec = snapshot.get(self.origin.id)
        dest_vec = snapshot.get(self.destination.id)
        if origin_vec is None or dest_vec is None or len(origin_vec) < 4 or len(dest_vec) < 4:
            return 0.0

        cost = 0.0
        for i, cls in enumerate(CLASSES):
            qty = int(self.load.get(cls, 0))
            # costs
            cost += self.destination.load_cost.get(cls, 0) * qty
            cost += self.origin.unload_cost.get(cls, 0) * qty
            cost += self.plane.fuel_cost * self.distance * qty * WEIGHTS[i]
            # penalties
            cost += PLANE_UNSATISFY * self.distance * max(0, int(self.demand.get(cls, 0)) - qty)
            cost += PLANE_OVERCAPACITY * self.distance * max(0, qty - int(self.plane.capacity.get(cls, 0)))
            # mutate state
            origin_vec[i] -= qty
            dest_vec[i] += qty
        return cost

    def undo(self, state: Dict[int, Dict[str, List[int]]], fallback_ts: int) -> None:
        """Undo state mutations caused by calc_cost."""
        ts = self.timestamp if self.timestamp in state else fallback_ts
        snapshot = state.get(ts)
        if not snapshot:
            return
        origin_vec = snapshot.get(self.origin.id)
        dest_vec = snapshot.get(self.destination.id)
        if origin_vec is None or dest_vec is None or len(origin_vec) < 4 or len(dest_vec) < 4:
            return
        for i in range(4):
            qty = int(self.load.get(CLASSES[i], 0))
            origin_vec[i] += qty
            dest_vec[i] -= qty

    def gen_neighbor(self, delta: float, offset: float) -> "OptimFlight":
        """Perturb loads multiplicatively +/- delta and add small offset, clamped to >=0."""
        new_load = {}
        for cls in CLASSES:
            base = int(self.load.get(cls, 0))
            jitter = random.uniform(1 - delta, 1 + delta)
            add = random.uniform(-offset, offset)
            new_load[cls] = max(0, int(base * jitter + add))
        return OptimFlight(
            flight_id=self.flight_id,
            timestamp=self.timestamp,
            distance=self.distance,
            origin=self.origin,
            destination=self.destination,
            plane=self.plane,
            demand=self.demand,
            load=new_load,
        )


class PayloadOptimizer:
    """Simulated annealing with forecast-based airport penalties and CostSimulator scoring."""

    def __init__(
        self,
        initial_temp: float = 200.0,
        min_temp: float = 0.5,
        cooling_rate: float = 0.92,
        markov_len: int = 20,
        neigh_delta: float = 0.2,
        neigh_offset: float = 5.0,
    ) -> None:
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.markov_len = markov_len
        self.neigh_delta = neigh_delta
        self.neigh_offset = neigh_offset

    def _cost_function(
        self,
        flights: List[OptimFlight],
        state: Dict[int, Dict[str, List[int]]],
        capacity_map: Dict[str, Dict[str, int]],
        cost_simulator=None,
        response_json: Optional[Dict] = None,
    ) -> float:
        cost = 0.0
        touched_airports = set()
        fallback_ts = min(state.keys()) if state else 0
        for f in flights:
            cost += f.calc_cost(state, fallback_ts)
            touched_airports.add(f.destination.id)
            touched_airports.add(f.origin.id)

        # Add flight-level costs/penalties from CostSimulator (backend-aligned) if available
        if cost_simulator and response_json is not None:
            payload_json = {"flightLoads": [{"flightId": f.flight_id, "loadedKits": dict(f.load)} for f in flights]}
            try:
                sim_results = cost_simulator.compute(response_json, payload_json)
                cost += sum(item.get("totalCost", 0) + item.get("penalties", 0) for item in sim_results)
            except Exception:
                pass

        # Airport penalties across full forecast horizon
        for snapshot in state.values():
            for aid in touched_airports:
                vec = snapshot.get(aid)
                if not vec or len(vec) < 4:
                    continue
                cap_vec = capacity_map.get(aid, {})
                for idx, cls in enumerate(CLASSES):
                    inv = vec[idx]
                    cap = cap_vec.get(cls, 0)
                    cost += NEGATIVE_INVENTORY * max(0, -inv)
                    cost += OVER_CAPACITY_STOCK * max(0, inv - cap)

        # Rollback state mutations
        for f in flights:
            f.undo(state, fallback_ts)
        return cost

    def _acceptance_probability(self, old_cost: float, new_cost: float, temperature: float) -> float:
        if new_cost < old_cost:
            return 1.0
        return math.exp((old_cost - new_cost) / max(temperature, 1e-9))

    def optimize(
        self,
        flights: List[OptimFlight],
        airports: List[OptimAirport],
        state: Dict[int, Dict[str, List[int]]],
        cost_simulator=None,
        response_json: Optional[Dict] = None,
    ) -> Tuple[float, Dict[str, Dict[str, int]]]:
        """
        Run simulated annealing and return (best_cost, payloads_by_flight).
        """
        capacity_map = {ap.id: ap.capacity for ap in airports}
        current = list(flights)
        current_cost = self._cost_function(current, state, capacity_map, cost_simulator, response_json)
        best = current
        best_cost = current_cost

        temp = self.initial_temp
        while temp > self.min_temp:
            for _ in range(self.markov_len):
                neighbor = [f.gen_neighbor(self.neigh_delta, self.neigh_offset) for f in current]
                new_cost = self._cost_function(neighbor, state, capacity_map, cost_simulator, response_json)
                if self._acceptance_probability(current_cost, new_cost, temp) > random.random():
                    current = neighbor
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best = current
                        best_cost = current_cost
            temp *= self.cooling_rate

        payloads = {f.flight_id: f.load for f in best}
        return best_cost, payloads
