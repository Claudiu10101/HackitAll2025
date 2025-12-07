from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from simulated_annealing import SimulatedAnnealing, SAAirport, SAPlane, SAFlight


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
    timestamp: int
    distance: float
    origin: OptimAirport
    destination: OptimAirport
    plane: OptimPlane
    demand: Dict[str, int]
    load: Dict[str, int]


class PayloadOptimizer:
    def __init__(self):
        self.sa_planes_map: Dict[str, SAPlane] = {}
        self.sa_airports_map: Dict[str, SAAirport] = {}
        self.airport_code_to_id: Dict[str, int] = {}
        self.next_airport_id = 0

    def optimize(
        self,
        flights: List[OptimFlight],
        airports: List[OptimAirport],
        forecast_state: Dict[int, Dict[str, List[int]]],
        cost_simulator: Any,
        response_json: Any,
        current_time: int = 0,
        hub_id: str = "HUB1",
    ) -> Tuple[float, Dict[str, Dict[str, int]], Dict[str, int]]:

        if not flights:
            return (
                0.0,
                {},
                {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0},
            )

        # Update/Create SAAirport objects and IDs
        for a in airports:
            if a.id not in self.airport_code_to_id:
                self.airport_code_to_id[a.id] = self.next_airport_id
                self.next_airport_id += 1

            idx = self.airport_code_to_id[a.id]

            # Prepare numpy arrays
            stock = np.array(
                [
                    a.stock.get(k, 0)
                    for k in ["first", "business", "premiumEconomy", "economy"]
                ]
            )

            if a.id in self.sa_airports_map:
                # Update existing
                sa_airport = self.sa_airports_map[a.id]
                sa_airport.stock = stock
            else:
                # Create new
                cap = np.array(
                    [
                        a.capacity.get(k, 0)
                        for k in ["first", "business", "premiumEconomy", "economy"]
                    ]
                )
                load_cost = np.array(
                    [
                        a.load_cost.get(k, 0)
                        for k in ["first", "business", "premiumEconomy", "economy"]
                    ]
                )
                processing_cost = np.array(
                    [
                        a.unload_cost.get(k, 0)
                        for k in ["first", "business", "premiumEconomy", "economy"]
                    ]
                )

                self.sa_airports_map[a.id] = SAAirport(
                    id=idx,
                    code=a.id,
                    cap=cap,
                    stock=stock,
                    load_cost=load_cost,
                    processing_cost=processing_cost,
                )

        # Create SAFlight objects
        sa_flights: List[SAFlight] = []

        # Determine the current time to identify future flights
        current_time = min((f.timestamp for f in flights), default=0)

        for f in flights:
            if f.plane.id not in self.sa_planes_map:
                self.sa_planes_map[f.plane.id] = SAPlane(
                    cap=np.array(
                        [
                            f.plane.capacity.get(k, 0)
                            for k in ["first", "business", "premiumEconomy", "economy"]
                        ]
                    ),
                    fuel=f.plane.fuel_cost,
                )

            if (
                f.origin.id not in self.airport_code_to_id
                or f.destination.id not in self.airport_code_to_id
            ):
                continue

            # Apply 50% fullness factor to future flights (beyond the near-term horizon)
            is_future_flight = f.timestamp > current_time
            fullness_factor = 0.9 if is_future_flight else 1.0

            sa_flights.append(
                SAFlight(
                    flight_id=f.flight_id,
                    timestamp=f.timestamp,
                    distance=f.distance,
                    source=self.sa_airports_map[f.origin.id],
                    destination=self.sa_airports_map[f.destination.id],
                    plane=self.sa_planes_map[f.plane.id],
                    demands=np.array(
                        [
                            int(f.demand.get(k, 0) * fullness_factor)
                            for k in ["first", "business", "premiumEconomy", "economy"]
                        ]
                    ),
                    resources=np.array(
                        [
                            f.load.get(k, 0)
                            for k in ["first", "business", "premiumEconomy", "economy"]
                        ]
                    ),
                )
            )

        if not forecast_state:
            max_time = 0
        else:
            max_time = max(forecast_state.keys())

        horizon = max_time + 48  # Add buffer
        num_airports = self.next_airport_id  # Use total known airports

        initial_state = np.zeros((horizon + 1, num_airports, 4))

        for t, airport_states in forecast_state.items():
            if t > horizon:
                continue
            for code, stock_list in airport_states.items():
                if code in self.airport_code_to_id:
                    idx = self.airport_code_to_id[code]
                    initial_state[t, idx] = np.array(stock_list)

        sa_airports_by_id = {
            airport.id: airport for airport in self.sa_airports_map.values()
        }

        sa = SimulatedAnnealing(num_airports=num_airports, horizon=horizon)

        # Run SA
        # Parameters from original.py
        t_start = 100.0
        t_min = 0.1
        alpha = 0.92
        m_len = 100
        delta = 0.5
        offsets = [5.0, 10.0, 6.0, 50.0]  # [first, business, premiumEconomy, economy]
        selector = 0.2

        # Parallel execution
        futures = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            for kit_idx in range(4):
                futures.append(
                    executor.submit(
                        sa.run_single,
                        kit_idx,
                        sa_flights,
                        initial_state,
                        sa_airports_by_id,
                        t_start,
                        t_min,
                        alpha,
                        m_len,
                        delta,
                        offsets[kit_idx],
                        selector,
                    )
                )

        results = [f.result() for f in futures]

        total_cost = sum(r[0] for r in results)

        # Combine results
        payloads: Dict[str, Dict[str, int]] = {}
        kit_names = ["first", "business", "premiumEconomy", "economy"]

        for i, f in enumerate(sa_flights):
            payloads[f.flight_id] = {}
            for k in range(4):
                val = results[k][1][i]
                payloads[f.flight_id][kit_names[k]] = int(val)

        # Decide on kit purchases for the hub
        purchase_orders = self._decide_kit_purchases(
            hub_id, airports, forecast_state, current_time, horizon
        )

        return total_cost, payloads, purchase_orders

    def _decide_kit_purchases(
        self,
        hub_id: str,
        airports: List[OptimAirport],
        forecast_state: Dict[int, Dict[str, List[int]]],
        current_time: int,
        horizon: int,
    ) -> Dict[str, int]:
        """
        Decide how many kits to purchase for each class based on:
        - Current and forecasted stock levels at the hub
        - Lead times for kit delivery (ORDER_TIME)
        - Cost of purchasing kits (ORDER_COST)
        - Storage capacity constraints
        """
        from simulated_annealing import ORDER_TIME, ORDER_COST

        kit_names = ["first", "business", "premiumEconomy", "economy"]
        purchase_order: Dict[str, int] = {
            "first": 0,
            "business": 0,
            "premiumEconomy": 0,
            "economy": 0,
        }

        # Find the hub airport
        hub_airport = None
        for airport in airports:
            if airport.id == hub_id:
                hub_airport = airport
                break

        if hub_airport is None:
            return purchase_order

        # Get hub capacity and current stock
        hub_capacity = np.array([hub_airport.capacity.get(k, 0) for k in kit_names])

        hub_stock = np.array([hub_airport.stock.get(k, 0) for k in kit_names])

        # Analyze forecast to find minimum stock levels over the planning horizon
        min_stock = hub_stock.copy()
        avg_stock = hub_stock.copy().astype(float)
        count = 1

        for t in range(
            current_time, min(current_time + horizon, max(forecast_state.keys()) + 1)
        ):
            if t in forecast_state and hub_id in forecast_state[t]:
                stock_at_t = np.array(forecast_state[t][hub_id])
                min_stock = np.minimum(min_stock, stock_at_t)
                avg_stock += stock_at_t
                count += 1

        if count > 1:
            avg_stock /= count

        # Decision logic for each kit type
        for k in range(4):
            lead_time = int(ORDER_TIME[k])
            # kit_cost = ORDER_COST[k]  # Reserved for future cost-benefit analysis
            capacity = int(hub_capacity[k])
            current = int(hub_stock[k])
            min_forecast = float(min_stock[k])
            avg_forecast = float(avg_stock[k])

            if capacity <= 0:
                continue

            # Debug logging for large capacity values
            # Calculate utilization thresholds
            utilization = current / capacity if capacity > 0 else 0
            min_utilization = min_forecast / capacity if capacity > 0 else 0
            avg_utilization = avg_forecast / capacity if capacity > 0 else 0

            # Order strategy:
            # 1. If current stock is below 15% of capacity, order aggressively
            # 2. If forecasted min drops below 10%, order to prevent stockout
            # 3. Target maintaining 30-40% capacity (minimal stock)
            # 4. Consider lead time - order earlier for longer lead times

            should_order = False
            target_level = 0.35  # Target 35% capacity (reduced from 70%)

            # Critical low stock - order immediately
            if utilization < 0.15:
                should_order = True
                target_level = 0.40
            # Forecasted stockout risk
            elif min_utilization < 0.10:
                should_order = True
                target_level = 0.35
            # Proactive ordering based on average forecast
            elif avg_utilization < 0.25 and lead_time > 12:
                should_order = True
                target_level = 0.30
            # Low stock with long lead time - order early
            elif utilization < 0.25 and lead_time >= 36:
                should_order = True
                target_level = 0.30

            if should_order:
                # Calculate order quantity
                target_qty = int(capacity * target_level)
                order_qty = max(0, target_qty - current)

                # Don't exceed capacity
                max_order = capacity - current
                order_qty = min(order_qty, max_order)

                # Apply safety limit: never order more than 10,000 kits at once
                order_qty = min(order_qty, 10000)

                # Minimum order quantity (avoid tiny orders)
                min_order_qty = int(
                    capacity * 0.05
                )  # At least 5% of capacity (reduced from 10%)
                if order_qty > 0 and order_qty < min_order_qty:
                    order_qty = min(min_order_qty, max_order)

                # Ensure we use native Python int, not numpy int
                purchase_order[kit_names[k]] = max(0, int(order_qty))

        # Ensure all values are native Python integers
        purchase_order = {
            k: int(v) if not (np.isnan(v) or np.isinf(v)) else 0
            for k, v in purchase_order.items()
        }

        return purchase_order
