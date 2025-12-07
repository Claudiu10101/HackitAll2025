"""
Minimal helpers for calling the Rotables API from Python.

This is testing-only: three functions, a global session id, and stubbed main.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, List
import requests
import numpy as np

# Seed at import time to initialize all random modules (including Numba)
_import_seed = round(time.time() * 1000000) % (2**32)
random.seed(_import_seed)
np.random.seed(_import_seed)

from flight import flight, AIRCRAFT_TYPES
from penalty_logger import PenaltyLogger
from plane_tracker import PlaneTracker
from airport import AirportTracker, load_initial_data
from airport_predictor import AirportPredictor
from payload_optimizer import (
    PayloadOptimizer,
    OptimAirport,
    OptimPlane,
    OptimFlight,
)

from cost_simulator import CostSimulator
from cost_logger import CostLogger

cost_logger = CostLogger(out_dir="../reports/costs")


flight = flight()
logger = PenaltyLogger()
plane_tracker = PlaneTracker()
(
    initial_inventory,
    processing_time,
    capacity,
    loading_cost_table,
    processing_cost_table,
) = load_initial_data()
airport_tracker = AirportTracker(
    initial_inventory=initial_inventory,
    processing_time=processing_time,
    capacity=capacity,
    loading_cost=loading_cost_table,
    processing_cost=processing_cost_table,
)
airport_predictor = AirportPredictor(airport_tracker, plane_tracker)
payload_optimizer = PayloadOptimizer()
PENALTY_FACTORS = {
    "FLIGHT_OVERLOAD_FACTOR_PER_DISTANCE": 5.0,
    "UNFULFILLED_KIT_FACTOR_PER_DISTANCE": 0.003,
}
# Kit weights/costs are not provided by the dataset; assume unit weight/cost per kit for now.
KIT_WEIGHT = {"first": 1.0, "business": 1.0, "premiumEconomy": 1.0, "economy": 1.0}
KIT_COST = {"first": 1.0, "business": 1.0, "premiumEconomy": 1.0, "economy": 1.0}

# Distance lookup seeded from known flights; fallback is 0 if missing.
FLIGHT_DISTANCES: Dict[str, float] = {}
cost_simulator = CostSimulator(
    loading_cost=loading_cost_table,
    processing_cost=processing_cost_table,
    flight_distances=FLIGHT_DISTANCES,
    penalty_factors=PENALTY_FACTORS,
    kit_weight=KIT_WEIGHT,
    kit_cost=KIT_COST,
)
DEBUG_AIRPORTS = ["HUB1"]


DEFAULT_BASE_URL = "http://127.0.0.1:8080/api/v1"
SESSION_ID: Optional[str] = None
DEFAULT_API_KEY = "7bcd6334-bc2e-4cbf-b9d4-61cb9e868869"
OPTIMIZER_PURCHASE_ORDERS: Dict[str, int] = {
    "first": 0,
    "business": 0,
    "premiumEconomy": 0,
    "economy": 0,
}
DEFAULT_PLAY_PAYLOAD = {
    "day": 0,
    "hour": 0,  # first round in a fresh session starts at day 0, hour 0
    "flightLoads": [],
    "kitPurchasingOrders": {
        "first": 0,
        "business": 0,
        "premiumEconomy": 0,
        "economy": 0,
    },
}


def _default_costs() -> Dict[str, float]:
    return {"first": 0.0, "business": 0.0, "premiumEconomy": 0.0, "economy": 0.0}


def _build_forecast_state(
    airport_codes: List[str], day: int, hour: int, horizon: int = 72
) -> Dict[int, Dict[str, List[int]]]:
    """
    Build a state map abs_hour -> airport_code -> [f,b,pe,e] using predictor forecasts.
    Includes the current snapshot at abs_now from airport_tracker.inventory.
    """
    abs_now = day * 24 + hour
    state: Dict[int, Dict[str, List[int]]] = {}
    # Current snapshot
    snap_now: Dict[str, List[int]] = {}
    for code in airport_codes:
        inv = airport_tracker.inventory.get(code, {})
        snap_now[code] = [
            int(inv.get("first", 0)),
            int(inv.get("business", 0)),
            int(inv.get("premiumEconomy", 0)),
            int(inv.get("economy", 0)),
        ]
    state[abs_now] = snap_now

    for code in airport_codes:
        try:
            forecast = airport_predictor.forecast(code, day, hour, horizon)
        except Exception:
            continue
        for slot in forecast:
            abs_h = int(slot["abs_hour"])
            totals = slot.get("total", {})
            entry = state.setdefault(abs_h, {})
            entry[code] = [
                int(totals.get("first", 0)),
                int(totals.get("business", 0)),
                int(totals.get("premiumEconomy", 0)),
                int(totals.get("economy", 0)),
            ]
    # Ensure every timestamp has every airport with at least zeroed vectors
    for abs_h, entry in list(state.items()):
        for code in airport_codes:
            if code not in entry:
                entry[code] = [0, 0, 0, 0]
    return state


def apply_payload_optimizer(day: int, hour: int) -> tuple[int, int]:
    """
    Build optimizer inputs from live trackers and set flight.loaded_packs accordingly.
    Falls back silently if anything is missing.
    """
    try:
        abs_now = day * 24 + hour
        flights_list = [
            f for f in flight.flights.values() if f.current_status in ("SCHEDULED",)
        ]
        if not flights_list:
            return 0, 0
        # Build airport map for all airports touched by these flights using real stock/capacity/costs
        airport_codes = set()
        for fli in flights_list:
            airport_codes.add(fli.origin)
            airport_codes.add(fli.destination)
        optim_airports: Dict[str, OptimAirport] = {}
        for code in airport_codes:
            optim_airports[code] = OptimAirport(
                id=code,
                capacity=airport_tracker.capacity.get(
                    code, {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}
                ),
                stock=airport_tracker.inventory.get(
                    code, {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}
                ),
                load_cost=airport_tracker.loading_cost.get(code, _default_costs()),
                unload_cost=airport_tracker.loading_cost.get(code, _default_costs()),
            )
        # Planes lookup
        plane_objs: Dict[str, OptimPlane] = {}
        for p in AIRCRAFT_TYPES.values():
            plane_objs[p.id] = OptimPlane(
                id=p.id, capacity=p.kit_capacity, fuel_cost=p.cost_per_kg_per_km
            )
        # Build optimizer flights
        opt_flights: List[OptimFlight] = []
        for fli in flights_list:
            if not fli.plane or fli.plane.id not in plane_objs:
                continue
            origin = optim_airports.get(fli.origin)
            dest = optim_airports.get(fli.destination)
            if not origin or not dest:
                continue
            demand = {
                "first": int(fli.real_passengers.get("first", 0)),
                "business": int(fli.real_passengers.get("business", 0)),
                "premiumEconomy": int(fli.real_passengers.get("premiumEconomy", 0)),
                "economy": int(fli.real_passengers.get("economy", 0)),
            }
            # Seed load with demand (can be tweaked by annealing)
            initial_load = dict(demand)
            distance = float(fli.distance or 0)
            if distance <= 0:
                continue  # skip if no distance info yet
            opt_flights.append(
                OptimFlight(
                    flight_id=fli.id,
                    timestamp=abs_now,
                    distance=distance,
                    origin=origin,
                    destination=dest,
                    plane=plane_objs[fli.plane.id],
                    demand=demand,
                    load=initial_load,
                )
            )
        if not opt_flights:
            return 0, 0
        # Refresh distance lookup for cost simulator from current flights
        for fli in flights_list:
            if fli.distance:
                FLIGHT_DISTANCES[fli.id] = float(fli.distance)
        forecast_state = _build_forecast_state(
            list(airport_codes), day, hour, horizon=72
        )
        current_time = abs_now
        t0 = time.time()
        # Reseed with round-specific value to ensure Numba JIT gets different random sequences
        round_seed = abs_now + random.randint(0, 1000000)
        np.random.seed(round_seed)
        random.seed(round_seed)
        _, payloads, purchase_orders = payload_optimizer.optimize(
            opt_flights,
            list(optim_airports.values()),
            forecast_state,
            cost_simulator=cost_simulator,
            response_json=airport_predictor._last_response,
            current_time=current_time,
            hub_id="HUB1",
        )
        dt = time.time() - t0
        print(f"[optimizer] optimization took {dt:.3f}s")

        # Apply flight payloads
        for fli in flights_list:
            if fli.id in payloads:
                fli.loaded_packs = payloads[fli.id]

        # Store purchase orders for use in play_round
        global OPTIMIZER_PURCHASE_ORDERS
        OPTIMIZER_PURCHASE_ORDERS = purchase_orders
        total_ordered = sum(purchase_orders.values())
        if total_ordered > 0:
            print(
                f"[optimizer] purchase orders: {purchase_orders} (total: {total_ordered})"
            )

        return len(opt_flights), len(payloads)
    except Exception as exc:
        print(f"Warning: payload optimizer failed: {exc}")
        return 0, 0


def _save_json(data: Any, file_path: str) -> None:
    Path(file_path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _raise_for_status(resp: requests.Response) -> None:
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        detail = resp.text
        raise RuntimeError(f"API error {resp.status_code}: {detail}") from exc


def start_session(
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    output: str = "start_session_response.json",
) -> str:
    """Start a session and store the session id globally."""
    global SESSION_ID
    # Reset penalty aggregation for a fresh session
    try:
        logger.reset()
    except Exception:
        pass
    resp = requests.post(f"{base_url}/session/start", headers={"API-KEY": api_key})
    _raise_for_status(resp)
    SESSION_ID = resp.text.strip().strip('"')
    _save_json({"sessionId": SESSION_ID}, output)
    return SESSION_ID


def play_round(
    api_key: str,
    payload: Dict[str, Any],
    base_url: str = DEFAULT_BASE_URL,
    output: str = "play_round_response.json",
) -> Dict[str, Any]:
    """Call play/round with the given JSON payload."""
    if not SESSION_ID:
        raise RuntimeError("No SESSION_ID set. Call start_session first.")
    headers = {
        "API-KEY": api_key,
        "SESSION-ID": SESSION_ID,
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(f"{base_url}/play/round", headers=headers, json=payload)
        _raise_for_status(resp)
    except Exception as e:
        # Dump the payload that caused the error to a file
        import json

        error_file = (
            f"error_request_day{payload.get('day')}_hour{payload.get('hour')}.json"
        )
        print(
            f"\n[ERROR] API request failed for day={payload.get('day')}, hour={payload.get('hour')}"
        )
        print(f"[ERROR] Request payload dumped to: {error_file}")
        with open(error_file, "w") as f:
            json.dump(payload, f, indent=2)
        raise

    body = resp.json()
    _save_json(body, output)
    return body


def end_session(
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    output: str = "end_session_response.json",
) -> Dict[str, Any]:
    """End the active session."""
    global SESSION_ID
    if not SESSION_ID:
        raise RuntimeError("No SESSION_ID set. Call start_session first.")
    headers = {"API-KEY": api_key}
    resp = requests.post(f"{base_url}/session/end", headers=headers)
    _raise_for_status(resp)
    body = resp.json()
    _save_json(body, output)
    try:
        logger.record(body)
        logger.write_reports()
    except Exception:
        pass
    SESSION_ID = None
    return body


def next_hour(day: int, hour: int) -> tuple[int, int]:
    hour += 1
    if hour >= 24:
        hour = 0
        day += 1
    return day, hour


def handle_play(api_key: str, day: int, hour: int) -> tuple[int, int]:
    """Handle one play command and return updated (day, hour)."""
    global OPTIMIZER_PURCHASE_ORDERS

    print(f"\n{'='*60}")
    print(f"[ROUND] Starting day={day}, hour={hour}")
    print(f"{'='*60}")

    # allocate payloads via heuristic optimizer
    gen_cnt, applied_cnt = apply_payload_optimizer(day, hour)
    print(f"[optimizer] generated {gen_cnt} candidate payloads, applied {applied_cnt}")

    # Fallback to reservation-based allocator if optimizer fails to apply to scheduled flights
    if applied_cnt == 0:
        scheduled = [
            f for f in flight.flights.values() if f.current_status == "SCHEDULED"
        ]
        if scheduled:
            print(
                f"[optimizer] fallback: allocating for {len(scheduled)} scheduled flights using heuristic"
            )
            airport_tracker.allocate_for_flights(flight.flights.values())

    # Use purchase orders from optimizer, with fallback to initial purchase on day 0
    purchase_order = OPTIMIZER_PURCHASE_ORDERS.copy()

    # Ensure all values are valid Python integers (not numpy types, not NaN/inf)
    for key in ["first", "business", "premiumEconomy", "economy"]:
        val = purchase_order.get(key, 0)
        try:
            # Convert to int, handle any invalid values
            purchase_order[key] = max(0, int(val))
        except (ValueError, TypeError):
            purchase_order[key] = 0

    # Additional safety check: cap all orders at 10,000 per kit type
    for key in ["first", "business", "premiumEconomy", "economy"]:
        if purchase_order[key] > 20000:
            print(f"[WARNING] Capping {key} order from {purchase_order[key]} to 20,000")
            purchase_order[key] = 20000

    # Fallback: one-time economy purchase on day 0 hour 0 if optimizer didn't decide
    if day == 0 and hour == 0 and sum(purchase_order.values()) == 0:
        try:
            cap = airport_tracker.capacity.get("HUB1", {}).get("economy", 0)
            cur = airport_tracker.inventory["HUB1"].get("economy", 0)
            pending = airport_tracker.pending_arrivals["HUB1"].get("economy", 0)
            room = max(0, cap - (cur + pending))
            purchase_order["economy"] = min(10000, room)
            print(
                f"[optimizer] fallback initial purchase: {purchase_order['economy']} economy kits"
            )
        except Exception:
            purchase_order["economy"] = 0
    # Reset optimizer purchase orders for next round
    OPTIMIZER_PURCHASE_ORDERS = {
        "first": 0,
        "business": 0,
        "premiumEconomy": 0,
        "economy": 0,
    }

    # --- Prevent NEGATIVE_INVENTORY by truncating allocations and purchases ---
    # Truncate purchase orders so that inventory cannot go below zero after arrival
    for kit_type in ["first", "business", "premiumEconomy", "economy"]:
        cur = airport_tracker.inventory["HUB1"].get(kit_type, 0)
        pending = airport_tracker.pending_arrivals["HUB1"].get(kit_type, 0)
        cap = airport_tracker.capacity.get("HUB1", {}).get(kit_type, 0)
        # Calculate max allowed purchase so inventory never goes negative
        # Assume all kits sent in flight loads are deducted separately
        min_allowed = 0
        max_allowed = max(0, cap - (cur + pending))
        if purchase_order[kit_type] < min_allowed:
            purchase_order[kit_type] = 0
        elif purchase_order[kit_type] > max_allowed:
            purchase_order[kit_type] = max_allowed

    def _per_unit_costs(fli, kit_type):
        distance = float(fli.distance or 0.0)
        plane_cost = float(getattr(fli.plane, "cost_per_kg_per_km", 0.0))
        carry_cost = distance * plane_cost * KIT_WEIGHT.get(kit_type, 1.0)
        process_cost = airport_tracker.processing_cost_table.get(
            fli.destination, {}
        ).get(kit_type, 0.0)
        score = 4 * carry_cost - process_cost
        return carry_cost, process_cost, score

    payload_flight_data = flight.to_play_round_updates(include_zero_loads=True)

    from collections import defaultdict

    payload_by_id = {str(item.get("flightId")): item for item in payload_flight_data}
    flights_by_origin = defaultdict(list)
    for fli in flight.flights.values():
        if fli.current_status == "SCHEDULED":
            flights_by_origin[fli.origin].append(fli)

    for origin, flights in flights_by_origin.items():
        inventory_origin = airport_tracker.inventory.get(origin)
        if not inventory_origin:
            continue
        for kit_type in ["first", "business", "premiumEconomy", "economy"]:
            available = max(0, inventory_origin.get(kit_type, 0))
            candidates = []
            for fli in flights:
                item = payload_by_id.get(fli.id)
                if not item:
                    continue
                kits = item.setdefault("loadedKits", {})
                demand = max(0, kits.get(kit_type, 0))
                kits[kit_type] = 0  # default to zero; will be set below if selected
                carry_cost, process_cost, score = _per_unit_costs(fli, kit_type)
                candidates.append(
                    {
                        "flight": fli,
                        "item": item,
                        "demand": demand,
                        "score": score,
                        "carry_cost": carry_cost,
                        "process_cost": process_cost,
                    }
                )

            candidates.sort(key=lambda c: c["score"], reverse=True)

            for cand in candidates:
                if available <= 0:
                    break
                if cand["score"] <= 0 or cand["demand"] <= 0:
                    continue  # cheaper to pay penalty or nothing requested
                alloc_value = min(cand["demand"], available)
                kits = cand["item"].setdefault("loadedKits", {})
                kits[kit_type] = max(0, alloc_value)
                available -= alloc_value

    total_kits_sent = 0
    for item in payload_flight_data:
        kits = item.get("loadedKits", {})
        total_kits_sent += sum(kits.values())
    print(f"[optimizer] total kits sent in this round: {total_kits_sent}")

    payload = {
        "day": day,
        "hour": hour,
        "flightLoads": payload_flight_data,
        "kitPurchasingOrders": purchase_order,
    }

    print(f"[debug] purchase_order values: {purchase_order}")

    # Capture the round time before it is advanced by the API response
    round_day = day
    round_hour = hour

    # --- Cost logging ---
    # Calculate cost of new stock purchased this round
    stock_cost = 0.0
    for k, v in purchase_order.items():
        stock_cost += v * KIT_COST.get(k, 1.0)

    # Schedule purchase arrivals (respect lead times)
    airport_tracker.schedule_purchase("HUB1", purchase_order, day, hour)

    # We no longer record airport loads here; plane_tracker will ingest payload directly
    try:
        plane_tracker.ingest_payload(payload)
    except Exception:
        pass

    resp = play_round(api_key, payload)

    # Calculate cost of flying planes and airport ops for this round using the response
    flight_cost = 0.0
    try:
        cost_breakdowns = cost_simulator.compute(resp, payload)
        for c in cost_breakdowns:
            flight_cost += c.get("totalCost", 0.0)
    except Exception as e:
        print(f"[cost_logger] Error computing flight costs: {e}")

    # Record costs for this day (using the day/hour at the time of the request)
    cost_logger.record_day(round_day, stock_cost, flight_cost)

    day = resp.get("day", day)
    hour = resp.get("hour", hour)
    logger.record(resp)
    logger.write_reports()
    flight.load_data_from_json(resp)
    try:
        plane_tracker.update(resp, payload)
        airport_tracker.update(resp, plane_tracker)
        airport_predictor.update(payload, resp)  # keep predictor in sync
        # Predictor kept in sync; heavy debug logging removed
        # airport_predictor.forecast("HUB1", day, hour)
    except Exception as exc:
        print(f"Warning: tracker update failed: {exc}")
    # print(plane_tracker.get_all())

    day, hour = next_hour(day, hour)
    # print(f"Played round. Total cost so far: {total_cost}")
    # print(f"Next expected request should use day {day}, hour {hour}")
    # print(f"Next expected request should use day {day}, hour {hour}")
    # print(f"Flight loads sent: {json.dumps(payload_flight_data, indent=2)}")
    return day, hour


def main() -> None:
    # Set random seed fresh for each run to ensure different randomization
    seed = round(time.time() * 1000000) % (
        2**32
    )  # Use microseconds for better granularity
    with open("seed.txt", "w") as f:
        f.write(str(seed))
    random.seed(seed)
    np.random.seed(seed)
    print(f"[INFO] Random seed set to: {seed}")

    day = 0
    hour = 0

    api_key = input(f"API key [{DEFAULT_API_KEY}]: ").strip() or DEFAULT_API_KEY
    print("Commands: start | play | stop | exit")

    try:
        # ...existing code...
        pass
    finally:
        # Write cost log at end
        cost_logger.write_csv()
    while True:
        cmd = input("> ").strip().lower()
        if not cmd:
            continue
        if cmd == "exit":
            break
        if cmd == "start":
            try:
                sid = start_session(api_key)
                print(f"Session started: {sid}")
            except Exception as exc:  # pragma: no cover - interactive convenience
                print(f"Error starting session: {exc}")
            continue

        if cmd == "p":
            try:
                day, hour = handle_play(api_key, day, hour)
            except Exception as exc:  # pragma: no cover
                print(f"Error playing round: {exc}")
            continue

        if cmd == "full":
            try:
                for _ in range(720):
                    day, hour = handle_play(api_key, day, hour)
            except Exception as exc:  # pragma: no cover
                print(f"Error playing full session: {exc}")
            continue

        if cmd == "stop":
            try:
                resp = end_session(api_key)
                print(json.dumps(resp, indent=2))
            except Exception as exc:  # pragma: no cover
                print(f"Error ending session: {exc}")
            continue
        print("Unknown command. Use start | play | stop | exit.")


if __name__ == "__main__":
    main()
