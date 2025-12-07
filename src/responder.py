"""
Minimal helpers for calling the Rotables API from Python.

This is testing-only: three functions, a global session id, and stubbed main.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, List
import requests
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
USE_PAYLOAD_OPTIMIZER = True  # set False to fall back to reservation-based loading
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
DEFAULT_PLAY_PAYLOAD = {
    "day": 0,
    "hour": 0,  # first round in a fresh session starts at day 0, hour 0
    "flightLoads": [],
    "kitPurchasingOrders": {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0},
}


def _default_costs() -> Dict[str, float]:
    return {"first": 0.0, "business": 0.0, "premiumEconomy": 0.0, "economy": 0.0}


def _build_forecast_state(airport_codes: List[str], day: int, hour: int, horizon: int = 72) -> Dict[int, Dict[str, List[int]]]:
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
        flights_list = [f for f in flight.flights.values() if f.current_status in ("SCHEDULED",)]
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
                capacity=airport_tracker.capacity.get(code, {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}),
                stock=airport_tracker.inventory.get(code, {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}),
                load_cost=airport_tracker.loading_cost.get(code, _default_costs()),
                unload_cost=airport_tracker.loading_cost.get(code, _default_costs()),
            )
        # Planes lookup
        plane_objs: Dict[str, OptimPlane] = {}
        for p in AIRCRAFT_TYPES.values():
            plane_objs[p.id] = OptimPlane(id=p.id, capacity=p.kit_capacity, fuel_cost=p.cost_per_kg_per_km)
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
        forecast_state = _build_forecast_state(list(airport_codes), day, hour, horizon=72)
        _, payloads = payload_optimizer.optimize(
            opt_flights,
            list(optim_airports.values()),
            forecast_state,
            cost_simulator=cost_simulator,
            response_json=airport_predictor._last_response,
        )
        for fli in flights_list:
            if fli.id in payloads:
                fli.loaded_packs = payloads[fli.id]
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


def start_session(api_key: str, base_url: str = DEFAULT_BASE_URL, output: str = "start_session_response.json") -> str:
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


def play_round(api_key: str, payload: Dict[str, Any], base_url: str = DEFAULT_BASE_URL, output: str = "play_round_response.json") -> Dict[str, Any]:
    """Call play/round with the given JSON payload."""
    if not SESSION_ID:
        raise RuntimeError("No SESSION_ID set. Call start_session first.")
    headers = {"API-KEY": api_key, "SESSION-ID": SESSION_ID, "Content-Type": "application/json"}
    
    resp = requests.post(f"{base_url}/play/round", headers=headers, json=payload)
    _raise_for_status(resp)
    body = resp.json()
    _save_json(body, output)
    return body


def end_session(api_key: str, base_url: str = DEFAULT_BASE_URL, output: str = "end_session_response.json") -> Dict[str, Any]:
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
    # print(f"Default payload: {json.dumps(DEFAULT_PLAY_PAYLOAD)}")
    # allocate payloads either via heuristic optimizer or reservation-based allocator
    if USE_PAYLOAD_OPTIMIZER:
        gen_cnt, applied_cnt = apply_payload_optimizer(day, hour)
        print(f"[optimizer] generated {gen_cnt} candidate payloads, applied {applied_cnt}")
    else:
        airport_tracker.allocate_for_flights(flight.flights.values())
    # Basic purchase: one-time 10k economy on day 0 hour 0, else no buys
    purchase_order = {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}
    if day == 0 and hour == 0:
        try:
            cap = airport_tracker.capacity.get("HUB1", {}).get("economy", 0)
            cur = airport_tracker.inventory["HUB1"].get("economy", 0)
            pending = airport_tracker.pending_arrivals["HUB1"].get("economy", 0)
            room = max(0, cap - (cur + pending))
            purchase_order["economy"] = min(10000, room)
        except Exception:
            purchase_order["economy"] = 0
    # Schedule purchase arrivals (respect lead times)
    airport_tracker.schedule_purchase("HUB1", purchase_order, day, hour)
    # Always include zero-load entries so the API does not auto-load defaults
    payload_flight_data = flight.to_play_round_updates(include_zero_loads=True)
    payload = {
        "day": day,
        "hour": hour,
        "flightLoads": payload_flight_data,
        "kitPurchasingOrders": purchase_order,
    }
    # print payload
    # print(f"Playing round for day {day}, hour {hour} with {len(payload_flight_data)} flight loads.")
    # print(f"Payload: {json.dumps(payload, indent=2)}")
    
    
    # We no longer record airport loads here; plane_tracker will ingest payload directly
    try:
        plane_tracker.ingest_payload(payload)
    except Exception:
        pass

    resp = play_round(api_key, payload)
     
    
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
    day = 0
    hour = 0
    
    api_key = input(f"API key [{DEFAULT_API_KEY}]: ").strip() or DEFAULT_API_KEY
    print("Commands: start | play | stop | exit")
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
