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
from cost_logger import CostLogger
from cost_simulator import CostSimulator
from plane_tracker import PlaneTracker
from airport import AirportTracker, load_initial_data
from airport_predictor import AirportPredictor


flight = flight()
logger = PenaltyLogger()
cost_logger = CostLogger()
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
airport_tracker.set_predictor(airport_predictor)
# Cost simulator (distance map empty initially; will still allow cost-based ordering)
penalty_factors = {
    "FLIGHT_OVERLOAD_FACTOR_PER_DISTANCE": 5.0,
    "UNFULFILLED_KIT_FACTOR_PER_DISTANCE": 0.003,
}
kit_weight = {"first": 1.0, "business": 1.0, "premiumEconomy": 1.0, "economy": 1.0}
kit_cost = {"first": 1.0, "business": 1.0, "premiumEconomy": 1.0, "economy": 1.0}
cost_simulator = CostSimulator(
    loading_cost=loading_cost_table,
    processing_cost=processing_cost_table,
    flight_distances={},  # response provides distance; empty map is fine
    penalty_factors=penalty_factors,
    kit_weight=kit_weight,
    kit_cost=kit_cost,
)
airport_tracker.set_cost_simulator(cost_simulator)
USE_PAYLOAD_OPTIMIZER = False  # optimizer disabled
DEBUG_AIRPORTS = ["HUB1"]
TRACK_AIRPORT = "ZJDI"


DEFAULT_BASE_URL = "http://127.0.0.1:8080/api/v1"
SESSION_ID: Optional[str] = None
DEFAULT_API_KEY = "7bcd6334-bc2e-4cbf-b9d4-61cb9e868869"
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

    resp = requests.post(f"{base_url}/play/round", headers=headers, json=payload)
    _raise_for_status(resp)
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
        cost_logger.record(body)
        cost_logger.write_reports()
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
    airport_tracker.allocate_for_flights(flight.flights.values())
    # Per-round purchase: if HUB1 36h min usable (with pending) dips below 5% cap, top up minimally toward 5% (per class).
    # Disabled after day 10.
    purchase_order = {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}
    hub = "HUB1"
    if day <= 10:
        try:
            hub_caps = airport_tracker.capacity.get(hub, {})
            bought_msgs = []
            fc = (
                airport_predictor.forecast(hub, day, hour, horizon_hours=9)
                if airport_predictor
                else []
            )
            slots = len(fc) if fc else 0
            for cls, cap in hub_caps.items():
                if cap <= 0:
                    continue
                # Average usable forecast over next horizon
                if slots > 0:
                    avg_usable = sum(slot["usable"].get(cls, 0) for slot in fc) / slots
                    min_usable = min(slot["usable"].get(cls, 0) for slot in fc)
                else:
                    avg_usable = airport_tracker.usable_inventory[hub].get(cls, 0)
                    min_usable = avg_usable
                usable = airport_tracker.usable_inventory[hub].get(cls, 0)
                pending = airport_tracker.pending_arrivals[hub].get(cls, 0)
                cur_total = airport_tracker.inventory[hub].get(cls, 0)
                cap_limit = int(cap * 1.5)
                # Pending arrivals are not in the predictor; distribute them for checks
                avg_with_pending = avg_usable + (
                    pending / slots if slots > 0 else pending
                )
                min_with_pending = (
                    min_usable + pending
                )  # conservative: pending may all arrive before the dip
                if min_with_pending < 0.05 * cap and (usable + pending) < 0.05 * cap:
                    target = int(0.05 * cap)
                    desired = max(0, target - int(min_with_pending))
                    # Also cap per-round purchase to 5% of capacity to avoid overspending
                    desired = min(desired, int(0.05 * cap))
                    room = max(0, cap_limit - (cur_total + pending))
                    qty = min(desired, room)
                    if qty > 0:
                        purchase_order[cls] = qty
                        short = {
                            "first": "F",
                            "business": "B",
                            "premiumEconomy": "P",
                            "economy": "E",
                        }.get(cls, cls)
                        bought_msgs.append(f"{short}:{qty}")
            if any(purchase_order.values()):
                if bought_msgs:
                    print(f"HUB buy: {' '.join(bought_msgs)}")
                airport_tracker.schedule_purchase(hub, purchase_order, day, hour)
        except Exception:
            purchase_order = {
                "first": 0,
                "business": 0,
                "premiumEconomy": 0,
                "economy": 0,
            }

    # Always include zero-load entries so the API does not auto-load defaults
    payload_flight_data = flight.to_play_round_updates(include_zero_loads=True)
    payload = {
        "day": day,
        "hour": hour,
        "flightLoads": payload_flight_data,
        "kitPurchasingOrders": purchase_order,
    }
    print("day,hour", day, hour)
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
    cost_logger.record(resp)
    cost_logger.write_reports()
    flight.load_data_from_json(resp)
    try:
        plane_tracker.update(resp, payload)
        airport_tracker.update(resp, plane_tracker)
        airport_predictor.update(payload, resp)  # keep predictor in sync
        # Per-round HUB1 inventory snapshot
        try:
            hub_total = airport_tracker.inventory.get("HUB1", {})
            hub_usable = airport_tracker.usable_inventory.get("HUB1", {})
            print(f"HUB1 total: {hub_total} | usable: {hub_usable}")
        except Exception:
            pass
    except Exception as exc:
        print(f"Warning: tracker update failed: {exc}")
    try:
        airport_tracker.print_hub_denied_stats()
    except Exception:
        pass
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
        if cmd == "hub":
            try:
                airport_tracker.print_airport("HUB1")
            except Exception as exc:  # pragma: no cover
                print(f"Warning: failed to print HUB1 inventory: {exc}")
            continue
        if cmd == "start":
            try:
                sid = start_session(api_key)
                print(f"Session started: {sid}")
                # Compute and print network economy percentage once at start
                cap_totals = airport_tracker._sum_dicts(*airport_tracker.capacity.values())  # type: ignore[attr-defined]
                inv_totals = airport_tracker._sum_dicts(*airport_tracker.inventory.values())  # type: ignore[attr-defined]
                if cap_totals.get("economy", 0):
                    pct = inv_totals.get("economy", 0) / cap_totals["economy"] * 100
                    print(
                        f"Network economy stock at start: {inv_totals.get('economy',0)}/{cap_totals['economy']} ({pct:.2f}%)"
                    )
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
