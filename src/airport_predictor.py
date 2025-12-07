from __future__ import annotations

from copy import deepcopy
from typing import Dict, List


class AirportPredictor:
    """
    Predicts per-hour stock for one airport over a given horizon.
    Uses current airport tracker state (total/usable), pending cleaning/purchase events,
    and known flight schedules to produce both total and usable projections.
    Negatives are allowed to surface expected shortages.
    """

    def __init__(self, airport_tracker, plane_tracker) -> None:
        self.airport_tracker = airport_tracker
        self.plane_tracker = plane_tracker
        self._last_response = None
        self._last_request = None

    def update(self, request_json: Dict, response_json: Dict) -> None:
        """Store latest request/response for on-demand forecasting."""
        self._last_request = request_json
        self._last_response = response_json

    def _abs_hour(self, day: int, hour: int) -> int:
        return day * 24 + hour

    def _extract_flight_infos(self, now_abs: int, horizon_abs: int) -> List[Dict[str, object]]:
        """Extract and normalize flight data from the last response for use in forecasts."""
        flights_iter = self._last_response.get("flightUpdates", []) if self._last_response else []
        flights_by_id = {item["flightId"]: item for item in flights_iter}  # dedupe

        infos: List[Dict[str, object]] = []
        for fli in flights_by_id.values():
            dep_abs = self._abs_hour(fli["departure"]["day"], fli["departure"]["hour"])
            arr_abs = self._abs_hour(fli["arrival"]["day"], fli["arrival"]["hour"])
            # ignore flights completely outside horizon
            if dep_abs > horizon_abs and arr_abs > horizon_abs:
                continue
            info = {
                "id": fli["flightId"],
                "evt": (fli.get("eventType") or "").upper().replace(" ", "_"),
                "origin": fli["originAirport"],
                "dest": fli["destinationAirport"],
                "dep_abs": dep_abs,
                "arr_abs": arr_abs,
                "passengers": {
                    "first": int(fli.get("passengers", {}).get("first", 0)),
                    "business": int(fli.get("passengers", {}).get("business", 0)),
                    "premiumEconomy": int(fli.get("passengers", {}).get("premiumEconomy", 0)),
                    "economy": int(fli.get("passengers", {}).get("economy", 0)),
                },
                "passenger_loaded": {
                    "first": int(fli.get("passengers", {}).get("first", 0)),
                    "business": int(fli.get("passengers", {}).get("business", 0)),
                    "premiumEconomy": int(fli.get("passengers", {}).get("premiumEconomy", 0)),
                    "economy": int(fli.get("passengers", {}).get("economy", 0)),
                },
                "payload_loaded": {},
                "committed": {},
            }
            # payload loads if available
            if self._last_request:
                for item in self._last_request.get("flightLoads", []):
                    if item.get("flightId") == info["id"]:
                        info["payload_loaded"] = {
                            "first": int(item["loadedKits"].get("first", 0)),
                            "business": int(item["loadedKits"].get("business", 0)),
                            "premiumEconomy": int(item["loadedKits"].get("premiumEconomy", 0)),
                            "economy": int(item["loadedKits"].get("economy", 0)),
                        }
                        break
            # committed loads from plane tracker
            if self.plane_tracker:
                info["committed"] = self.plane_tracker.get_inventory(info["id"])
            infos.append(info)
        return infos

    def _load_for(self, f) -> Dict[str, int]:
        """Prefer committed loaded_packs; fallback to passenger counts."""
        loaded = {cls: int(f.loaded_packs.get(cls, 0)) for cls in ("first", "business", "premiumEconomy", "economy")}
        if any(loaded.values()):
            return loaded
        return {
            "first": int(f.real_passengers.get("first", 0)),
            "business": int(f.real_passengers.get("business", 0)),
            "premiumEconomy": int(f.real_passengers.get("premiumEconomy", 0)),
            "economy": int(f.real_passengers.get("economy", 0)),
        }

    def simple_forecast(
        self,
        airport_code: str,
        current_day: int,
        current_hour: int,
        horizon_hours: int = 24,
    ) -> List[Dict[str, object]]:
        now_abs = self._abs_hour(current_day, current_hour)
        horizon_abs = now_abs + horizon_hours

        # Starting points
        total = deepcopy(self.airport_tracker.inventory.get(airport_code, {}))
        usable = deepcopy(self.airport_tracker.usable_inventory.get(airport_code, {}))

        total_events: Dict[int, Dict[str, int]] = {}
        usable_events: Dict[int, Dict[str, int]] = {}

        def add_event(bucket: Dict[int, Dict[str, int]], when: int, delta: Dict[str, int]) -> None:
            if when <= now_abs or when > horizon_abs:
                return
            cur = bucket.setdefault(when, {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0})
            for cls, val in delta.items():
                cur[cls] = cur.get(cls, 0) + int(val)

        # Pending cleaning/purchase events already scheduled in tracker (usable only)
        for when, delta in self.airport_tracker._events.get(airport_code, []):  # type: ignore[attr-defined]
            add_event(usable_events, when, delta)

        # Flight-based events using extracted flight info
        flight_infos = self._extract_flight_infos(now_abs, horizon_abs)
        for info in flight_infos:
            if info["evt"] == "LANDED":
                continue
            dep_abs = info["dep_abs"]
            arr_abs = info["arr_abs"]
            origin = info["origin"]
            dest = info["dest"]
            passengers = info["passengers"]
            loaded = info["passenger_loaded"]

            # Departure: subtract load at origin if in future horizon and flight not yet departed
            if origin == airport_code and dep_abs > now_abs and info["evt"] in ("SCHEDULED", "CHECKED_IN"):
                add_event(total_events, dep_abs, {cls: -val for cls, val in loaded.items()})
                add_event(usable_events, dep_abs, {cls: -val for cls, val in loaded.items()})
            # Arrival: add load at destination if in future horizon and flight not yet landed
            if dest == airport_code and arr_abs > now_abs and info["evt"] in ("SCHEDULED", "CHECKED_IN"):
                print(f"[Predictor] Incoming flight {info['id']} to {airport_code} at abs {arr_abs} economy load (pax-based): {loaded.get('economy',0)}")
                add_event(total_events, arr_abs, loaded)
                unused = {cls: max(0, loaded.get(cls, 0) - passengers.get(cls, 0)) for cls in loaded}
                used = {cls: min(loaded.get(cls, 0), passengers.get(cls, 0)) for cls in loaded}
                # Unused kits become usable immediately
                add_event(usable_events, arr_abs, unused)
                # Used kits become usable after processing time
                proc_time = self.airport_tracker.processing_time.get(dest, {})
                for cls, val in used.items():
                    if val <= 0:
                        continue
                    ready_abs = arr_abs + int(proc_time.get(cls, 0))
                    add_event(usable_events, ready_abs, {cls: val})

        # Build timeline hour by hour applying events (negatives allowed)
        timeline: List[Dict[str, object]] = []
        for abs_h in range(now_abs + 1, horizon_abs + 1):
            if abs_h in total_events:
                for cls, val in total_events[abs_h].items():
                    total[cls] = total.get(cls, 0) + val
            if abs_h in usable_events:
                for cls, val in usable_events[abs_h].items():
                    usable[cls] = usable.get(cls, 0) + val
            timeline.append(
                {
                    "abs_hour": abs_h,
                    "day": abs_h // 24,
                    "hour": abs_h % 24,
                    "total": deepcopy(total),
                    "usable": deepcopy(usable),
                }
            )
        return timeline

    def forecast(
        self,
        airport_code: str,
        current_day: int,
        current_hour: int,
        horizon_hours: int = 24,
    ) -> List[Dict[str, object]]:
        """
        Full forecast including optimistic passenger-based loads (simple_forecast)
        plus committed loads from plane_tracker/payload when available.
        """
        now_abs = self._abs_hour(current_day, current_hour)
        horizon_abs = now_abs + horizon_hours

        # Start from simple forecast baseline
        base = self.simple_forecast(airport_code, current_day, current_hour, horizon_hours)

        # Apply committed loads from plane_tracker (if seen in last response and not landed)
        total_events: Dict[int, Dict[str, int]] = {}
        usable_events: Dict[int, Dict[str, int]] = {}

        def add_event(bucket: Dict[int, Dict[str, int]], when: int, delta: Dict[str, int]) -> None:
            if when <= now_abs or when > horizon_abs:
                return
            cur = bucket.setdefault(when, {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0})
            for cls, val in delta.items():
                cur[cls] = cur.get(cls, 0) + int(val)

        infos = self._extract_flight_infos(now_abs, horizon_abs)
        for info in infos:
            if info["evt"] == "LANDED":
                continue
            if info["origin"] != airport_code and info["dest"] != airport_code:
                continue
            loaded = info["committed"] or info["payload_loaded"] or info["passenger_loaded"]
            if not loaded:
                continue
            if info["origin"] == airport_code and info["dep_abs"] > now_abs and info["evt"] in ("SCHEDULED", "CHECKED_IN"):
                add_event(total_events, info["dep_abs"], {cls: -val for cls, val in loaded.items()})
                add_event(usable_events, info["dep_abs"], {cls: -val for cls, val in loaded.items()})
            if info["dest"] == airport_code and info["arr_abs"] > now_abs and info["evt"] in ("SCHEDULED", "CHECKED_IN"):
                add_event(total_events, info["arr_abs"], loaded)
                add_event(usable_events, info["arr_abs"], loaded)

        # Apply committed deltas over the baseline timeline
        timeline: List[Dict[str, object]] = []
        for slot in base:
            abs_h = slot["abs_hour"]
            total = deepcopy(slot["total"])
            usable = deepcopy(slot["usable"])
            if abs_h in total_events:
                for cls, val in total_events[abs_h].items():
                    total[cls] = total.get(cls, 0) + val
            if abs_h in usable_events:
                for cls, val in usable_events[abs_h].items():
                    usable[cls] = usable.get(cls, 0) + val
            timeline.append(
                {"abs_hour": abs_h, "day": slot["day"], "hour": slot["hour"], "total": total, "usable": usable}
            )
        return timeline

    def forecast_with_planes(
        self,
        airport_code: str,
        current_day: int,
        current_hour: int,
        horizon_hours: int = 24,
    ) -> List[Dict[str, object]]:
        """
        Forecast that starts from simple_forecast and then applies what planes are carrying.
        Extraction points (you can tweak these):
          - committed = plane_tracker.get_inventory(fid)  # extracted from plane tracker
          - payload_loads[...]                             # extracted from last payload
          - passengers[...]                                # extracted from last response (passenger counts)
        You can modify the `loaded` dict inside this function to simulate different payloads.
        """
        now_abs = self._abs_hour(current_day, current_hour)
        horizon_abs = now_abs + horizon_hours

        base = self.simple_forecast(airport_code, current_day, current_hour, horizon_hours)

        # Build lookup for payload loads
        payload_loads: Dict[str, Dict[str, int]] = {}
        if self._last_request:
            for item in self._last_request.get("flightLoads", []):
                payload_loads[item["flightId"]] = {
                    "first": int(item["loadedKits"].get("first", 0)),
                    "business": int(item["loadedKits"].get("business", 0)),
                    "premiumEconomy": int(item["loadedKits"].get("premiumEconomy", 0)),
                    "economy": int(item["loadedKits"].get("economy", 0)),
                }

        total_events: Dict[int, Dict[str, int]] = {}
        usable_events: Dict[int, Dict[str, int]] = {}

        def add_event(bucket: Dict[int, Dict[str, int]], when: int, delta: Dict[str, int]) -> None:
            if when <= now_abs or when > horizon_abs:
                return
            cur = bucket.setdefault(when, {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0})
            for cls, val in delta.items():
                cur[cls] = cur.get(cls, 0) + int(val)

        infos = self._extract_flight_infos(now_abs, horizon_abs)
        for info in infos:
            if info["evt"] == "LANDED":
                continue
            if info["origin"] != airport_code and info["dest"] != airport_code:
                continue

            committed = self.plane_tracker.get_inventory(info["id"]) if self.plane_tracker else {}
            payload_loaded = payload_loads.get(info["id"], {})
            passenger_loaded = info["passenger_loaded"]

            loaded = committed if any(committed.values()) else payload_loaded if any(payload_loaded.values()) else passenger_loaded

            if info["origin"] == airport_code and info["dep_abs"] > now_abs and info["evt"] in ("SCHEDULED", "CHECKED_IN"):
                add_event(total_events, info["dep_abs"], {cls: -val for cls, val in loaded.items()})
                add_event(usable_events, info["dep_abs"], {cls: -val for cls, val in loaded.items()})
            if info["dest"] == airport_code and info["arr_abs"] > now_abs and info["evt"] in ("SCHEDULED", "CHECKED_IN"):
                add_event(total_events, info["arr_abs"], loaded)
                unused = {cls: max(0, loaded.get(cls, 0) - passenger_loaded.get(cls, 0)) for cls in loaded}
                used = {cls: min(loaded.get(cls, 0), passenger_loaded.get(cls, 0)) for cls in loaded}
                add_event(usable_events, info["arr_abs"], unused)
                proc_time = self.airport_tracker.processing_time.get(info["dest"], {})
                for cls, val in used.items():
                    if val <= 0:
                        continue
                    ready_abs = info["arr_abs"] + int(proc_time.get(cls, 0))
                    add_event(usable_events, ready_abs, {cls: val})

        timeline: List[Dict[str, object]] = []
        for slot in base:
            abs_h = slot["abs_hour"]
            total = deepcopy(slot["total"])
            usable = deepcopy(slot["usable"])
            if abs_h in total_events:
                for cls, val in total_events[abs_h].items():
                    total[cls] = total.get(cls, 0) + val
            if abs_h in usable_events:
                for cls, val in usable_events[abs_h].items():
                    usable[cls] = usable.get(cls, 0) + val
            timeline.append(
                {"abs_hour": abs_h, "day": slot["day"], "hour": slot["hour"], "total": total, "usable": usable}
            )
        return timeline

    def forecast_penalty(
        self,
        airport_code: str,
        current_day: int,
        current_hour: int,
        horizon_hours: int = 24,
        capacity: Dict[str, int] | None = None,
    ) -> Dict[str, int]:
        """
        Wrapper over forecast(): returns aggregate overstock and negative-stock amounts across the horizon.
        Over_stock is how much total exceeds capacity (per class); negative is how far below zero (per class).
        Also returns a rough monetary estimate using hard-coded penalty factors:
          - OVER_CAPACITY_STOCK = 777 per extra kit
          - NEGATIVE_INVENTORY  = 5342 per missing kit
        """
        timeline = self.forecast(airport_code, current_day, current_hour, horizon_hours)
        cap = capacity or self.airport_tracker.capacity.get(airport_code, {})
        over = {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}
        neg = {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}
        for slot in timeline:
            for cls, val in slot["total"].items():
                if val < 0:
                    neg[cls] += abs(val)
                if cls in cap and val > cap[cls]:
                    over[cls] += val - cap[cls]
        over_cap_factor = 777
        neg_inv_factor = 5342
        total_over = sum(over.values())
        total_neg = sum(neg.values())
        estimate_cost = total_over * over_cap_factor + total_neg * neg_inv_factor
        per_class_penalty = {
            cls: over[cls] * over_cap_factor + neg[cls] * neg_inv_factor for cls in over
        }
        return {
            "over_stock": over,                 # per-class over-capacity amounts aggregated over horizon
            "negative": neg,                    # per-class negative stock amounts aggregated over horizon
            "over_stock_total": total_over,     # sum of over-capacity across all classes
            "negative_total": total_neg,        # sum of negative stock across all classes
            "estimated_penalty": estimate_cost, # rough total penalty using hard-coded factors
            "per_class_penalty": per_class_penalty,  # per-class penalty using same factors
        }

    def forecast_with_custom_load(
        self,
        airport_code: str,
        flight_id: str,
        custom_loaded_kits: Dict[str, int],
        current_day: int,
        current_hour: int,
        horizon_hours: int = 24,
    ) -> List[Dict[str, object]]:
        """
        Forecast using a custom payload for one flight id (overrides any existing payload/committed load).
        Useful for simulating how a different load on a specific flight affects stock.
        """
        # Build a temporary payload overriding this flight
        payload_override = deepcopy(self._last_request) if self._last_request else {"flightLoads": []}
        # Remove any existing entry for this flight
        payload_override["flightLoads"] = [
            item for item in payload_override.get("flightLoads", []) if item.get("flightId") != flight_id
        ]
        # Add the custom load
        payload_override["flightLoads"].append(
            {
                "flightId": flight_id,
                "loadedKits": {
                    "first": int(custom_loaded_kits.get("first", 0)),
                    "business": int(custom_loaded_kits.get("business", 0)),
                    "premiumEconomy": int(custom_loaded_kits.get("premiumEconomy", 0)),
                    "economy": int(custom_loaded_kits.get("economy", 0)),
                },
            }
        )
        # Run the simulation with this payload override
        return self.simulate_with_payload(airport_code, payload_override, current_day, current_hour, horizon_hours)

    def penalty_from_forecast(self, forecast_slots: List[Dict[str, object]], capacity: Dict[str, int] | None = None) -> Dict[str, object]:
        """
        Given the forecast output (list of slots with total/usable), compute over/negative and estimated penalty.
        Mirrors forecast_penalty but works off a precomputed forecast.
        """
        cap = capacity or {}
        over = {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}
        neg = {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}
        for slot in forecast_slots:
            for cls, val in slot["total"].items():
                if val < 0:
                    neg[cls] += abs(val)
                if cls in cap and val > cap[cls]:
                    over[cls] += val - cap[cls]
        over_cap_factor = 777
        neg_inv_factor = 5342
        total_over = sum(over.values())
        total_neg = sum(neg.values())
        estimate_cost = total_over * over_cap_factor + total_neg * neg_inv_factor
        per_class_penalty = {
            cls: over[cls] * over_cap_factor + neg[cls] * neg_inv_factor for cls in over
        }
        return {
            "over_stock": over,
            "negative": neg,
            "over_stock_total": total_over,
            "negative_total": total_neg,
            "estimated_penalty": estimate_cost,
            "per_class_penalty": per_class_penalty,
        }

    def estimated_penalty_only(self, forecast_slots: List[Dict[str, object]], capacity: Dict[str, int] | None = None) -> float:
        """
        Convenience wrapper: given the output of any forecast* call, return only the estimated penalty number.
        Capacity defaults to the tracker capacity if not provided.
        """
        penalties = self.penalty_from_forecast(forecast_slots, capacity or {})
        return float(penalties.get("estimated_penalty", 0.0))

    def penalty_for_custom_load(
        self,
        airport_code: str,
        flight_id: str,
        custom_loaded_kits: Dict[str, int],
        current_day: int,
        current_hour: int,
        horizon_hours: int = 24,
        capacity: Dict[str, int] | None = None,
    ) -> Dict[str, object]:
        """
        Simulate a single-flight load override and return the penalty breakdown.

        Example:
            slots = predictor.forecast_with_custom_load("ZHVK", "<flight-id>", {"economy": 50}, day, hour)
            penalties = predictor.penalty_for_custom_load("ZHVK", "<flight-id>", {"economy": 50}, day, hour)

        Returns the same shape as forecast_penalty (over/negative, totals, per-class, estimated_penalty).
        """
        forecast_slots = self.forecast_with_custom_load(
            airport_code, flight_id, custom_loaded_kits, current_day, current_hour, horizon_hours
        )
        cap = capacity or self.airport_tracker.capacity.get(airport_code, {})
        return self.penalty_from_forecast(forecast_slots, cap)

    
    
    
    
    def simulate_with_payload(
        self,
        airport_code: str,
        payload: Dict,
        current_day: int,
        current_hour: int,
        horizon_hours: int = 24,
    ) -> List[Dict[str, object]]:
        """
        Forecast using a hypothetical payload: uses payload loads when present, otherwise passenger counts.
        Useful to see how a candidate payload would affect stocks.
        """
        now_abs = self._abs_hour(current_day, current_hour)
        horizon_abs = now_abs + horizon_hours

        total = deepcopy(self.airport_tracker.inventory.get(airport_code, {}))
        usable = deepcopy(self.airport_tracker.usable_inventory.get(airport_code, {}))

        total_events: Dict[int, Dict[str, int]] = {}
        usable_events: Dict[int, Dict[str, int]] = {}

        def add_event(bucket: Dict[int, Dict[str, int]], when: int, delta: Dict[str, int]) -> None:
            if when <= now_abs or when > horizon_abs:
                return
            cur = bucket.setdefault(when, {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0})
            for cls, val in delta.items():
                cur[cls] = cur.get(cls, 0) + int(val)

        # Pending events already scheduled (cleaning/purchases)
        for when, delta in self.airport_tracker._events.get(airport_code, []):  # type: ignore[attr-defined]
            add_event(usable_events, when, delta)

        # Map payload loads for quick lookup
        payload_loads = {}
        for item in payload.get("flightLoads", []):
            payload_loads[item["flightId"]] = {
                "first": int(item["loadedKits"].get("first", 0)),
                "business": int(item["loadedKits"].get("business", 0)),
                "premiumEconomy": int(item["loadedKits"].get("premiumEconomy", 0)),
                "economy": int(item["loadedKits"].get("economy", 0)),
            }

        flights_iter = self._last_response.get("flightUpdates", []) if self._last_response else []
        flights_by_id = {item["flightId"]: item for item in flights_iter}
        for fli in flights_by_id.values():
            evt = (fli.get("eventType") or "").upper().replace(" ", "_")
            if evt == "LANDED":
                continue
            dep_abs = self._abs_hour(fli["departure"]["day"], fli["departure"]["hour"])
            arr_abs = self._abs_hour(fli["arrival"]["day"], fli["arrival"]["hour"])
            origin = fli["originAirport"]
            dest = fli["destinationAirport"]
            passengers = fli.get("passengers", {})
            loaded = payload_loads.get(
                fli["flightId"],
                {
                    "first": int(passengers.get("first", 0)),
                    "business": int(passengers.get("business", 0)),
                    "premiumEconomy": int(passengers.get("premiumEconomy", 0)),
                    "economy": int(passengers.get("economy", 0)),
                },
            )

            if origin == airport_code and dep_abs > now_abs and evt in ("SCHEDULED", "CHECKED_IN"):
                add_event(total_events, dep_abs, {cls: -val for cls, val in loaded.items()})
                add_event(usable_events, dep_abs, {cls: -val for cls, val in loaded.items()})
            if dest == airport_code and arr_abs > now_abs and evt in ("SCHEDULED", "CHECKED_IN"):
                add_event(total_events, arr_abs, loaded)
                unused = {cls: max(0, loaded.get(cls, 0) - passengers.get(cls, 0)) for cls in loaded}
                used = {cls: min(loaded.get(cls, 0), passengers.get(cls, 0)) for cls in loaded}
                add_event(usable_events, arr_abs, unused)
                proc_time = self.airport_tracker.processing_time.get(dest, {})
                for cls, val in used.items():
                    if val <= 0:
                        continue
                    ready_abs = arr_abs + int(proc_time.get(cls, 0))
                    add_event(usable_events, ready_abs, {cls: val})

        timeline: List[Dict[str, object]] = []
        for abs_h in range(now_abs + 1, horizon_abs + 1):
            if abs_h in total_events:
                for cls, val in total_events[abs_h].items():
                    total[cls] = total.get(cls, 0) + val
            if abs_h in usable_events:
                for cls, val in usable_events[abs_h].items():
                    usable[cls] = usable.get(cls, 0) + val
            timeline.append(
                {"abs_hour": abs_h, "day": abs_h // 24, "hour": abs_h % 24, "total": deepcopy(total), "usable": deepcopy(usable)}
            )
        return timeline
