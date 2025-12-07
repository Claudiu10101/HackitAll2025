from __future__ import annotations

from typing import Any, Dict, List, Optional

from flight import AIRCRAFT_TYPES, AIRCRAFT_TYPES_BY_ID


class CostSimulator:
    """
    Computes cost/penalty breakdowns for flights based on the last API response
    and a given payload (flightLoads).

    Expected inputs:
      - loading_cost:  map airport_code -> {class -> cost per kit to load}
      - processing_cost: map airport_code -> {class -> cost per kit to process}
      - flight_distances: map flightId -> distance (used if response lacks distance)
      - penalty_factors: dict with:
            FLIGHT_OVERLOAD_FACTOR_PER_DISTANCE (float)
            UNFULFILLED_KIT_FACTOR_PER_DISTANCE (float)
        (other keys are ignored)
      - kit_weight: per-class weight used for movement cost
      - kit_cost: per-class unit cost (used for penalties)
    """

    def __init__(
        self,
        loading_cost: Dict[str, Dict[str, float]],
        processing_cost: Dict[str, Dict[str, float]],
        flight_distances: Dict[str, float],
        penalty_factors: Dict[str, float],
        kit_weight: Dict[str, float],
        kit_cost: Dict[str, float],
    ) -> None:
        self.loading_cost = loading_cost
        self.processing_cost = processing_cost
        self.flight_distances = flight_distances
        self.penalty_factors = penalty_factors
        self.kit_weight = kit_weight
        self.kit_cost = kit_cost

    def _payload_kits(self, payload: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
        """Map flightId -> loadedKits from the payload."""
        kits_map: Dict[str, Dict[str, int]] = {}
        for item in payload.get("flightLoads", []):
            fid = str(item.get("flightId"))
            lk = item.get("loadedKits", {}) or {}
            kits_map[fid] = {k: int(lk.get(k, 0)) for k in ["first", "business", "premiumEconomy", "economy"]}
        return kits_map

    def _aircraft_for(self, upd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Resolve aircraft info from flight update using code or id.
        Returns the aircraft entry from flight module maps, or None if unknown.
        """
        ac_code = upd.get("aircraftType")
        if ac_code and ac_code in AIRCRAFT_TYPES:
            return AIRCRAFT_TYPES[ac_code]
        ac_id = upd.get("aircraftTypeId") or upd.get("actualAircraftTypeId")
        if ac_id and ac_id in AIRCRAFT_TYPES_BY_ID:
            return AIRCRAFT_TYPES_BY_ID[ac_id]
        return None

    def compute(
        self,
        response_json: Dict[str, Any],
        payload_json: Dict[str, Any],
        origin_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compute cost + penalty per flight present in response_json using the provided payload_json.
        origin_filter (optional) restricts to flights departing that airport.
        Returns a list of dicts with cost/penalty breakdowns per flight.
        """
        kits_map = self._payload_kits(payload_json)
        results: List[Dict[str, Any]] = []
        for upd in response_json.get("flightUpdates", []):
            fid = str(upd.get("flightId"))
            origin = upd.get("originAirport")
            dest = upd.get("destinationAirport")
            if origin_filter and origin != origin_filter:
                continue
            kits = kits_map.get(fid, {})
            if not kits or all(v == 0 for v in kits.values()):
                continue

            passengers = upd.get("passengers", {}) or {}
            distance = float(
                upd.get("actualDistance")
                or upd.get("distance")
                or upd.get("plannedDistance")
                or self.flight_distances.get(fid, 0)
                or 0
            )
            aircraft = self._aircraft_for(upd)
            fuel_cost = float(getattr(aircraft, "cost_per_kg_per_km", 0) if aircraft else 0)
            kit_capacity = getattr(aircraft, "kit_capacity", {}) if aircraft else {}

            lc = self.loading_cost.get(origin, {})
            pc = self.processing_cost.get(dest, {})

            per_class: Dict[str, Dict[str, float]] = {}
            for cls in ["first", "business", "premiumEconomy", "economy"]:
                qty = int(kits.get(cls, 0))
                per_class[cls] = {
                    "qty": qty,
                    "loading": qty * float(lc.get(cls, 0)),
                    "movement": distance * fuel_cost * qty * float(self.kit_weight.get(cls, 0)),
                    "processing": qty * float(pc.get(cls, 0)),
                }

            loading_cost_val = sum(v["loading"] for v in per_class.values())
            movement_cost_val = sum(v["movement"] for v in per_class.values())
            processing_cost_val = sum(v["processing"] for v in per_class.values())
            total_cost = loading_cost_val + movement_cost_val + processing_cost_val

            penalties = 0.0
            overload_detail: Dict[str, Dict[str, float]] = {}
            unfulfilled_detail: Dict[str, Dict[str, float]] = {}

            overload_factor = float(self.penalty_factors.get("FLIGHT_OVERLOAD_FACTOR_PER_DISTANCE", 0))
            unfulfilled_factor = float(self.penalty_factors.get("UNFULFILLED_KIT_FACTOR_PER_DISTANCE", 0))

            if kit_capacity:
                for cls, cap in kit_capacity.items():
                    overload = max(0, int(kits.get(cls, 0)) - int(cap))
                    if overload > 0:
                        cost = overload_factor * distance * fuel_cost * self.kit_cost.get(cls, 0) * overload
                        penalties += cost
                        overload_detail[cls] = {"overloadQty": overload, "cost": cost}

            for cls, pax in passengers.items():
                missing = max(0, int(pax) - int(kits.get(cls, 0)))
                if missing > 0:
                    cost = unfulfilled_factor * distance * self.kit_cost.get(cls, 0) * missing
                    penalties += cost
                    unfulfilled_detail[cls] = {"missingQty": missing, "cost": cost}

            penalty_breakdown = {
                "overload": overload_detail,
                "unfulfilled": unfulfilled_detail,
                "totalPenalties": round(penalties, 2),
            }

            results.append(
                {
                    "flightId": fid,
                    "origin": origin,
                    "destination": dest,
                    "distance": distance,
                    "loadingCost": round(loading_cost_val, 2),
                    "movementCost": round(movement_cost_val, 2),
                    "processingCost": round(processing_cost_val, 2),
                    "totalCost": round(total_cost, 2),
                    "perClass": per_class,
                    "penalties": round(penalties, 2),
                    "penalty_breakdown": penalty_breakdown,
                }
            )
        return results

