from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
from typing import Dict, List, Tuple

from plane_tracker import PlaneTracker

DEFAULT_AIRPORTS_CSV = (
    Path(__file__).resolve().parent.parent
    / "eval-platform/src/main/resources/liquibase/data/airports_with_stocks.csv"
)


def load_initial_data(
    csv_path: Path = DEFAULT_AIRPORTS_CSV,
) -> Tuple[
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
]:
    inv: Dict[str, Dict[str, int]] = {}
    proc: Dict[str, Dict[str, int]] = {}
    cap: Dict[str, Dict[str, int]] = {}
    load_cost: Dict[str, Dict[str, float]] = {}
    proc_cost: Dict[str, Dict[str, float]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            code = row["code"]
            # Initial stocks per class
            inv[code] = {
                "first": int(row["initial_fc_stock"]),
                "business": int(row["initial_bc_stock"]),
                "premiumEconomy": int(row["initial_pe_stock"]),
                "economy": int(row["initial_ec_stock"]),
            }
            # Processing times per class (hours until used kits become clean)
            proc[code] = {
                "first": int(row["first_processing_time"]),
                "business": int(row["business_processing_time"]),
                "premiumEconomy": int(row["premium_economy_processing_time"]),
                "economy": int(row["economy_processing_time"]),
            }
            # Warehouse capacity per class
            cap[code] = {
                "first": int(row["capacity_fc"]),
                "business": int(row["capacity_bc"]),
                "premiumEconomy": int(row["capacity_pe"]),
                "economy": int(row["capacity_ec"]),
            }
            load_cost[code] = {
                "first": float(row.get("first_loading_cost", 0)),
                "business": float(row.get("business_loading_cost", 0)),
                "premiumEconomy": float(row.get("premium_economy_loading_cost", 0)),
                "economy": float(row.get("economy_loading_cost", 0)),
            }
            proc_cost[code] = {
                "first": float(row.get("first_processing_cost", 0)),
                "business": float(row.get("business_processing_cost", 0)),
                "premiumEconomy": float(row.get("premium_economy_processing_cost", 0)),
                "economy": float(row.get("economy_processing_cost", 0)),
            }
    return inv, proc, cap, load_cost, proc_cost


class AirportTracker:
    """
    Tracks arrivals/departures per airport and keeps inventories:
      - inventory: total kits on the ground (clean + dirty)
      - usable_inventory: kits ready now (clean only)
      - reservations: per (airport, abs_hour) reserved kits for outgoing payloads
    It also tracks:
      - processing times to turn dirty kits into clean ones
      - capacity per class (to avoid overfills)
      - pending arrivals of purchases/processed kits (events queued by schedule_purchase/_apply_ready)
    The main flow each round:
      * call allocate_for_flights before sending payload to reserve usable stock
      * call update after receiving response to apply departures/arrivals and move ready events
      * optional: print_airport/print_total_network_inventory for debugging snapshots
    """

    def __init__(
        self,
        initial_inventory: Dict[str, Dict[str, int]],
        processing_time: Dict[str, Dict[str, int]],
        capacity: Dict[str, Dict[str, int]],
        loading_cost: Dict[str, Dict[str, float]] | None = None,
        processing_cost: Dict[str, Dict[str, float]] | None = None,
    ) -> None:
        self.departures: Dict[str, List[str]] = defaultdict(list)  # flights checked-in from this airport this round
        self.arrivals: Dict[str, List[str]] = defaultdict(list)  # flights landed at this airport this round
        self.arrival_inventory: Dict[str, List[Dict[str, object]]] = defaultdict(list)  # inventory per arriving flight
        self.inventory: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}
        )  # total stock (clean + dirty)
        self.usable_inventory: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}
        )  # clean stock ready to load
        self.processing_time = processing_time  # hours for used kits to become clean
        self.capacity = capacity  # storage caps per class
        self.loading_cost = loading_cost or {}
        self.processing_cost_table = processing_cost or {}
        # Purchase lead-times (hours) from backend KitType enum
        self.purchase_lead_hours = {
            "first": 48,
            "business": 36,
            "premiumEconomy": 24,
            "economy": 12,
        }
        for code, inv in initial_inventory.items():
            self.inventory[code] = dict(inv)
            self.usable_inventory[code] = dict(inv)

        self._events: Dict[str, List[Tuple[int, Dict[str, int]]]] = defaultdict(list)  # airport -> list of (ready_abs, delta)
        self.reservations: Dict[Tuple[str, int], Dict[str, int]] = defaultdict(
            lambda: {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}
        )  # reserved kits per airport-hour
        self.reservation_by_flight: Dict[str, Dict[str, int]] = {}  # reservation applied to a flight
        self.reservation_info: Dict[str, Dict[str, object]] = {}  # debug info per flight reservation
        self.pending_purchases: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}
        )  # outstanding purchase orders per airport
        self.pending_arrivals: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}
        )  # outstanding clean arrivals scheduled

    def _apply_ready(self, current_abs: int) -> None:
        # Move any ready events into usable inventory and reduce pending trackers.
        # Events come from purchases or processed kits scheduled via schedule_purchase/_events.
        for ap, events in list(self._events.items()):
            ready = [evt for evt in events if evt[0] <= current_abs]
            self._events[ap] = [evt for evt in events if evt[0] > current_abs]
            for _, delta in ready:
                for cls, val in delta.items():
                    self.usable_inventory[ap][cls] = self.usable_inventory[ap].get(cls, 0) + val
                    # decrease pending arrivals as they land
                    if self.pending_arrivals[ap].get(cls, 0) > 0:
                        self.pending_arrivals[ap][cls] = max(self.pending_arrivals[ap][cls] - val, 0)

    def max_available(self, airport_code: str, abs_hour: int) -> Dict[str, int]:
        """Return usable inventory minus reservations at a given abs_hour."""
        base = self.usable_inventory.get(airport_code, {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0})
        reserved = self.reservations.get((airport_code, abs_hour), {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0})
        return {cls: max(base.get(cls, 0) - reserved.get(cls, 0), 0) for cls in base}

    def plan_hub_purchase(self, day: int, hour: int, target_fill: float = 0.5) -> Dict[str, int]:
        """
        Simple purchase strategy for HUB1: top up towards target_fill * capacity.
        Returns dict ready to send in kitPurchasingOrders (no immediate stock change; delivery is delayed).
        """
        hub = "HUB1"
        if hub not in self.capacity:
            return {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}
        order: Dict[str, int] = {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}
        for cls, cap in self.capacity[hub].items():
            desired = int(cap * target_fill)
            cur = self.inventory[hub].get(cls, 0)
            pending = self.pending_arrivals[hub].get(cls, 0)
            available_room = max(0, cap - (cur + pending))
            need = max(0, desired - (cur + pending))
            order[cls] = min(need, available_room)
        return {k: int(v) for k, v in order.items()}

    def schedule_purchase(self, airport_code: str, order: Dict[str, int], day: int, hour: int) -> None:
        """Schedule arrival of a purchase order using lead times; does not change inventory immediately."""
        if not order:
            return
        abs_now = day * 24 + hour
        for cls, qty in order.items():
            qty = int(qty)
            if qty <= 0:
                continue
            cap = self.capacity.get(airport_code, {}).get(cls)
            cur = self.inventory[airport_code].get(cls, 0)
            pending = self.pending_arrivals[airport_code].get(cls, 0)
            if cap is not None:
                room = max(0, cap - (cur + pending))
                qty = min(qty, room)
            if qty <= 0:
                continue
            lead = self.purchase_lead_hours.get(cls, 0)
            ready_abs = abs_now + lead
            self._events[airport_code].append((ready_abs, {cls: qty}))
            # Track pending arrivals and bump total inventory immediately for visibility
            self.pending_arrivals[airport_code][cls] = self.pending_arrivals[airport_code].get(cls, 0) + qty
            self.inventory[airport_code][cls] = self.inventory[airport_code].get(cls, 0) + qty


    def reserve_inventory(self, airport_code: str, abs_hour: int, requested: Dict[str, int]) -> Dict[str, int]:
        """
        Reserve kits for a departure. Returns the amount actually reserved (clamped by availability).
        """
        available = self.max_available(airport_code, abs_hour)
        reservation_key = (airport_code, abs_hour)
        current_res = self.reservations[reservation_key]

        actual: Dict[str, int] = {}
        for cls, val in requested.items():
            take = min(available.get(cls, 0), int(val))
            actual[cls] = take
            current_res[cls] = current_res.get(cls, 0) + take
        self.reservations[reservation_key] = current_res
        return actual


    def allocate_for_flights(self, flights) -> None:
        """
        Rebuild reservations and set flight.loaded_packs based on available usable stock.
        Only SCHEDULED flights are processed; CHECKED_IN flights keep their previous load.
        """
        self.reservations.clear()
        self.reservation_by_flight.clear()
        self.reservation_info.clear()
        # iterate in given order
        for fli in flights:
            # Only reserve for flights not yet checked in; once CHECKED_IN we keep the previous load
            if fli.current_status not in ("SCHEDULED",) or not getattr(fli, "plane", None):
                continue
            abs_dep = fli.departure_time[0] * 24 + fli.departure_time[1]
            # Load based on passenger demand (no forced full capacity)
            # requested = {
            #     "first": int(fli.real_passengers.get("first", 0)),
            #     "business": int(fli.real_passengers.get("business", 0)),
            #     "premiumEconomy": int(fli.real_passengers.get("premiumEconomy", 0)),
            #     "economy": int(fli.real_passengers.get("economy", 0)),
            # }
            requested = {           
                "first" : 0,
                "business" : 0,
                "premiumEconomy" : 0,
                "economy" : 0
                
            }
            
            for k in requested:
                requested[k] = min(requested[k], fli.plane.kit_capacity[k])  # avoid overloading plane
            available_before = self.max_available(fli.origin, abs_dep)  # usable minus existing reservations
            actual = self.reserve_inventory(fli.origin, abs_dep, requested)  # clamp by current usable
            fli.loaded_packs = actual  # store what we actually reserved to send in payload
            self.reservation_by_flight[fli.id] = actual
            self.reservation_info[fli.id] = {
                "airport": fli.origin,
                "abs_hour": abs_dep,
                "requested": requested,
                "allocated": actual,
                "available_before": available_before,
            }


    def update(self, response_json: Dict, plane_tracker: PlaneTracker) -> None:
        """
        Apply the latest API response to airport state:
          - advance time and apply ready events
          - clear per-cycle arrivals/departures
          - on CHECKED_IN: record departure, reduce ground + usable (load leaves)
          - on LANDED: record arrival, add kits to ground, split unused vs used (used go to processing queue)
        """
        day = int(response_json.get("day", 0))
        hour = int(response_json.get("hour", 0))
        current_abs = day * 24 + hour
        self._apply_ready(current_abs)

        # Clear per-cycle events
        self.departures = defaultdict(list)
        self.arrivals = defaultdict(list)
        self.arrival_inventory = defaultdict(list)
        for upd in response_json.get("flightUpdates", []):
            fid = str(upd.get("flightId"))
            evt = (upd.get("eventType") or "").upper().replace(" ", "_")
            origin = upd.get("originAirport")
            dest = upd.get("destinationAirport")
            if evt == "CHECKED_IN" and origin:
                self.departures[origin].append(fid)  # mark this flight as departing now
                # Clear any reservation tied to this flight (it is now committed)
                try:
                    dep = upd.get("departure", {})
                    abs_dep = int(dep.get("day", 0)) * 24 + int(dep.get("hour", 0))
                except Exception:
                    abs_dep = None
                res_flight = self.reservation_by_flight.pop(fid, None)
                self.reservation_info.pop(fid, None)
                if abs_dep is not None and res_flight:
                    key = (origin, abs_dep)
                    res_hour = self.reservations.get(key)
                    if res_hour:
                        for cls, val in res_flight.items():
                            res_hour[cls] = max(res_hour.get(cls, 0) - int(val), 0)
                        if all(v == 0 for v in res_hour.values()):
                            self.reservations.pop(key, None)
                inv = plane_tracker.get_inventory(fid)
                # already reserved usable; just adjust total
                for cls, val in inv.items():
                    self.inventory[origin][cls] = self.inventory[origin].get(cls, 0) - int(val)  # remove from ground (allow negative)
                    self.usable_inventory[origin][cls] = self.usable_inventory[origin].get(cls, 0) - int(val)  # remove from clean (allow negative)
            if evt == "LANDED" and dest:
                self.arrivals[dest].append(fid)  # mark this flight as arrived
                inv = plane_tracker.get_inventory(fid)
                passengers = upd.get(
                    "passengers", {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}
                )
                self.arrival_inventory[dest].append({"flightId": fid, "inventory": inv})
                for cls, val in inv.items():
                    self.inventory[dest][cls] = self.inventory[dest].get(cls, 0) + int(val)  # add all kits to ground
                proc_time = self.processing_time.get(dest, {})
                for cls, val in inv.items():
                    loaded = int(val)
                    pax = int(passengers.get(cls, 0))
                    unused = max(0, loaded - pax)
                    used = min(loaded, pax)
                    if unused:
                        self.usable_inventory[dest][cls] = self.usable_inventory[dest].get(cls, 0) + unused
                    if used:
                        ready = current_abs + proc_time.get(cls, 0)
                        self._events[dest].append((ready, {cls: used}))

    def print_airport(self, airport_code: str) -> None:
        # Minimal debug snapshot for a single airport
        print(f"\n=== Airport {airport_code} ===")
        print(f"Total inventory: {self.inventory.get(airport_code, {})}")
        print(f"Usable inventory: {self.usable_inventory.get(airport_code, {})}")
        print("=== End ===\n")

    def _sum_dicts(self, *dicts: Dict[str, int]) -> Dict[str, int]:
        agg = {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}
        for d in dicts:
            for cls in agg:
                agg[cls] += int(d.get(cls, 0))
        return agg

    def print_total_network_inventory(self, plane_tracker: PlaneTracker | None = None) -> None:
        """
        Print network totals broken down by:
        - Ground inventory (sum of airport totals)
        - Pending arrivals (purchases/processing events scheduled)
        - In-transit (kits on planes not yet landed)
        """
        ground = self._sum_dicts(*self.inventory.values())
        pending = self._sum_dicts(*self.pending_arrivals.values())
        in_transit = {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}
        if plane_tracker:
            for fid, status in plane_tracker.get_all().items():
                # Only count flights seen in the latest response and not yet LANDED
                if not status.seen_in_response:
                    continue
                if status.status.upper() == "LANDED":
                    continue
                in_transit = {
                    "first": in_transit["first"] + int(status.onboard_kits.get("first", 0)),
                    "business": in_transit["business"] + int(status.onboard_kits.get("business", 0)),
                    "premiumEconomy": in_transit["premiumEconomy"] + int(status.onboard_kits.get("premiumEconomy", 0)),
                    "economy": in_transit["economy"] + int(status.onboard_kits.get("economy", 0)),
                }
        total_all = {
            cls: ground.get(cls, 0) + pending.get(cls, 0) + in_transit.get(cls, 0)
            for cls in ground
        }
        print(f"\n=== Network totals ===")
        print(f"Ground total: {ground}")
        print(f"Pending arrivals (purchases/processing): {pending}")
        print(f"In transit (on planes not yet LANDED): {in_transit}")
        print(f"Combined (ground + pending + in-transit): {total_all}")
        print(f"=== End ===\n")

    def debug_print(self, airport_code: str) -> None:
        self.print_airport(airport_code)

    def get_plane_inventory(self, plane_tracker: PlaneTracker, airport_code: str) -> Dict[str, Dict[str, int]]:
        """
        Return onboard kits for planes departing from this airport in the current cycle.
        """
        inventories: Dict[str, Dict[str, int]] = {}
        for fid in self.departures.get(airport_code, []):
            inventories[fid] = plane_tracker.get_status(fid).onboard_kits
        return inventories
