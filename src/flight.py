from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import airport
PassengerCounts = Dict[str, int]
TimePoint = Tuple[int, int]  # (day, hour)

@dataclass(frozen=True)
class plane_data:
    id: str
    type_code: str
    seat_capacity: PassengerCounts
    kit_capacity: PassengerCounts
    cost_per_kg_per_km: float


def _plane(
    *,
    id: str,
    type_code: str,
    seats: Tuple[int, int, int, int],
    kits: Tuple[int, int, int, int],
    cost_per_kg_per_km: float,
) -> plane_data:
    """Helper to build plane_data without repeating the class keys."""
    return plane_data(
        id=id,
        type_code=type_code,
        seat_capacity={
            "first": seats[0],
            "business": seats[1],
            "premiumEconomy": seats[2],
            "economy": seats[3],
        },
        kit_capacity={
            "first": kits[0],
            "business": kits[1],
            "premiumEconomy": kits[2],
            "economy": kits[3],
        },
        cost_per_kg_per_km=cost_per_kg_per_km,
    )


# id;type_code;first_class_seats;business_seats;premium_economy_seats;economy_seats;cost_per_kg_per_km;first_class_kits_capacity;business_kits_capacity;premium_economy_kits_capacity;economy_kits_capacity
AIRCRAFT_TYPES: Dict[str, plane_data] = {
    "OJF294": _plane(
        id="b2017ed7-66c7-4498-b647-500ea1ef03d6",
        type_code="OJF294",
        seats=(13, 67, 31, 335),
        cost_per_kg_per_km=0.08,
        kits=(18, 105, 44, 781),
    ),
    "NHY337": _plane(
        id="6f8c4d73-5e4d-48dc-9396-51000e68a562",
        type_code="NHY337",
        seats=(4, 30, 17, 156),
        cost_per_kg_per_km=0.09,
        kits=(4, 66, 44, 438),
    ),
    "WTA646": _plane(
        id="020941c7-3b40-4654-8b78-c755f48571c6",
        type_code="WTA646",
        seats=(20, 63, 28, 329),
        cost_per_kg_per_km=0.10,
        kits=(30, 126, 71, 770),
    ),
    "UHB596": _plane(
        id="7e61f470-91fe-40e5-9926-9ec7849506dd",
        type_code="UHB596",
        seats=(7, 41, 27, 196),
        cost_per_kg_per_km=0.11,
        kits=(15, 67, 54, 329),
    ),
}

AIRCRAFT_TYPES_BY_ID: Dict[str, plane_data] = {plane.id: plane for plane in AIRCRAFT_TYPES.values()}


@dataclass
class flight_data:
    id: str
    origin: str
    destination: str
    departure_time: TimePoint
    arrival_time: TimePoint
    real_passengers: PassengerCounts
    current_status: str
    plane: Optional[plane_data] = None
    processed: bool = False
    loaded_packs: PassengerCounts = field(
        default_factory=lambda: {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}
    )
    distance: Optional[float] = None


class flight:
    flights: Dict[str, flight_data] = {}

    @classmethod
    def load_data_from_json(cls, json_data):
        for flight_update in json_data.get("flightUpdates", []):
            plane = AIRCRAFT_TYPES.get(flight_update.get("aircraftType"))
            fid = flight_update["flightId"]
            existing = cls.flights.get(fid)
            loaded_packs = existing.loaded_packs if existing else {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}
            flight_instance = flight_data(
                id=fid,
                origin=flight_update["originAirport"],
                destination=flight_update["destinationAirport"],
                departure_time=(flight_update["departure"]["day"], flight_update["departure"]["hour"]),
                arrival_time=(flight_update["arrival"]["day"], flight_update["arrival"]["hour"]),
                real_passengers={
                    "first": flight_update["passengers"]["first"],
                    "business": flight_update["passengers"]["business"],
                    "premiumEconomy": flight_update["passengers"]["premiumEconomy"],
                    "economy": flight_update["passengers"]["economy"],
                },
                current_status=flight_update["eventType"],
                plane=plane,
                loaded_packs=loaded_packs,
                distance=float(
                    flight_update.get("actualDistance")
                    or flight_update.get("distance")
                    or flight_update.get("plannedDistance")
                    or 0
                ),
            )
            cls.flights[flight_instance.id] = flight_instance
            cls.cleanup_finished(json_data.get("day", 0), json_data.get("hour", 0))

    @classmethod
    def to_play_round_updates(cls, include_zero_loads: bool = False):
        """
        Return the `flightLoads` list for the play/round payload.

        Expected shape per API (FlightLoadDto):
        [{"flightId": "<uuid>", "loadedKits": {"first":0,"business":0,"premiumEconomy":0,"economy":0}}, ...]
        """
        updates = []
        for f in cls.flights.values():
            # Always clamp outgoing payload to plane kit capacity (defensive).
            loaded_raw = f.loaded_packs or {}
            if f.plane:
                loaded = {
                    "first": min(int(loaded_raw.get("first", 0)), f.plane.kit_capacity["first"]),
                    "business": min(int(loaded_raw.get("business", 0)), f.plane.kit_capacity["business"]),
                    "premiumEconomy": min(int(loaded_raw.get("premiumEconomy", 0)), f.plane.kit_capacity["premiumEconomy"]),
                    "economy": min(int(loaded_raw.get("economy", 0)), f.plane.kit_capacity["economy"]),
                }
            else:
                loaded = {
                    "first": int(loaded_raw.get("first", 0)),
                    "business": int(loaded_raw.get("business", 0)),
                    "premiumEconomy": int(loaded_raw.get("premiumEconomy", 0)),
                    "economy": int(loaded_raw.get("economy", 0)),
                }
            if not include_zero_loads and all(value == 0 for value in loaded.values()):
                continue
            updates.append({"flightId": f.id, "loadedKits": loaded})
        return updates
            



    # def load_plane(cls):
    #     for fli in flight.flights.values():
    #         if fli.current_status not in ("CHECKED_IN", "SCHEDULED"):
    #             continue
    #         first, business, premium, economy = airport.AirportTracker.max_available( cls,fli.origin, fli.departure_time[0]*24 + fli.departure_time[1]).values()
    #         fli.loaded_packs = {
    #             "first": int(fli.real_passengers.get("first", 0)),
    #             "business": int(fli.real_passengers.get("business", 0)),
    #             "premiumEconomy": int(fli.real_passengers.get("premiumEconomy", 0)),
    #             "economy": int(fli.real_passengers.get("economy", 0)) + 20,
    #         }
        

    @classmethod
    def cleanup_finished(cls, current_day: int, current_hour: int) -> None:
        """Remove flights whose arrival time is strictly before the given (day, hour)."""
        finished = []
        for fid, f in cls.flights.items():
            arr_day, arr_hour = f.arrival_time
            if (arr_day < current_day) or (arr_day == current_day and arr_hour < current_hour):
                finished.append(fid)
        for fid in finished:
            cls.flights.pop(fid, None)

    @classmethod
    def debug_airport_tracking(cls, airport_code: str, request_json: Dict, response_json: Dict) -> None:
        """Verbose log of flights touching an airport with passengers and loaded kits."""
        print(f"\n=== Airport tracking debug for {airport_code} ===")
        loads = {}
        for item in request_json.get("flightLoads", []):
            fid = str(item.get("flightId"))
            loads[fid] = item.get("loadedKits", {})
        for upd in response_json.get("flightUpdates", []):
            fid = str(upd.get("flightId"))
            origin = upd.get("originAirport")
            dest = upd.get("destinationAirport")
            evt = upd.get("eventType")
            if origin != airport_code and dest != airport_code:
                continue
            passengers = upd.get("passengers", {})
            kits = loads.get(fid, {})
            print(
                f"[{evt}] flight {upd.get('flightNumber')} ({fid}) {origin}->{dest} "
                f"dep {upd.get('departure')} arr {upd.get('arrival')}"
            )
            print(f"    passengers: {passengers}")
            print(f"    loaded kits sent: {kits}")
        print("=== End airport tracking ===\n")
            
    
