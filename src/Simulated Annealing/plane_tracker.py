from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


def _load_map_from_request(request_json: Dict) -> Dict[str, Dict[str, int]]:
    """Extract committed loaded kits per flight from the last payload."""
    loads = {}
    for item in request_json.get("flightLoads", []):
        fid = str(item.get("flightId"))
        kits = item.get("loadedKits", {})
        loads[fid] = {
            "first": int(kits.get("first", 0)),
            "business": int(kits.get("business", 0)),
            "premiumEconomy": int(kits.get("premiumEconomy", 0)),
            "economy": int(kits.get("economy", 0)),
        }
    return loads


@dataclass
class PlaneStatus:
    status: str = "SCHEDULED"
    onboard_kits: Dict[str, int] = field(
        default_factory=lambda: {"first": 0, "business": 0, "premiumEconomy": 0, "economy": 0}
    )
    seen_in_response: bool = False


class PlaneTracker:
    """Tracks plane status and onboard kits using API responses and latest payload."""

    def __init__(self) -> None:
        self.planes: Dict[str, PlaneStatus] = {}
        self._last_payload_loads: Dict[str, Dict[str, int]] = {}

    def ingest_payload(self, request_json: Dict) -> None:
        """Store loads from the outgoing payload and assign them to planes as CHECKED_IN."""
        loads_map = _load_map_from_request(request_json)
        if loads_map:
            self._last_payload_loads = loads_map
        for fid, kits in loads_map.items():
            status = self.planes.get(fid, PlaneStatus())
            status.status = "CHECKED_IN"
            status.onboard_kits = kits
            self.planes[fid] = status

    def update(self, response_json: Dict, last_request_json: Dict) -> None:
        """Consume the latest response + payload to refresh plane statuses and onboard kits."""
        # Mark all as unseen; we'll flip to True for any flight present in this response
        for status in self.planes.values():
            status.seen_in_response = False
        loads_map = _load_map_from_request(last_request_json)
        if loads_map:
            self._last_payload_loads = loads_map
        committed_map = self._last_payload_loads or loads_map
        for upd in response_json.get("flightUpdates", []):
            fid = str(upd.get("flightId"))
            evt_raw = upd.get("eventType", "")
            evt = evt_raw.upper().replace(" ", "_")
            status = self.planes.get(fid, PlaneStatus())
            committed = committed_map.get(fid)

            # Always attach committed load if present (unless landed clears it later)
            if committed:
                status.onboard_kits = committed
            status.seen_in_response = True
            if evt == "SCHEDULED":
                status.status = "SCHEDULED"
            elif evt == "CHECKED_IN":
                status.status = "CHECKED_IN"
                if committed:
                    status.onboard_kits = committed

            elif evt == "LANDED":
                status.status = "LANDED"
            elif evt:
                status.status = evt

            self.planes[fid] = status

    def get_status(self, flight_id: str) -> PlaneStatus:
        """Get the tracked status for a given flight id (default is SCHEDULED/empty kits)."""
        return self.planes.get(flight_id, PlaneStatus())

    def get_all(self) -> Dict[str, PlaneStatus]:
        """Return the full map of tracked planes."""
        return self.planes

    def get_inventory(self, flight_id: str) -> Dict[str, int]:
        """Return the onboard kits for a flight, defaults to zeros if unknown."""
        return dict(self.get_status(flight_id).onboard_kits)
