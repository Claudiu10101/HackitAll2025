package com.sap.hackaton2025.service;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

import com.sap.hackaton2025.model.KitMovement;

public interface KitMovementService {

	Optional<KitMovement> getByFlightAndSession(UUID flightId, UUID sessionId);

	void saveAll(List<KitMovement> kitMovements);

	List<KitMovement> getByDayHourAndSession(int day, int hour, UUID sessionId);

}
