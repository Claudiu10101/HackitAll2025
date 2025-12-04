package com.sap.hackaton2025.service;

import java.util.List;
import java.util.UUID;

import com.sap.hackaton2025.model.KitMovement;
import com.sap.hackaton2025.model.KitProcessing;

public interface KitProcessingService {
	void addAllToQueue(List<KitProcessing> kitProcessing);

	List<KitMovement> generateKitMovements(int day, int hour, UUID evaluationSessionId);

	List<KitProcessing> getPendingKitProcessings(UUID evaluationSessionId);

}
