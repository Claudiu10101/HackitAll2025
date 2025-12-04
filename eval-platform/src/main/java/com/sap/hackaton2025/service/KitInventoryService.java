package com.sap.hackaton2025.service;

import java.util.Collection;
import java.util.List;
import java.util.UUID;

import com.sap.hackaton2025.model.Airport;
import com.sap.hackaton2025.model.EvaluationSession;
import com.sap.hackaton2025.model.KitInventory;

public interface KitInventoryService {
	void initKitInventories(EvaluationSession session, Collection<Airport> airports);

	void saveAll(List<KitInventory> kitInventories);

	List<KitInventory> getCurrentInventories(UUID sessionId);

}