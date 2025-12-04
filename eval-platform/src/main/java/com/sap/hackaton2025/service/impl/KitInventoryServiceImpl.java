package com.sap.hackaton2025.service.impl;

import java.util.Collection;
import java.util.List;
import java.util.UUID;
import java.util.stream.Stream;

import org.springframework.stereotype.Service;

import com.sap.hackaton2025.model.Airport;
import com.sap.hackaton2025.model.EvaluationSession;
import com.sap.hackaton2025.model.KitInventory;
import com.sap.hackaton2025.model.KitType;
import com.sap.hackaton2025.persistence.KitInventoryRepository;
import com.sap.hackaton2025.service.KitInventoryService;

import jakarta.transaction.Transactional;

@Service
public class KitInventoryServiceImpl implements KitInventoryService {

	private final KitInventoryRepository kitInventoryRepository;

	public KitInventoryServiceImpl(KitInventoryRepository kitInventoryRepository) {
		this.kitInventoryRepository = kitInventoryRepository;
	}

	@Override
	@Transactional
	public void initKitInventories(EvaluationSession session, Collection<Airport> airports) {
		kitInventoryRepository.saveAll(airports.stream()
				.flatMap(airport -> Stream.of(KitType.values()).map(kitType -> initialKit(session, airport, kitType)))
				.toList());
	}

	private KitInventory initialKit(EvaluationSession evaluationSession, Airport airport, KitType kitType) {
		KitInventory kitInventory = new KitInventory();
		kitInventory.setAirportId(airport.getId());
		kitInventory.setKitType(kitType);
		kitInventory.setEvaluationSessionId(evaluationSession.getId());
		int initialStock = switch (kitType) {
		case B_BUSINESS -> airport.getInitialBCStock();
		case D_ECONOMY -> airport.getInitialECStock();
		case A_FIRST_CLASS -> airport.getInitialFCStock();
		case C_PREMIUM_ECONOMY -> airport.getInitialPEStock();
		default -> 0;
		};

		int capacity = switch (kitType) {
		case B_BUSINESS -> airport.getCapacityBC();
		case D_ECONOMY -> airport.getCapacityEC();
		case A_FIRST_CLASS -> airport.getCapacityFC();
		case C_PREMIUM_ECONOMY -> airport.getCapacityPE();
		default -> 0;
		};

		kitInventory.setAvailableKits(initialStock);
		kitInventory.setCapacity(capacity);
		return kitInventory;
	}

	@Override
	@Transactional
	public void saveAll(List<KitInventory> kitInventories) {
		kitInventoryRepository.saveAll(kitInventories);
	}

	@Override
	public List<KitInventory> getCurrentInventories(UUID sessionId) {
		return kitInventoryRepository.findBySessionId(sessionId);
	}

}