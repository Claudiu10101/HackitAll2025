package com.sap.hackaton2025.service.impl;

import java.util.List;
import java.util.UUID;

import org.springframework.stereotype.Service;

import com.sap.hackaton2025.controller.dto.ReferenceHour;
import com.sap.hackaton2025.model.KitMovement;
import com.sap.hackaton2025.model.KitProcessing;
import com.sap.hackaton2025.persistence.KitProcessingRepository;
import com.sap.hackaton2025.service.AirportService;
import com.sap.hackaton2025.service.KitProcessingService;

import jakarta.transaction.Transactional;

@Service
public class KitProcessingServiceImpl implements KitProcessingService {

	private final KitProcessingRepository repository;
	private final AirportService airportService;

	public KitProcessingServiceImpl(KitProcessingRepository repository, AirportService airportService) {
		this.repository = repository;
		this.airportService = airportService;
	}

	@Override
	@Transactional
	public void addAllToQueue(List<KitProcessing> kitProcessing) {
		repository.saveAll(kitProcessing);

	}

	@Override
	@Transactional
	public List<KitMovement> generateKitMovements(int day, int hour, UUID evaluationSessionId) {
		List<KitProcessing> processingQueue = repository.getAllAvailableForProcessing(evaluationSessionId, day, hour);

		List<KitMovement> movements =

				processingQueue.stream().map(kitProc -> {

					int leadTime;
					double costPerKit;
					KitMovement movement = new KitMovement();
					var airport = airportService.getById(kitProc.getAirportId());

					switch (kitProc.getKitType()) {
					case A_FIRST_CLASS:
						movement.setFirstKits(kitProc.getQuantity());
						leadTime = airport.getFirstProcessingTimeHours();
						costPerKit = airport.getFirstProcessingCost();
						break;
					case B_BUSINESS:
						movement.setBusinessKits(kitProc.getQuantity());
						leadTime = airport.getBusinessProcessingTimeHours();
						costPerKit = airport.getBusinessProcessingCost();
						break;
					case C_PREMIUM_ECONOMY:
						movement.setPremiumEconomyKits(kitProc.getQuantity());
						leadTime = airport.getPremiumEconomyProcessingTimeHours();
						costPerKit = airport.getPremiumEconomyProcessingCost();
						break;
					case D_ECONOMY:
						movement.setEconomyKits(kitProc.getQuantity());
						leadTime = airport.getEconomyProcessingTimeHours();
						costPerKit = airport.getEconomyProcessingCost();
						break;
					default:
						leadTime = 0;
						costPerKit = 0;
						break;

					}

					movement.setEvaluationSessionId(kitProc.getEvaluationSessionId());
					movement.setAirportId(kitProc.getAirportId());
					ReferenceHour completionHour = ReferenceHour.addHours(day, hour, leadTime);
					movement.setDay(completionHour.day());
					movement.setHour(completionHour.hour());
					movement.setCost(costPerKit * kitProc.getQuantity());
					movement.setFlightId(null);

					return movement;
				}).toList();
		repository.markAsProcessed(evaluationSessionId, day, hour);
		return movements;
	}

	@Override
	public List<KitProcessing> getPendingKitProcessings(UUID evaluationSessionId) {
		return repository.getAllPendingProcessings(evaluationSessionId);
	}

}
