package com.sap.hackaton2025.service.impl;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

import org.springframework.stereotype.Service;

import com.sap.hackaton2025.model.KitMovement;
import com.sap.hackaton2025.persistence.KitMovementRepository;
import com.sap.hackaton2025.service.KitMovementService;

import jakarta.transaction.Transactional;

@Service
public class KitMovementServiceImpl implements KitMovementService {

	private final KitMovementRepository kitMovementRepository;
	
	public KitMovementServiceImpl(KitMovementRepository kitMovementRepository) {
		this.kitMovementRepository = kitMovementRepository;
	}
	
	@Override
	public Optional<KitMovement> getByFlightAndSession(UUID flightId, UUID sessionId) {
		return kitMovementRepository.findByFlightIdAndSessionId(flightId, sessionId);
	}

	@Override
	@Transactional
	public void saveAll(List<KitMovement> kitMovements) {
		kitMovementRepository.saveAll(kitMovements);
	}

	@Override
	public List<KitMovement> getByDayHourAndSession(int day, int hour, UUID sessionId) {
		return kitMovementRepository.findByDayHourAndEvaluationSessionId(day, hour, sessionId);
	}

}
