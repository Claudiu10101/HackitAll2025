package com.sap.hackaton2025.service.impl;

import java.util.List;

import org.springframework.stereotype.Service;

import com.sap.hackaton2025.model.FlightLoad;
import com.sap.hackaton2025.persistence.FlightLoadRepository;
import com.sap.hackaton2025.service.FlightLoadService;

import jakarta.transaction.Transactional;

@Service
public class FlightLoadServiceImpl implements FlightLoadService {

	private final FlightLoadRepository repository;

	public FlightLoadServiceImpl(FlightLoadRepository repository) {
		this.repository = repository;
	}

	@Override
	@Transactional
	public void saveAll(List<FlightLoad> flightLoads) {
		if (flightLoads != null && !flightLoads.isEmpty()) {
			repository.saveAll(flightLoads);
		}
	}

}
