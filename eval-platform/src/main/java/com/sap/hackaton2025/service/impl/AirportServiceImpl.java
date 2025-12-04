package com.sap.hackaton2025.service.impl;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;

import com.sap.hackaton2025.model.Airport;
import com.sap.hackaton2025.persistence.AirportRepository;
import com.sap.hackaton2025.service.AirportService;

import jakarta.annotation.PostConstruct;

@Service
public class AirportServiceImpl implements AirportService {

	private Map<String, Airport> airports;
	private Map<UUID, Airport> airportsById;

	private final AirportRepository repository;

	public AirportServiceImpl(AirportRepository repository) {
		this.repository = repository;
	}

	@Override
	public Airport getById(UUID id) {
		return airportsById.get(id);
	}

	@Override
	public Airport getByCode(String code) {
		return airports.get(code);
	}

	@PostConstruct
	void init() {
		List<Airport> allAirports = repository.findAll();
		airports = allAirports.stream().collect(Collectors.toMap(Airport::getCode, a -> a));
		airportsById = allAirports.stream().collect(Collectors.toMap(Airport::getId, a -> a));

	}

	@Override
	public Collection<Airport> getAll() {
		return airports.values();
	}

}
