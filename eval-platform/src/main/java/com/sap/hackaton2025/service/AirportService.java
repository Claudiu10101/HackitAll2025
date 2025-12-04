package com.sap.hackaton2025.service;

import java.util.Collection;
import java.util.UUID;

import com.sap.hackaton2025.model.Airport;

public interface AirportService {
	Airport getById(UUID id);

	Airport getByCode(String code);

	Collection<Airport> getAll();

}
