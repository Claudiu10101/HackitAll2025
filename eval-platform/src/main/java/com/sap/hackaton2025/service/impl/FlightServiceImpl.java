package com.sap.hackaton2025.service.impl;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import com.sap.hackaton2025.controller.dto.ReferenceHour;
import com.sap.hackaton2025.model.Flight;
import com.sap.hackaton2025.persistence.FlightRepository;
import com.sap.hackaton2025.service.FlightService;

import jakarta.annotation.PostConstruct;

@Service
public class FlightServiceImpl implements FlightService {

	private static final Logger logger = LoggerFactory.getLogger(FlightServiceImpl.class);

	private List<Flight> cachedFlights;
	private Map<UUID, Flight> flightsById;

	private final FlightRepository flightRepository;

	public FlightServiceImpl(FlightRepository flightRepository) {
		this.flightRepository = flightRepository;
	}

	@PostConstruct
	private void init() {
		logger.atInfo().log("Initializing FlightServiceImpl and caching flights from repository");
		cachedFlights = Collections.unmodifiableList(flightRepository.findAll());
		logger.atInfo().log("Cached {} flights", cachedFlights.size());
		flightsById = cachedFlights.stream().collect(java.util.stream.Collectors.toMap(Flight::getId, f -> f));
		logger.atInfo().log("FlightServiceImpl initialization complete");

	}

	@Override
	public Flight getFlightById(UUID flightId) {
		return flightsById.get(flightId);
	}

	@Override
	public List<Flight> getScheduledFlights(int day, int hour) {
		ReferenceHour endHour = ReferenceHour.addHours(day, hour, 24);
		ReferenceHour startHour = new ReferenceHour(day, hour).addHours(1);
		return cachedFlights.stream().filter(flight -> {
			int flightDay = flight.getScheduledDepartDay();
			int flightHour = flight.getScheduledDepartHour();
			boolean afterStart = (flightDay > startHour.day())
					|| (flightDay == startHour.day() && flightHour >= startHour.hour());
			boolean beforeEnd = (flightDay < endHour.day())
					|| (flightDay == endHour.day() && flightHour <= endHour.hour());
			return afterStart && beforeEnd;
		}).toList();
	}

	@Override
	public List<Flight> getCheckedInFlights(int day, int hour) {
		return cachedFlights.stream()
				.filter(flight -> flight.getScheduledDepartDay() == day && flight.getScheduledDepartHour() == hour)
				.toList();
	}

	@Override
	public List<Flight> getLandedFlights(int day, int hour) {
		return cachedFlights.stream()
				.filter(flight -> flight.getActualArrivalDay() == day && flight.getActualArrivalHour() == hour)
				.toList();
	}

	@Override
	public List<Flight> getUpcomingFlights(int day, int hour) {
		return cachedFlights.stream().filter(flight -> {
			int flightDay = flight.getActualArrivalDay();
			int flightHour = flight.getActualArrivalHour();
			return (flightDay > day) || (flightDay == day && flightHour >= hour);
		}).toList();
	}

	@Override
	public List<Flight> getAllScheduledFlights() {
		return cachedFlights;
	}
}