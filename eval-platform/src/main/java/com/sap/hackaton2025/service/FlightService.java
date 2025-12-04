package com.sap.hackaton2025.service;

import java.util.List;
import java.util.UUID;

import com.sap.hackaton2025.model.Flight;

public interface FlightService {

	Flight getFlightById(UUID flightId);

	List<Flight> getScheduledFlights(int day, int hour);

	List<Flight> getCheckedInFlights(int day, int hour);

	List<Flight> getLandedFlights(int day, int hour);

	List<Flight> getUpcomingFlights(int day, int hour);

	List<Flight> getAllScheduledFlights();
}