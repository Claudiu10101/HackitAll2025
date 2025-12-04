package com.sap.hackaton2025.controller.dto;

import java.util.UUID;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(description = "Represents an event related to a flight, including its details and status.")
public record FlightEvent(@Schema(description = "The type of flight event", nullable = false) FlightEventType eventType,
		@Schema(description = "The flight number", nullable = false) String flightNumber, UUID flightId,
		@Schema(description = "The code of the origin airport", nullable = false) String originAirport,
		@Schema(description = "The code of the destination airport", nullable = false) String destinationAirport,
		@Schema(description = "The departure time", nullable = false) ReferenceHour departure,
		@Schema(description = "The arrival time", nullable = false) ReferenceHour arrival,
		@Schema(description = "The number of passengers per class", nullable = false) PerClassAmount passengers,
		@Schema(description = "The code of the type of aircraft", nullable = false) String aircraftType) {

}