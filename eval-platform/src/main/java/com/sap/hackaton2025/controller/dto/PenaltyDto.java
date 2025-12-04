package com.sap.hackaton2025.controller.dto;

import java.util.UUID;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(description = "Object representing a penalty issued for a flight, kit movement or other network state, such as ovestock or negative stock.")
public record PenaltyDto(@Schema(description = "The code for the penalty", nullable = false) String code,
		@Schema(description = "The optional unique identifier of the flight associated with the penalty, as UUID", nullable = true) UUID flightId,
		@Schema(description = "The optional flight number associated with the penalty", nullable = true) String flightNumber,
		@Schema(description = "The day the penalty was issued", nullable = false) int issuedDay,
		@Schema(description = "The hour the penalty was issued", nullable = false) int issuedHour,
		@Schema(description = "The amount of the penalty", nullable = false, minimum = "0", exclusiveMinimum = true) double penalty,
		@Schema(description = "The reason for the penalty", nullable = false) String reason) {

}