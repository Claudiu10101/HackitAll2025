package com.sap.hackaton2025.controller.dto;

import java.util.List;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(description = "Payload for hourly flight updates and penalties incurred")
public record HourResponseDto(@Schema(description = "The current day", nullable = false) int day,
		@Schema(description = "The current hour", nullable = false) int hour,
		@Schema(description = "The updates for flights during the hour") List<FlightEvent> flightUpdates,
		@Schema(description = "The applied penalties for the hourly request") List<PenaltyDto> penalties,
		@Schema(description = "The total cost incurred, running total, including the current hour", nullable = false) double totalCost) {

}