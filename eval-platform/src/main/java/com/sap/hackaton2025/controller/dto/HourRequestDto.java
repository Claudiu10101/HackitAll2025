package com.sap.hackaton2025.controller.dto;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;

@JsonIgnoreProperties(ignoreUnknown = true)
@Schema(description = "Payload for hourly flight load and kit purchasing orders")
public record HourRequestDto(
		@Schema(description = "The current day", maximum = "50", minimum = "0", nullable = false) @Min(0) @Max(50) int day,
		@Schema(description = "The current hour", minimum = "0", maximum = "23", nullable = false) @Min(0) @Max(23) int hour,
		@Schema(description = "The kit load instructions for flights") @Valid List<FlightLoadDto> flightLoads,
		@Schema(description = "The order for new kits, per class") @Valid PerClassAmount kitPurchasingOrders) {

}