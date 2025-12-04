package com.sap.hackaton2025.controller.dto;

import java.util.UUID;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;

@JsonIgnoreProperties(ignoreUnknown = true)
@Schema(description = "Object representing the load of kits for a specific flight")
public record FlightLoadDto(
		@Schema(description = "The unique identifier of the flight, as UUID", nullable = false) @NotNull UUID flightId,
		@Schema(description = "The number of kits to be loaded per class", nullable = false) @NotNull @Valid PerClassAmount loadedKits) {

}
