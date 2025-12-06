package com.sap.hackaton2025.controller.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;

@JsonIgnoreProperties(ignoreUnknown = true)
@Schema(description = "Object representing the amount per class")
public record PerClassAmount(
		@Schema(description = "Amount for first class", minimum = "0", maximum = "42000", nullable = false) @Min(0) @Max(42000) int first,
		@Schema(description = "Amount for business class", minimum = "0", maximum = "42000", nullable = false) @Min(0) @Max(42000) int business,
		@Schema(description = "Amount for premium economy class", minimum = "0", maximum = "42000", nullable = false) @Min(0) @Max(42000) int premiumEconomy,
		@Schema(description = "Amount for economy class", minimum = "0", maximum = "42000", nullable = false) @Min(0) @Max(42000) int economy) {

}
