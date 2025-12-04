package com.sap.hackaton2025.service;

import java.util.List;
import java.util.UUID;

import com.sap.hackaton2025.controller.dto.FlightLoadDto;
import com.sap.hackaton2025.controller.dto.HourResponseDto;
import com.sap.hackaton2025.controller.dto.PerClassAmount;

public interface SessionService {
	UUID createSessionForApiKey(UUID teamId);

	HourResponseDto playRound(UUID sessionId, int day, int hour, List<FlightLoadDto> flightLoads,
			PerClassAmount kitPurchasingOrders);

	HourResponseDto stopSession(UUID apiKey);

}