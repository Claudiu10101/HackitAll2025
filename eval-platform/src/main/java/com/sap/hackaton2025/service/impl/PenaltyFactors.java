package com.sap.hackaton2025.service.impl;

public final class PenaltyFactors {

	public static final double FLIGHT_OVERLOAD_FACTOR_PER_DISTANCE = 5d;
	public static final double UNFULFILLED_KIT_FACTOR_PER_DISTANCE = 0.003d;
	public static final double INCORRECT_FLIGHT_LOAD = 5000d;
	public static final double NEGATIVE_INVENTORY = 5342d;
	public static final double OVER_CAPACITY_STOCK = 777d;
	public static final double END_OF_GAME_REMAINING_STOCK = 0.0013d;
	public static final double EARLY_END_OF_GAME = 1000d;
	public static final double END_OF_GAME_PENDING_KIT_PROCESSING =  0.0013d;
	public static final double END_OF_GAME_UNFULFILLED_FLIGHT_KITS = 1.5d;

	private PenaltyFactors() {

	}
}
