package com.sap.hackaton2025.model;

import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "kit_movement")
public class KitMovement {

	@Id
	@GeneratedValue(strategy = GenerationType.UUID)
	private UUID id;

	@Column(name = "evaluation_session_id")
	private UUID evaluationSessionId;

	@Column(name = "movement_day")
	private int day;

	@Column(name = "movement_hour")
	private int hour;

	@Column(name = "first_kits")
	private int firstKits;

	@Column(name = "business_kits")
	private int businessKits;

	@Column(name = "premium_economy_kits")
	private int premiumEconomyKits;

	@Column(name = "economy_kits")
	private int economyKits;

	@Column(name = "flight_id", insertable = true, updatable = false)
	private UUID flightId;

	@Column(name = "airport_id", insertable = true, updatable = false)
	private UUID airportId;

	@Column(name = "cost_value")
	private double cost;

	public UUID getId() {
		return id;
	}

	public void setId(UUID id) {
		this.id = id;
	}

	public int getDay() {
		return day;
	}

	public void setDay(int day) {
		this.day = day;
	}

	public int getHour() {
		return hour;
	}

	public void setHour(int hour) {
		this.hour = hour;
	}

	public int getFirstKits() {
		return firstKits;
	}

	public void setFirstKits(int firstKits) {
		this.firstKits = firstKits;
	}

	public int getBusinessKits() {
		return businessKits;
	}

	public void setBusinessKits(int businessKits) {
		this.businessKits = businessKits;
	}

	public int getPremiumEconomyKits() {
		return premiumEconomyKits;
	}

	public void setPremiumEconomyKits(int premiumEconomyKits) {
		this.premiumEconomyKits = premiumEconomyKits;
	}

	public int getEconomyKits() {
		return economyKits;
	}

	public void setEconomyKits(int economyKits) {
		this.economyKits = economyKits;
	}

	public double getCost() {
		return cost;
	}

	public void setCost(double cost) {
		this.cost = cost;
	}

	public UUID getEvaluationSessionId() {
		return evaluationSessionId;
	}

	public void setEvaluationSessionId(UUID evaluationSessionId) {
		this.evaluationSessionId = evaluationSessionId;
	}

	public UUID getFlightId() {
		return flightId;
	}

	public void setFlightId(UUID flightId) {
		this.flightId = flightId;
	}

	public UUID getAirportId() {
		return airportId;
	}

	public void setAirportId(UUID airportId) {
		this.airportId = airportId;
	}

}
