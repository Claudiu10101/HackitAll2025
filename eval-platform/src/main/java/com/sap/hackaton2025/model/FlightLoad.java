package com.sap.hackaton2025.model;

import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;

@Entity
@Table(name = "flight_load")
public class FlightLoad {
	@Id
	@GeneratedValue(strategy = GenerationType.UUID)
	private UUID id;

	@ManyToOne
	@JoinColumn(name = "evaluation_session_id")
	private EvaluationSession evaluationSession;

	@Column(name = "flight_id")
	private UUID flightId;

	@Column(name = "first_kits")
	private int firstKits;

	@Column(name = "business_kits")
	private int businessKits;

	@Column(name = "premium_economy_kits")
	private int premiumEconomyKits;

	@Column(name = "economy_kits")
	private int economyKits;

	public UUID getId() {
		return id;
	}

	public void setId(UUID id) {
		this.id = id;
	}

	public EvaluationSession getEvaluationSession() {
		return evaluationSession;
	}

	public void setEvaluationSession(EvaluationSession session) {
		this.evaluationSession = session;
	}

	public UUID getFlightId() {
		return flightId;
	}

	public void setFlightId(UUID flightId) {
		this.flightId = flightId;
	}

	public int getFirstKits() {
		return firstKits;
	}

	public void setFirstKits(int firstClassKits) {
		this.firstKits = firstClassKits;
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

}
