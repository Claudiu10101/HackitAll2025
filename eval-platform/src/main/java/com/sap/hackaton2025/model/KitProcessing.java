package com.sap.hackaton2025.model;

import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "kit_processing")
public class KitProcessing {
	@Id
	@GeneratedValue(strategy = GenerationType.UUID)
	private UUID id;

	@Column(name = "evaluation_session_id", nullable = false, insertable = true, updatable = false)
	private UUID evaluationSessionId;

	@Column(name = "processed")
	private boolean processed;

	@Column(name = "available_day")
	private int availableDay;

	@Column(name = "available_hour")
	private int availableHour;

	@Column(name = "quantity")
	private int quantity;

	@Column(name = "remaining_quantity")
	private int remainingQuantity;

	@Column(name = "kit_type")
	@Enumerated(EnumType.STRING)
	private KitType kitType;

	@Column(name = "airport_id", nullable = false, insertable = true, updatable = false)
	private UUID airportId;

	public UUID getId() {
		return id;
	}

	public void setId(UUID id) {
		this.id = id;
	}

	public boolean isProcessed() {
		return processed;
	}

	public void setProcessed(boolean processed) {
		this.processed = processed;
	}

	public int getQuantity() {
		return quantity;
	}

	public void setQuantity(int quantity) {
		this.quantity = quantity;
	}

	public KitType getKitType() {
		return kitType;
	}

	public void setKitType(KitType kitType) {
		this.kitType = kitType;
	}

	public int getAvailableDay() {
		return availableDay;
	}

	public void setAvailableDay(int availableDay) {
		this.availableDay = availableDay;
	}

	public int getAvailableHour() {
		return availableHour;
	}

	public void setAvailableHour(int availableHour) {
		this.availableHour = availableHour;
	}

	public int getRemainingQuantity() {
		return remainingQuantity;
	}

	public void setRemainingQuantity(int remainingQuantity) {
		this.remainingQuantity = remainingQuantity;
	}

	public UUID getEvaluationSessionId() {
		return evaluationSessionId;
	}

	public void setEvaluationSessionId(UUID evaluationSessionId) {
		this.evaluationSessionId = evaluationSessionId;
	}

	public UUID getAirportId() {
		return airportId;
	}

	public void setAirportId(UUID airportId) {
		this.airportId = airportId;
	}

}
