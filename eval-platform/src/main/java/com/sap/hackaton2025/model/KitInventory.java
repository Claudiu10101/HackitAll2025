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
@Table(name = "kit_inventory")
public class KitInventory {

	@Id
	@GeneratedValue(strategy = GenerationType.UUID)
	private UUID id;

	@Column(name = "airport_id", nullable = false, insertable = true, updatable = false)
	private UUID airportId;

	@Column(name = "eval_session_id", nullable = false, insertable = true, updatable = false)
	private UUID evaluationSessionId;

	@Enumerated(EnumType.STRING)
	@Column(name = "kit_type", nullable = false, insertable = true, updatable = false)
	private KitType kitType;

	@Column(name = "available_kits", nullable = false)
	private int availableKits;

	@Column(name = "capacity", nullable = false)
	private int capacity;

	public UUID getId() {
		return id;
	}

	public void setId(UUID id) {
		this.id = id;
	}

	public KitType getKitType() {
		return kitType;
	}

	public void setKitType(KitType kitType) {
		this.kitType = kitType;
	}

	public int getAvailableKits() {
		return availableKits;
	}

	public void setAvailableKits(int availableKits) {
		this.availableKits = availableKits;
	}

	public int getCapacity() {
		return capacity;
	}

	public void setCapacity(int capacity) {
		this.capacity = capacity;
	}

	public UUID getAirportId() {
		return airportId;
	}

	public void setAirportId(UUID airportId) {
		this.airportId = airportId;
	}

	public UUID getEvaluationSessionId() {
		return evaluationSessionId;
	}

	public void setEvaluationSessionId(UUID evaluationSessionId) {
		this.evaluationSessionId = evaluationSessionId;
	}

}