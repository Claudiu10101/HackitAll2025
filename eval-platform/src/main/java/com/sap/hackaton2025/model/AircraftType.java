package com.sap.hackaton2025.model;

import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "aircraft_type")
public class AircraftType {

	@Id
	@GeneratedValue(strategy = GenerationType.UUID)
	private UUID id;

	@Column(name = "type_code", unique = true, nullable = false)
	private String typeName;

	@Column(name = "first_class_seats")
	private int firstClassSeats;

	@Column(name = "business_seats")
	private int businessSeats;

	@Column(name = "premium_economy_seats")
	private int premiumEconomySeats;

	@Column(name = "economy_seats")
	private int economySeats;

	@Column(name = "cost_per_kg_per_km")
	private double costPerKgPerKm;

	@Column(name = "first_class_kits_capacity")
	private int firstClassKitsCapacity;

	@Column(name = "business_kits_capacity")
	private int businessKitsCapacity;

	@Column(name = "premium_economy_kits_capacity")
	private int premiumEconomyKitsCapacity;

	@Column(name = "economy_kits_capacity")
	private int economyKitsCapacity;

	// Constructors
	public AircraftType() {
	}

	// Getters and setters
	public UUID getId() {
		return id;
	}

	public void setId(UUID id) {
		this.id = id;
	}

	public String getTypeName() {
		return typeName;
	}

	public void setTypeName(String typeName) {
		this.typeName = typeName;
	}

	public int getFirstClassSeats() {
		return firstClassSeats;
	}

	public void setFirstClassSeats(int firstClassSeats) {
		this.firstClassSeats = firstClassSeats;
	}

	public int getBusinessSeats() {
		return businessSeats;
	}

	public void setBusinessSeats(int businessSeats) {
		this.businessSeats = businessSeats;
	}

	public int getPremiumEconomySeats() {
		return premiumEconomySeats;
	}

	public void setPremiumEconomySeats(int premiumEconomySeats) {
		this.premiumEconomySeats = premiumEconomySeats;
	}

	public int getEconomySeats() {
		return economySeats;
	}

	public void setEconomySeats(int economySeats) {
		this.economySeats = economySeats;
	}

	public double getCostPerKgPerKm() {
		return costPerKgPerKm;
	}

	public void setCostPerKgPerKm(double costPerKgPerKm) {
		this.costPerKgPerKm = costPerKgPerKm;
	}

	public int getTotalSeats() {
		return firstClassSeats + businessSeats + premiumEconomySeats + economySeats;
	}

	public int getFirstClassKitsCapacity() {
		return firstClassKitsCapacity;
	}

	public void setFirstClassKitsCapacity(int firstClassKitsCapacity) {
		this.firstClassKitsCapacity = firstClassKitsCapacity;
	}

	public int getBusinessKitsCapacity() {
		return businessKitsCapacity;
	}

	public void setBusinessKitsCapacity(int businessKitsCapacity) {
		this.businessKitsCapacity = businessKitsCapacity;
	}

	public int getPremiumEconomyKitsCapacity() {
		return premiumEconomyKitsCapacity;
	}

	public void setPremiumEconomyKitsCapacity(int premiumEconomyKitsCapacity) {
		this.premiumEconomyKitsCapacity = premiumEconomyKitsCapacity;
	}

	public int getEconomyKitsCapacity() {
		return economyKitsCapacity;
	}

	public void setEconomyKitsCapacity(int economyKitsCapacity) {
		this.economyKitsCapacity = economyKitsCapacity;
	}
}