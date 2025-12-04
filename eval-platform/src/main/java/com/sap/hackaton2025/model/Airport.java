package com.sap.hackaton2025.model;

import java.util.Objects;
import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "airport")
public class Airport {

	@Id
	@GeneratedValue(strategy = GenerationType.UUID)
	private UUID id;

	@Column(name = "code", unique = true, nullable = false)
	private String code;

	@Column(name = "name")
	private String name;

	@Column(name = "first_processing_time")
	private int firstProcessingTimeHours;

	@Column(name = "business_processing_time")
	private int businessProcessingTimeHours;

	@Column(name = "premium_economy_processing_time")
	private int premiumEconomyProcessingTimeHours;

	@Column(name = "economy_processing_time")
	private int economyProcessingTimeHours;

	@Column(name = "first_processing_cost")
	private double firstProcessingCost;

	@Column(name = "business_processing_cost")
	private double businessProcessingCost;

	@Column(name = "premium_economy_processing_cost")
	private double premiumEconomyProcessingCost;

	@Column(name = "economy_processing_cost")
	private double economyProcessingCost;

	@Column(name = "first_loading_cost")
	private double firstLoadingCost;

	@Column(name = "business_loading_cost")
	private double businessLoadingCost;

	@Column(name = "premium_economy_loading_cost")
	private double premiumEconomyLoadingCost;

	@Column(name = "economy_loading_cost")
	private double economyLoadingCost;

	@Column(name = "initial_fc_stock")
	private int initialFCStock;

	@Column(name = "initial_bc_stock")
	private int initialBCStock;

	@Column(name = "initial_pe_stock")
	private int initialPEStock;

	@Column(name = "initial_ec_stock")
	private int initialECStock;

	@Column(name = "capacity_fc")
	private int capacityFC;
	@Column(name = "capacity_bc")
	private int capacityBC;
	@Column(name = "capacity_pe")
	private int capacityPE;
	@Column(name = "capacity_ec")
	private int capacityEC;

	public UUID getId() {
		return id;
	}

	public void setId(UUID id) {
		this.id = id;
	}

	public String getCode() {
		return code;
	}

	public void setCode(String code) {
		this.code = code;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public int getFirstProcessingTimeHours() {
		return firstProcessingTimeHours;
	}

	public void setFirstProcessingTimeHours(int firstProcessingTimeHours) {
		this.firstProcessingTimeHours = firstProcessingTimeHours;
	}

	public int getBusinessProcessingTimeHours() {
		return businessProcessingTimeHours;
	}

	public void setBusinessProcessingTimeHours(int businessProcessingTimeHours) {
		this.businessProcessingTimeHours = businessProcessingTimeHours;
	}

	public int getPremiumEconomyProcessingTimeHours() {
		return premiumEconomyProcessingTimeHours;
	}

	public void setPremiumEconomyProcessingTimeHours(int premiumEconomyProcessingTimeHours) {
		this.premiumEconomyProcessingTimeHours = premiumEconomyProcessingTimeHours;
	}

	public int getEconomyProcessingTimeHours() {
		return economyProcessingTimeHours;
	}

	public void setEconomyProcessingTimeHours(int economyProcessingTimeHours) {
		this.economyProcessingTimeHours = economyProcessingTimeHours;
	}

	public double getFirstProcessingCost() {
		return firstProcessingCost;
	}

	public void setFirstProcessingCost(double firstProcessingCost) {
		this.firstProcessingCost = firstProcessingCost;
	}

	public double getBusinessProcessingCost() {
		return businessProcessingCost;
	}

	public void setBusinessProcessingCost(double businessProcessingCost) {
		this.businessProcessingCost = businessProcessingCost;
	}

	public double getPremiumEconomyProcessingCost() {
		return premiumEconomyProcessingCost;
	}

	public void setPremiumEconomyProcessingCost(double premiumEconomyProcessingCost) {
		this.premiumEconomyProcessingCost = premiumEconomyProcessingCost;
	}

	public double getEconomyProcessingCost() {
		return economyProcessingCost;
	}

	public void setEconomyProcessingCost(double economyProcessingCost) {
		this.economyProcessingCost = economyProcessingCost;
	}

	public double getFirstLoadingCost() {
		return firstLoadingCost;
	}

	public void setFirstLoadingCost(double firstLoadingCost) {
		this.firstLoadingCost = firstLoadingCost;
	}

	public double getBusinessLoadingCost() {
		return businessLoadingCost;
	}

	public void setBusinessLoadingCost(double businessLoadingCost) {
		this.businessLoadingCost = businessLoadingCost;
	}

	public double getPremiumEconomyLoadingCost() {
		return premiumEconomyLoadingCost;
	}

	public void setPremiumEconomyLoadingCost(double premiumEconomyLoadingCost) {
		this.premiumEconomyLoadingCost = premiumEconomyLoadingCost;
	}

	public double getEconomyLoadingCost() {
		return economyLoadingCost;
	}

	public void setEconomyLoadingCost(double economyLoadingCost) {
		this.economyLoadingCost = economyLoadingCost;
	}

	public int getInitialFCStock() {
		return initialFCStock;
	}

	public void setInitialFCStock(int initialFCStock) {
		this.initialFCStock = initialFCStock;
	}

	public int getInitialBCStock() {
		return initialBCStock;
	}

	public void setInitialBCStock(int initialBCStock) {
		this.initialBCStock = initialBCStock;
	}

	public int getInitialPEStock() {
		return initialPEStock;
	}

	public void setInitialPEStock(int initialPEStock) {
		this.initialPEStock = initialPEStock;
	}

	public int getInitialECStock() {
		return initialECStock;
	}

	public void setInitialECStock(int initialECStock) {
		this.initialECStock = initialECStock;
	}

	public int getCapacityFC() {
		return capacityFC;
	}

	public void setCapacityFC(int capacityFC) {
		this.capacityFC = capacityFC;
	}

	public int getCapacityBC() {
		return capacityBC;
	}

	public void setCapacityBC(int capacityBC) {
		this.capacityBC = capacityBC;
	}

	public int getCapacityPE() {
		return capacityPE;
	}

	public void setCapacityPE(int capacityPE) {
		this.capacityPE = capacityPE;
	}

	public int getCapacityEC() {
		return capacityEC;
	}

	public void setCapacityEC(int capacityEC) {
		this.capacityEC = capacityEC;
	}

	@Override
	public int hashCode() {
		return Objects.hash(code);
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Airport other = (Airport) obj;
		return Objects.equals(code, other.code);
	}

}