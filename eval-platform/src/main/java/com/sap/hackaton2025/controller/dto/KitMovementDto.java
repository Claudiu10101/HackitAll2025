package com.sap.hackaton2025.controller.dto;

import java.util.UUID;

import com.sap.hackaton2025.model.KitType;

public class KitMovementDto {
    
    private UUID fromAirportId;
    private UUID toAirportId;
    private UUID flightId;
    private KitType kitType;
    private int quantity;
    private double weightKg;
    private double additionalWeightCost;

    public KitMovementDto() {}

    // Getters and setters
    public UUID getFromAirportId() { return fromAirportId; }
    public void setFromAirportId(UUID fromAirportId) { this.fromAirportId = fromAirportId; }

    public UUID getToAirportId() { return toAirportId; }
    public void setToAirportId(UUID toAirportId) { this.toAirportId = toAirportId; }

    public UUID getFlightId() { return flightId; }
    public void setFlightId(UUID flightId) { this.flightId = flightId; }

    public KitType getKitType() { return kitType; }
    public void setKitType(KitType kitType) { this.kitType = kitType; }

    public int getQuantity() { return quantity; }
    public void setQuantity(int quantity) { this.quantity = quantity; }

    public double getWeightKg() { return weightKg; }
    public void setWeightKg(double weightKg) { this.weightKg = weightKg; }

    public double getAdditionalWeightCost() { return additionalWeightCost; }
    public void setAdditionalWeightCost(double additionalWeightCost) { this.additionalWeightCost = additionalWeightCost; }
}