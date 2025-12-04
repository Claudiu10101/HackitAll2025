package com.sap.hackaton2025.controller.dto;

import java.util.UUID;

import com.sap.hackaton2025.model.KitType;

public class KitInventoryDto {
    
    private UUID airportId;
    private String airportCode;
    private KitType kitType;
    private int availableKits;
    private int kitsBeingWashed;
    private int kitsBeingReplaced;
    private int unusableKits;
    private int safetyStock;

    public KitInventoryDto() {}

    // Getters and setters
    public UUID getAirportId() { return airportId; }
    public void setAirportId(UUID airportId) { this.airportId = airportId; }

    public String getAirportCode() { return airportCode; }
    public void setAirportCode(String airportCode) { this.airportCode = airportCode; }

    public KitType getKitType() { return kitType; }
    public void setKitType(KitType kitType) { this.kitType = kitType; }

    public int getAvailableKits() { return availableKits; }
    public void setAvailableKits(int availableKits) { this.availableKits = availableKits; }

    public int getKitsBeingWashed() { return kitsBeingWashed; }
    public void setKitsBeingWashed(int kitsBeingWashed) { this.kitsBeingWashed = kitsBeingWashed; }

    public int getKitsBeingReplaced() { return kitsBeingReplaced; }
    public void setKitsBeingReplaced(int kitsBeingReplaced) { this.kitsBeingReplaced = kitsBeingReplaced; }

    public int getUnusableKits() { return unusableKits; }
    public void setUnusableKits(int unusableKits) { this.unusableKits = unusableKits; }

    public int getSafetyStock() { return safetyStock; }
    public void setSafetyStock(int safetyStock) { this.safetyStock = safetyStock; }

    public int getTotalKits() {
        return availableKits + kitsBeingWashed + kitsBeingReplaced + unusableKits;
    }
}