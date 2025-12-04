package com.sap.hackaton2025.controller.dto;

import java.util.UUID;

import com.sap.hackaton2025.model.KitType;

public class KitReplacementDto {
    
    private UUID airportId;
    private KitType kitType;
    private int quantityToReplace;
    private int leadTimeHours;
    private double replacementCost;

    public KitReplacementDto() {}

    // Getters and setters
    public UUID getAirportId() { return airportId; }
    public void setAirportId(UUID airportId) { this.airportId = airportId; }

    public KitType getKitType() { return kitType; }
    public void setKitType(KitType kitType) { this.kitType = kitType; }

    public int getQuantityToReplace() { return quantityToReplace; }
    public void setQuantityToReplace(int quantityToReplace) { this.quantityToReplace = quantityToReplace; }

    public int getLeadTimeHours() { return leadTimeHours; }
    public void setLeadTimeHours(int leadTimeHours) { this.leadTimeHours = leadTimeHours; }

    public double getReplacementCost() { return replacementCost; }
    public void setReplacementCost(double replacementCost) { this.replacementCost = replacementCost; }
}