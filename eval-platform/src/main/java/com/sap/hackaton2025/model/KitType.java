package com.sap.hackaton2025.model;

public enum KitType {
	A_FIRST_CLASS(5d, 200d, 48), B_BUSINESS(3d, 150d, 36), C_PREMIUM_ECONOMY(2.5d, 100d, 24), D_ECONOMY(1.5d, 50d, 12);

	private double weight;
	private double kitCost;
	private int replacementLeadTime;

	KitType(double weight, double kitCost, int replacementLeadTime) {
		this.weight = weight;
		this.kitCost = kitCost;
		this.replacementLeadTime = replacementLeadTime;
	}

	public double cost() {
		return kitCost;
	}

	public double weightKg() {
		return weight;
	}

	public int replacementLeadTimeHours() {
		return replacementLeadTime;
	}
}