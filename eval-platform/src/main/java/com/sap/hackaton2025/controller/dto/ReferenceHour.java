package com.sap.hackaton2025.controller.dto;

import com.fasterxml.jackson.annotation.JsonIgnore;

public record ReferenceHour(int day, int hour) {

	@JsonIgnore
	public static ReferenceHour addHours(int day, int hour, int hoursToAdd) {
		int totalHours = day * 24 + hour + hoursToAdd;
		int newDay = totalHours / 24;
		int newHour = totalHours % 24;
		return new ReferenceHour(newDay, newHour);
	}

	@JsonIgnore
	public ReferenceHour addHours(int hoursToAdd) {
		return ReferenceHour.addHours(this.day, this.hour, hoursToAdd);
	}

}
