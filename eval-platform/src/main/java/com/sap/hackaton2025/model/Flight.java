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
@Table(name = "flight")
public class Flight {

	@Id
	@GeneratedValue(strategy = GenerationType.UUID)
	private UUID id;

	@Column(name = "flight_number", nullable = false)
	private String flightNumber;

	@ManyToOne(optional = false)
	@JoinColumn(name = "origin_airport_id")
	private Airport originAirport;

	@ManyToOne(optional = false)
	@JoinColumn(name = "destination_airport_id")
	private Airport destinationAirport;

	@ManyToOne(optional = false)
	@JoinColumn(name = "sched_aircraft_type_id")
	private AircraftType scheduledAircraftType;
	
	@ManyToOne(optional = false)
	@JoinColumn(name = "act_aircraft_type_id")
	private AircraftType actualAircraftType;

	@Column(name = "scheduled_depart_day")
	private int scheduledDepartDay;
	@Column(name = "scheduled_depart_hour")
	private int scheduledDepartHour;

	@Column(name = "scheduled_arrival_day")
	private int scheduledArrivalDay;
	@Column(name = "scheduled_arrival_hour")
	private int scheduledArrivalHour;

	@Column(name = "distance")
	private double distance;

	@Column(name = "actual_distance")
	private double actualDistance;

	@Column(name = "actual_arival_day")
	private int actualArrivalDay;

	@Column(name = "actual_arrival_hour")
	private int actualArrivalHour;

	@Column(name = "planned_first_passengers")
	private int plannedFirstPassengers;

	@Column(name = "planned_business_passengers")
	private int plannedBusinessPassengers;

	@Column(name = "planned_premium_economy_passengers")
	private int plannedPremiumEconomyPassengers;

	@Column(name = "planned_economy_passengers")
	private int plannedEconomyPassengers;

	@Column(name = "actual_first_passengers")
	private int actualFirstPassengers;

	@Column(name = "actual_business_passengers")
	private int actualBusinessPassengers;

	@Column(name = "actual_premium_economy_passengers")
	private int actualPremiumEconomyPassengers;

	@Column(name = "actual_economy_passengers")
	private int actualEconomyPassengers;

	public UUID getId() {
		return id;
	}

	public void setId(UUID id) {
		this.id = id;
	}

	public String getFlightNumber() {
		return flightNumber;
	}

	public void setFlightNumber(String flightNumber) {
		this.flightNumber = flightNumber;
	}

	public Airport getOriginAirport() {
		return originAirport;
	}

	public void setOriginAirport(Airport originAirport) {
		this.originAirport = originAirport;
	}

	public Airport getDestinationAirport() {
		return destinationAirport;
	}

	public void setDestinationAirport(Airport destinationAirport) {
		this.destinationAirport = destinationAirport;
	}

	

	public int getScheduledDepartDay() {
		return scheduledDepartDay;
	}

	public void setScheduledDepartDay(int scheduledDepartDay) {
		this.scheduledDepartDay = scheduledDepartDay;
	}

	public int getScheduledDepartHour() {
		return scheduledDepartHour;
	}

	public void setScheduledDepartHour(int scheduledDepartHour) {
		this.scheduledDepartHour = scheduledDepartHour;
	}

	public int getScheduledArrivalDay() {
		return scheduledArrivalDay;
	}

	public void setScheduledArrivalDay(int scheduledArrivalDay) {
		this.scheduledArrivalDay = scheduledArrivalDay;
	}

	public int getScheduledArrivalHour() {
		return scheduledArrivalHour;
	}

	public void setScheduledArrivalHour(int scheduledArrivalHour) {
		this.scheduledArrivalHour = scheduledArrivalHour;
	}

	public double getDistance() {
		return distance;
	}

	public void setDistance(double distance) {
		this.distance = distance;
	}

	public double getActualDistance() {
		return actualDistance;
	}

	public void setActualDistance(double actualDistance) {
		this.actualDistance = actualDistance;
	}

	public int getActualArrivalDay() {
		return actualArrivalDay;
	}

	public void setActualArrivalDay(int actualArrivalDay) {
		this.actualArrivalDay = actualArrivalDay;
	}

	public int getActualArrivalHour() {
		return actualArrivalHour;
	}

	public void setActualArrivalHour(int actualArrivalHour) {
		this.actualArrivalHour = actualArrivalHour;
	}

	public int getPlannedFirstPassengers() {
		return plannedFirstPassengers;
	}

	public void setPlannedFirstPassengers(int plannedFirstPassengers) {
		this.plannedFirstPassengers = plannedFirstPassengers;
	}

	public int getPlannedBusinessPassengers() {
		return plannedBusinessPassengers;
	}

	public void setPlannedBusinessPassengers(int plannedBusinessPassengers) {
		this.plannedBusinessPassengers = plannedBusinessPassengers;
	}

	public int getPlannedPremiumEconomyPassengers() {
		return plannedPremiumEconomyPassengers;
	}

	public void setPlannedPremiumEconomyPassengers(int plannedPremiumEconomyPassengers) {
		this.plannedPremiumEconomyPassengers = plannedPremiumEconomyPassengers;
	}

	public int getPlannedEconomyPassengers() {
		return plannedEconomyPassengers;
	}

	public void setPlannedEconomyPassengers(int plannedEconomyPassengers) {
		this.plannedEconomyPassengers = plannedEconomyPassengers;
	}

	public int getActualFirstPassengers() {
		return actualFirstPassengers;
	}

	public void setActualFirstPassengers(int actualFirstPassengers) {
		this.actualFirstPassengers = actualFirstPassengers;
	}

	public int getActualBusinessPassengers() {
		return actualBusinessPassengers;
	}

	public void setActualBusinessPassengers(int actualBusinessPassengers) {
		this.actualBusinessPassengers = actualBusinessPassengers;
	}

	public int getActualPremiumEconomyPassengers() {
		return actualPremiumEconomyPassengers;
	}

	public void setActualPremiumEconomyPassengers(int actualPremiumEconomyPassengers) {
		this.actualPremiumEconomyPassengers = actualPremiumEconomyPassengers;
	}

	public int getActualEconomyPassengers() {
		return actualEconomyPassengers;
	}

	public void setActualEconomyPassengers(int actualEconomyPassengers) {
		this.actualEconomyPassengers = actualEconomyPassengers;
	}

	public AircraftType getScheduledAircraftType() {
		return scheduledAircraftType;
	}

	public void setScheduledAircraftType(AircraftType scheduledAircraftType) {
		this.scheduledAircraftType = scheduledAircraftType;
	}

	public AircraftType getActualAircraftType() {
		return actualAircraftType;
	}

	public void setActualAircraftType(AircraftType actualAircraftType) {
		this.actualAircraftType = actualAircraftType;
	}

}