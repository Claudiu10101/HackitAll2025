package com.sap.hackaton2025.model;

import java.time.LocalDateTime;
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
@Table(name = "penalty")
public class Penalty {

	@Id
	@GeneratedValue(strategy = GenerationType.UUID)
	@Column(name = "id", nullable = false, updatable = false)
	private UUID id;

	@ManyToOne(optional = false)
	@JoinColumn(name = "session_id")
	private EvaluationSession session;

	@Column(name = "penalty_type")
	private String type;

	@ManyToOne(optional = true)
	@JoinColumn(name = "flight_id")
	private Flight flight;

	@Column(name = "cost")
	private Double cost;

	@Column(name = "issued_day")
	private int day;

	@Column(name = "issued_hour")
	private int hour;

	@Column(name = "message")
	private String message;

	@Column(name = "processed")
	private boolean processed;

	@Column(name = "created_at")
	private LocalDateTime createdAt;

	public UUID getId() {
		return id;
	}

	public void setId(UUID id) {
		this.id = id;
	}

	public EvaluationSession getSession() {
		return session;
	}

	public void setSession(EvaluationSession session) {
		this.session = session;
	}

	public String getType() {
		return type;
	}

	public void setType(String type) {
		this.type = type;
	}

	public Flight getFlight() {
		return flight;
	}

	public void setFlight(Flight flight) {
		this.flight = flight;
	}

	public Double getCost() {
		return cost;
	}

	public void setCost(Double cost) {
		this.cost = cost;
	}

	public int getDay() {
		return day;
	}

	public void setDay(int day) {
		this.day = day;
	}

	public String getMessage() {
		return message;
	}

	public void setMessage(String message) {
		this.message = message;
	}

	public boolean isProcessed() {
		return processed;
	}

	public void setProcessed(boolean processed) {
		this.processed = processed;
	}

	public int getHour() {
		return hour;
	}

	public void setHour(int hour) {
		this.hour = hour;
	}

}
