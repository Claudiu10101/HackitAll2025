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
@Table(name = "eval_track")
public class EvaluationTrack {

	@Id
	@GeneratedValue(strategy = GenerationType.UUID)
	@Column(name = "id", nullable = false, updatable = false, insertable = false)
	private UUID id;

	@ManyToOne(optional = false)
	@JoinColumn(name = "team_id")
	private Team team;

	@Column(name = "prod_Day")
	private int prodDay;

	@Column(name = "prod_Hour")
	private int prodHour;

	@Column(name = "ops_Cost")
	private double productionCost;

	@Column(name = "penalty_Cost")
	private double penaltyCost;

	@Column(name = "total_Cost")
	private double totalCost;

	@Column(name = "latest")
	private boolean latest;

	@Column(name = "session_id")
	private UUID sessionId;

	@Column(name = "time_received")
	private LocalDateTime timeReceived;

	public UUID getId() {
		return id;
	}

	public void setId(UUID id) {
		this.id = id;
	}

	public Team getTeam() {
		return team;
	}

	public void setTeam(Team team) {
		this.team = team;
	}

	public int getProdDay() {
		return prodDay;
	}

	public void setProdDay(int prodDay) {
		this.prodDay = prodDay;
	}

	public int getProdHour() {
		return prodHour;
	}

	public void setProdHour(int prodHour) {
		this.prodHour = prodHour;
	}

	public double getProductionCost() {
		return productionCost;
	}

	public void setProductionCost(double productionCost) {
		this.productionCost = productionCost;
	}

	public double getPenaltyCost() {
		return penaltyCost;
	}

	public void setPenaltyCost(double penaltyCost) {
		this.penaltyCost = penaltyCost;
	}

	public boolean isLatest() {
		return latest;
	}

	public void setLatest(boolean latest) {
		this.latest = latest;
	}

	public UUID getSessionId() {
		return sessionId;
	}

	public void setSessionId(UUID sessionId) {
		this.sessionId = sessionId;
	}

	public LocalDateTime getTimeReceived() {
		return timeReceived;
	}

	public void setTimeReceived(LocalDateTime timeReceived) {
		this.timeReceived = timeReceived;
	}

	public double getTotalCost() {
		return totalCost;
	}

	public void setTotalCost(double totalCost) {
		this.totalCost = totalCost;
	}

}
