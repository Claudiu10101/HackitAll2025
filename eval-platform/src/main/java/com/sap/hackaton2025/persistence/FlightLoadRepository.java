package com.sap.hackaton2025.persistence;

import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;

import com.sap.hackaton2025.model.FlightLoad;

public interface FlightLoadRepository extends JpaRepository<FlightLoad, UUID> {

}
