package com.sap.hackaton2025.persistence;

import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.sap.hackaton2025.model.AircraftType;

@Repository
public interface AircraftTypeRepository extends JpaRepository<AircraftType, UUID> {
    
}