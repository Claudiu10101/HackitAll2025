package com.sap.hackaton2025.persistence;

import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;

import com.sap.hackaton2025.model.Penalty;

public interface PenaltiesRepository extends JpaRepository<Penalty, UUID> {

}
