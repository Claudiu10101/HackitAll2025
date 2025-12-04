package com.sap.hackaton2025.persistence;

import java.util.List;
import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import com.sap.hackaton2025.model.KitInventory;

@Repository
public interface KitInventoryRepository extends JpaRepository<KitInventory, UUID> {

	@Query("SELECT k FROM KitInventory k WHERE k.evaluationSessionId = :sessionId")
	List<KitInventory> findBySessionId(UUID sessionId);

}