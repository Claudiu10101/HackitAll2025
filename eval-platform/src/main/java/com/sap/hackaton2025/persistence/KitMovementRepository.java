package com.sap.hackaton2025.persistence;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import com.sap.hackaton2025.model.KitMovement;

public interface KitMovementRepository extends JpaRepository<KitMovement, UUID> {

	@Query("""
			SELECT km FROM KitMovement km
			WHERE km.flightId = :flightId AND km.evaluationSessionId = :sessionId
			""")
	Optional<KitMovement> findByFlightIdAndSessionId(UUID flightId, UUID sessionId);

	@Query("""
			SELECT km FROM KitMovement km
			WHERE km.day = :day AND km.hour = :hour AND km.evaluationSessionId = :sessionId
			""")
	List<KitMovement> findByDayHourAndEvaluationSessionId(int day, int hour, UUID sessionId);

}
