package com.sap.hackaton2025.persistence;

import java.util.List;
import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;

import com.sap.hackaton2025.model.KitProcessing;

public interface KitProcessingRepository extends JpaRepository<KitProcessing, UUID> {

	@Query("""
			SELECT kp FROM KitProcessing kp WHERE kp.evaluationSessionId = :evaluationSessionId AND kp.processed = false
			and (kp.availableDay < :day OR (kp.availableDay = :day AND kp.availableHour <= :hour))
			order by kp.availableDay, kp.availableHour, kp.kitType
			""")
	List<KitProcessing> getAllAvailableForProcessing(UUID evaluationSessionId, int day, int hour);

	@Modifying
	@Query("""
			UPDATE KitProcessing kp SET kp.processed = true
			WHERE kp.evaluationSessionId = :evaluationSessionId AND kp.processed = false
			and (kp.availableDay < :day OR (kp.availableDay = :day AND kp.availableHour <= :hour))
			""")
	void markAsProcessed(UUID evaluationSessionId, int day, int hour);

	@Query("""
			SELECT kp FROM KitProcessing kp WHERE kp.evaluationSessionId = :evaluationSessionId AND kp.processed = false
			""")
	List<KitProcessing> getAllPendingProcessings(UUID evaluationSessionId);

}
