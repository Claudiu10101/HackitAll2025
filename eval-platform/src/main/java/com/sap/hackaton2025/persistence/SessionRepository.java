package com.sap.hackaton2025.persistence;

import java.util.Optional;
import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import com.sap.hackaton2025.model.EvaluationSession;

public interface SessionRepository extends JpaRepository<EvaluationSession, UUID> {

	boolean existsByTeamIdAndEndTimeIsNull(UUID teamId);

	@Query("SELECT s FROM EvaluationSession s WHERE s.team.apiKey = :apiKey AND s.endTime IS NULL")
	Optional<EvaluationSession> findByApiKey(UUID apiKey);

}
