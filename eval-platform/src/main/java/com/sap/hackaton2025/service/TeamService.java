package com.sap.hackaton2025.service;

import java.util.Optional;
import java.util.UUID;

import com.sap.hackaton2025.model.Team;

public interface TeamService {
	Optional<Team> getTeamByApiKey(UUID apiKey);
}
 