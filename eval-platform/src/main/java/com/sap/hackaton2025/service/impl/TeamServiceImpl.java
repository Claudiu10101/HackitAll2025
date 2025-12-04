package com.sap.hackaton2025.service.impl;

import java.util.Optional;
import java.util.UUID;

import org.springframework.stereotype.Service;

import com.sap.hackaton2025.model.Team;
import com.sap.hackaton2025.persistence.TeamRepository;
import com.sap.hackaton2025.service.TeamService;

@Service
public class TeamServiceImpl implements TeamService {
	
	private final TeamRepository teamRepository;
	
	TeamServiceImpl(TeamRepository teamRepository) {
        this.teamRepository = teamRepository;
	}

	@Override
	public Optional<Team> getTeamByApiKey(UUID apiKey) {
		return teamRepository.findByApiKey(apiKey);
	}

}
