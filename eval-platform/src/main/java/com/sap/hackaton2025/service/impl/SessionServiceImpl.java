package com.sap.hackaton2025.service.impl;

import java.text.MessageFormat;
import java.time.LocalDateTime;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.IntSupplier;
import java.util.function.ObjIntConsumer;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import com.sap.hackaton2025.controller.dto.FlightEvent;
import com.sap.hackaton2025.controller.dto.FlightEventType;
import com.sap.hackaton2025.controller.dto.FlightLoadDto;
import com.sap.hackaton2025.controller.dto.HourResponseDto;
import com.sap.hackaton2025.controller.dto.PenaltyDto;
import com.sap.hackaton2025.controller.dto.PerClassAmount;
import com.sap.hackaton2025.controller.dto.ReferenceHour;
import com.sap.hackaton2025.exception.BadRequestException;
import com.sap.hackaton2025.exception.BusinessException;
import com.sap.hackaton2025.exception.SessionAlreadyExistsException;
import com.sap.hackaton2025.exception.SessionNotFoundException;
import com.sap.hackaton2025.model.AircraftType;
import com.sap.hackaton2025.model.EvaluationSession;
import com.sap.hackaton2025.model.EvaluationTrack;
import com.sap.hackaton2025.model.Flight;
import com.sap.hackaton2025.model.FlightLoad;
import com.sap.hackaton2025.model.KitMovement;
import com.sap.hackaton2025.model.KitProcessing;
import com.sap.hackaton2025.model.KitType;
import com.sap.hackaton2025.model.Penalty;
import com.sap.hackaton2025.persistence.EvaluationTrackRepository;
import com.sap.hackaton2025.persistence.PenaltiesRepository;
import com.sap.hackaton2025.persistence.SessionRepository;
import com.sap.hackaton2025.service.AirportService;
import com.sap.hackaton2025.service.FlightLoadService;
import com.sap.hackaton2025.service.FlightService;
import com.sap.hackaton2025.service.KitInventoryService;
import com.sap.hackaton2025.service.KitMovementService;
import com.sap.hackaton2025.service.KitProcessingService;
import com.sap.hackaton2025.service.SessionService;
import com.sap.hackaton2025.service.TeamService;

@Service
public class SessionServiceImpl implements SessionService {

	private static final Logger logger = LoggerFactory.getLogger(SessionServiceImpl.class);

	private static final String HUB_AIRPORT_CODE = "HUB";

	private final TeamService teamService;
	private final FlightService flightService;
	private final SessionRepository sessionRepository;
	private final PenaltiesRepository penaltiesRepository;
	private final AirportService airportService;
	private final EvaluationTrackRepository evaluationTrackRepository;
	private final KitMovementService kitMovementService;
	private final KitProcessingService kitProcessingService;
	private final KitInventoryService kitInventoryService;
	private final FlightLoadService flightLoadService;

	private final int numberOfHours;

	SessionServiceImpl(TeamService teamService, SessionRepository sessionRepository, FlightService flightService,
			PenaltiesRepository penaltiesRepository, EvaluationTrackRepository evaluationTrackRepository,
			KitMovementService kitMovementService, AirportService airportService,
			KitProcessingService kitProcessingService, KitInventoryService kitInventoryService,
			FlightLoadService flightLoadService, @Value("${game.numberOfHours:720}") int numberOfHours) {
		this.teamService = teamService;
		this.sessionRepository = sessionRepository;
		this.flightService = flightService;
		this.penaltiesRepository = penaltiesRepository;
		this.evaluationTrackRepository = evaluationTrackRepository;
		this.numberOfHours = numberOfHours;
		this.kitMovementService = kitMovementService;
		this.airportService = airportService;
		this.kitProcessingService = kitProcessingService;
		this.kitInventoryService = kitInventoryService;
		this.flightLoadService = flightLoadService;
	}

	@Transactional(readOnly = false, rollbackFor = Throwable.class, propagation = Propagation.REQUIRED)
	@Override
	public UUID createSessionForApiKey(UUID apiKey) {

		var team = teamService.getTeamByApiKey(apiKey).orElseThrow(
				() -> new BusinessException(HttpStatus.FORBIDDEN, "SESS-001", "No team found for API key"));
		logger.atInfo().log("Trying to create a new session for team: {}", team.getName());
		if (sessionRepository.existsByTeamIdAndEndTimeIsNull(team.getId())) {
			logger.atWarn().log("Session creation failed: active session already exists for team: {}", team.getName());
			throw new SessionAlreadyExistsException();
		}

		EvaluationSession session = new EvaluationSession();
		session.setTeam(team);
		session.setCost(0);
		session.setCurrentHour(0);
		session.setCurrentDay(0);
		session.setStartTime(LocalDateTime.now());
		session.setLastUpdated(LocalDateTime.now());

		var updatedSession = sessionRepository.save(session);
		evaluationTrackRepository.markAsNotLatest();

		// Initialize inventories
		kitInventoryService.initKitInventories(session, airportService.getAll());

		// initialize kit movements with 0 loads for all flights
		var allScheduledFlights = flightService.getAllScheduledFlights();
		var scheduledKitFlightMovements = allScheduledFlights.stream().map(flight -> {
			KitMovement km = new KitMovement();
			km.setAirportId(flight.getOriginAirport().getId());
			km.setDay(flight.getScheduledDepartDay());
			km.setHour(flight.getScheduledDepartHour());
			km.setFlightId(flight.getId());
			km.setEvaluationSessionId(updatedSession.getId());
			km.setFirstKits(0);
			km.setBusinessKits(0);
			km.setPremiumEconomyKits(0);
			km.setEconomyKits(0);
			km.setCost(0d);
			return km;
		}).toList();

		kitMovementService.saveAll(scheduledKitFlightMovements);

		logger.atInfo().log("New session created with ID: {} for team: {}", session.getId(), team.getName());

		return session.getId();
	}

	@Override
	@Transactional(readOnly = false, rollbackFor = Throwable.class, propagation = Propagation.REQUIRED)
	public HourResponseDto playRound(UUID sessionId, int day, int hour, List<FlightLoadDto> flightLoads,
			PerClassAmount kitPurchasingOrders) {
		logger.atInfo().log("Playing round for session ID: {} at time {}:{}", sessionId, day, hour);
		var evaluationSession = sessionRepository.findById(sessionId).orElseThrow(SessionNotFoundException::new);
		logger.atInfo().log("Found session for ID: {}, with current time {}:{}", sessionId,
				evaluationSession.getCurrentDay(), evaluationSession.getCurrentHour());
		if (evaluationSession.getEndTime() != null) {
			throw new BadRequestException("SESS-003", "Session already ended");
		}
		if (evaluationSession.getCurrentDay() != day || evaluationSession.getCurrentHour() != hour) {
			throw new BadRequestException("SESS-004",
					MessageFormat.format("Provided time ({0}:{1}) is not the expected time ({2}:{3})", day, hour,
							evaluationSession.getCurrentDay(), evaluationSession.getCurrentHour()));
		}

		double currentCost = evaluationSession.getCost();

		double operationalCost = 0d;
		double penaltyCost = 0d;

		var evaluationTrack = new EvaluationTrack();
		evaluationTrack.setTeam(evaluationSession.getTeam());
		evaluationTrack.setSessionId(evaluationSession.getId());
		evaluationTrack.setLatest(true);
		evaluationTrack.setTimeReceived(LocalDateTime.now());
		evaluationTrack.setProdDay(day);
		evaluationTrack.setProdHour(hour);

		List<Penalty> penalties = new LinkedList<>();
		List<KitMovement> kitMovements = new LinkedList<>();

		logger.atInfo().log("Processing flight loads for session ID: {} at time {}:{}", sessionId, day, hour);
		processFlightLoads(day, hour, flightLoads, evaluationSession, kitMovements::add, penalties::add);

		logger.atInfo().log("Processing kit purchasing orders for session ID: {} at time {}:{}", sessionId, day, hour);

		operationalCost += processPurchasingOrders(day, hour, kitPurchasingOrders, evaluationSession,
				kitMovements::add);

		kitMovements.addAll(kitProcessingService.generateKitMovements(day, hour, evaluationSession.getId()));

		kitMovementService.saveAll(kitMovements);

		logger.atInfo().log("Evaluating kit movements for session ID: {} at time {}:{}", sessionId, day, hour);

		List<KitMovement> currentKitMovements = kitMovementService.getByDayHourAndSession(day, hour,
				evaluationSession.getId());

		logger.atInfo().log("Found {} kit movements for session ID: {} at time {}:{}", currentKitMovements.size(),
				sessionId, day, hour);

		operationalCost += currentKitMovements.stream().mapToDouble(KitMovement::getCost).sum();

		currentKitMovements.stream().filter(km -> km.getFlightId() != null).flatMap(km -> {
			List<Penalty> flightLoadPenalties = new LinkedList<>();
			flightLoadPenalties.addAll(addOverloadPenalties(day, hour, evaluationSession, km));

			flightLoadPenalties.addAll(addPassengerWithoutKitsPenalties(day, hour, evaluationSession, km));

			return flightLoadPenalties.stream();
		}).forEach(penalties::add);

		var kitProcessingsAtDestination = currentKitMovements
				.stream().filter(km -> km.getFlightId() != null).filter(km -> km.getFirstKits() <= 0
						&& km.getBusinessKits() <= 0 && km.getPremiumEconomyKits() <= 0 && km.getEconomyKits() <= 0)
				.flatMap(km -> {
					List<KitProcessing> kitProcessings = new LinkedList<>();

					toKitProcessing(km, KitType.A_FIRST_CLASS, km::getFirstKits).ifPresent(kitProcessings::add);
					toKitProcessing(km, KitType.B_BUSINESS, km::getBusinessKits).ifPresent(kitProcessings::add);
					toKitProcessing(km, KitType.C_PREMIUM_ECONOMY, km::getPremiumEconomyKits)
							.ifPresent(kitProcessings::add);
					toKitProcessing(km, KitType.D_ECONOMY, km::getEconomyKits).ifPresent(kitProcessings::add);

					return kitProcessings.stream();
				}).toList();

		if (!kitProcessingsAtDestination.isEmpty()) {
			logger.atInfo().log("Adding {} kit processings at destination airports for session ID: {} at time {}:{}",
					kitProcessingsAtDestination.size(), sessionId, day, hour);
			kitProcessingService.addAllToQueue(kitProcessingsAtDestination);
		} else {
			logger.atInfo().log("No kit processings at destination for session ID: {} at time {}:{}", sessionId, day,
					hour);
		}

		logger.atInfo().log("Updating stocks for session ID: {} at time {}:{}", sessionId, day, hour);
		updateStocks(day, hour, evaluationSession, penalties::add, currentKitMovements);

		var nextHour = ReferenceHour.addHours(day, hour, 1);
		evaluationSession.setCurrentDay(nextHour.day());
		evaluationSession.setCurrentHour(nextHour.hour());
		logger.atInfo().log("Session ID: {} advanced to time {}:{} from {}:{}", sessionId, nextHour.day(),
				nextHour.hour(), day, hour);

		if (evaluationSession.getCurrentDay() * 24 + evaluationSession.getCurrentHour() >= numberOfHours) {
			logger.atInfo().log("Session ID: {} has reached the end of the game at time {}:{}", sessionId, day, hour);
			penalties.addAll(endOfGame(evaluationSession, 1));
			evaluationSession.setEndTime(evaluationTrack.getTimeReceived());

		}

		penaltyCost = penalties.stream().mapToDouble(Penalty::getCost).sum();
		evaluationTrack.setProductionCost(operationalCost);
		evaluationTrack.setPenaltyCost(penaltyCost);

		evaluationSession.setCost(currentCost + operationalCost + penaltyCost);
		evaluationTrack.setTotalCost(evaluationSession.getCost());

		penaltiesRepository.saveAll(penalties);
		evaluationSession.setLastUpdated(LocalDateTime.now());
		sessionRepository.save(evaluationSession);
		evaluationTrackRepository.save(evaluationTrack);

		List<Flight> scheduledFlights = flightService.getScheduledFlights(nextHour.day(), nextHour.hour());

		logger.atInfo().log("Found {} scheduled flights for next hour {}:{}", scheduledFlights.size(), nextHour.day(),
				nextHour.hour());

		List<Flight> checkedInFlights = flightService.getCheckedInFlights(nextHour.day(), nextHour.hour());

		logger.atInfo().log("Found {} checked-in flights for next hour {}:{}", checkedInFlights.size(), nextHour.day(),
				nextHour.hour());

		List<Flight> landedFlight = flightService.getLandedFlights(nextHour.day(), nextHour.hour());
		logger.atInfo().log("Found {} landed flights for next hour {}:{}", landedFlight.size(), nextHour.day(),
				nextHour.hour());

		List<FlightEvent> flightUpdates = new LinkedList<>();

		flightUpdates.addAll(scheduledFlights.stream().map(this::toScheduledFlight).toList());
		flightUpdates.addAll(checkedInFlights.stream().map(this::toCheckedInFlight).toList());
		flightUpdates.addAll(landedFlight.stream().map(this::toLandedFlight).toList());

		return new HourResponseDto(day, hour, flightUpdates, penalties.stream().map(this::convertToPenaltyDto).toList(),
				evaluationSession.getCost());

	}

	private void updateStocks(int day, int hour, EvaluationSession evaluationSession, Consumer<Penalty> penalties,
			List<KitMovement> currentKitMovements) {

		logger.atInfo().log("Updating kit inventories for session ID: {} at time {}:{} with {} kit movements",
				evaluationSession.getId(), day, hour, currentKitMovements.size());
		if (currentKitMovements.isEmpty()) {
			logger.atInfo().log("No kit movements to process for session ID: {} at time {}:{}",
					evaluationSession.getId(), day, hour);
			return;
		}
		Map<UUID, List<KitMovement>> kitMovementsByAirport = currentKitMovements.stream()
				.collect(Collectors.groupingBy(km -> km.getAirportId()));

		Map<UUID, PerClassAmount> netKitMovementsByAirport = kitMovementsByAirport.entrySet().stream()
				.collect(Collectors.toMap(Map.Entry::getKey, e -> {
					int firstClassKits = e.getValue().stream().mapToInt(KitMovement::getFirstKits).sum();
					int businessKits = e.getValue().stream().mapToInt(KitMovement::getBusinessKits).sum();
					int premiumEconomyKits = e.getValue().stream().mapToInt(KitMovement::getPremiumEconomyKits).sum();
					int economyKits = e.getValue().stream().mapToInt(KitMovement::getEconomyKits).sum();
					return new PerClassAmount(firstClassKits, businessKits, premiumEconomyKits, economyKits);
				}));

		var kitInventories = kitInventoryService.getCurrentInventories(evaluationSession.getId());

		kitInventories.forEach(inventory -> {
			var airport = airportService.getById(inventory.getAirportId());
			var netMovement = netKitMovementsByAirport.get(inventory.getAirportId());
			if (netMovement != null) {
				double kitcost = inventory.getKitType().cost();
				int deltaKits = switch (inventory.getKitType()) {
				case A_FIRST_CLASS -> netMovement.first();
				case B_BUSINESS -> netMovement.business();
				case C_PREMIUM_ECONOMY -> netMovement.premiumEconomy();
				case D_ECONOMY -> netMovement.economy();
				default -> 0;
				};
				inventory.setAvailableKits(inventory.getAvailableKits() + deltaKits);
				if (inventory.getAvailableKits() < 0) {
					penalties.accept(createPenalty(evaluationSession, "NEGATIVE_INVENTORY", null, day, hour,
							"Negative inventory for airport " + airport.getCode() + " kit type "
									+ inventory.getKitType() + " of " + inventory.getAvailableKits() + " kits",
							PenaltyFactors.NEGATIVE_INVENTORY * Math.abs(inventory.getAvailableKits())));
				}

				if (inventory.getAvailableKits() > inventory.getCapacity()) {
					penalties.accept(createPenalty(evaluationSession, "INVENTORY_EXCEEDS_CAPACITY", null, day, hour,
							"Inventory exceeds capacity for airport " + airport.getCode() + " kit type "
									+ inventory.getKitType() + " of "
									+ (inventory.getAvailableKits() - inventory.getCapacity()) + " kits",
							PenaltyFactors.OVER_CAPACITY_STOCK
									* (inventory.getAvailableKits() - inventory.getCapacity())));
				}

			}
		});
		kitInventoryService.saveAll(kitInventories);
	}

	private Optional<KitProcessing> toKitProcessing(KitMovement km, KitType kitType, IntSupplier getter) {
		var value = getter.getAsInt();
		if (value < 0) {
			KitProcessing kp = new KitProcessing();
			var flight = flightService.getFlightById(km.getFlightId());
			kp.setEvaluationSessionId(km.getEvaluationSessionId());
			kp.setAvailableDay(flight.getActualArrivalDay());
			kp.setAvailableHour(flight.getActualArrivalHour());
			kp.setAirportId(flight.getDestinationAirport().getId());
			kp.setKitType(kitType);
			kp.setQuantity(0 - value);
			kp.setRemainingQuantity(kp.getQuantity());
			return Optional.of(kp);

		}
		return Optional.empty();
	}

	private double processPurchasingOrders(int day, int hour, PerClassAmount kitPurchasingOrders,
			EvaluationSession evaluationSession, Consumer<KitMovement> kitMovements) {
		var operationalCost = 0d;
		// add kit orders
		if (kitPurchasingOrders != null) {
			if (kitPurchasingOrders.first() > 0) {
				var fulfilmentTime = ReferenceHour.addHours(day, hour,
						KitType.A_FIRST_CLASS.replacementLeadTimeHours());
				operationalCost += KitType.A_FIRST_CLASS.cost() * kitPurchasingOrders.first();
				kitMovements.accept(buildKitMovementForOrder(fulfilmentTime.day(), fulfilmentTime.hour(),
						kitPurchasingOrders.first(), KitMovement::setFirstKits, evaluationSession));
			}
			if (kitPurchasingOrders.business() > 0) {
				var fulfilmentTime = ReferenceHour.addHours(day, hour, KitType.B_BUSINESS.replacementLeadTimeHours());
				kitMovements.accept(buildKitMovementForOrder(fulfilmentTime.day(), fulfilmentTime.hour(),
						kitPurchasingOrders.business(), KitMovement::setBusinessKits, evaluationSession));
				operationalCost += KitType.B_BUSINESS.cost() * kitPurchasingOrders.first();
			}
			if (kitPurchasingOrders.premiumEconomy() > 0) {
				var fulfilmentTime = ReferenceHour.addHours(day, hour,
						KitType.C_PREMIUM_ECONOMY.replacementLeadTimeHours());
				kitMovements.accept(buildKitMovementForOrder(fulfilmentTime.day(), fulfilmentTime.hour(),
						kitPurchasingOrders.premiumEconomy(), KitMovement::setPremiumEconomyKits, evaluationSession));
				operationalCost += KitType.C_PREMIUM_ECONOMY.cost() * kitPurchasingOrders.first();
			}
			if (kitPurchasingOrders.economy() > 0) {
				var fulfilmentTime = ReferenceHour.addHours(day, hour, KitType.D_ECONOMY.replacementLeadTimeHours());
				kitMovements.accept(buildKitMovementForOrder(fulfilmentTime.day(), fulfilmentTime.hour(),
						kitPurchasingOrders.economy(), KitMovement::setEconomyKits, evaluationSession));
				operationalCost += KitType.D_ECONOMY.cost() * kitPurchasingOrders.first();
			}
		}
		return operationalCost;
	}

	private boolean checkFlightValidity(EvaluationSession session, FlightLoadDto flightLoad, Flight flight,
			int currentDay, int currentHour, Consumer<Penalty> penaltyCollector) {
		if (flight == null) {
			penaltyCollector.accept(createPenalty(session, "FLIGHT_NOT_FOUND", flight, currentDay, currentHour,
					"Flight " + flightLoad.flightId() + " not found", PenaltyFactors.INCORRECT_FLIGHT_LOAD));
			return false;
		}

		if (flightLoad.loadedKits().first() < 0 || flightLoad.loadedKits().business() < 0
				|| flightLoad.loadedKits().premiumEconomy() < 0 || flightLoad.loadedKits().economy() < 0) {
			penaltyCollector.accept(createPenalty(session, "FLIGHT_INCORRECT_LOAD", flight, currentDay, currentHour,
					"Flight " + flightLoad.flightId() + " has incorrect load values",
					PenaltyFactors.INCORRECT_FLIGHT_LOAD * (negativeToZero(
							flightLoad.loadedKits().first() + negativeToZero(flightLoad.loadedKits().business())
									+ negativeToZero(flightLoad.loadedKits().premiumEconomy())
									+ negativeToZero(flightLoad.loadedKits().economy())))));
			return false;
		}
		return true;
	}

	private void processFlightLoads(int day, int hour, List<FlightLoadDto> flightLoads,
			EvaluationSession evaluationSession, Consumer<KitMovement> kitMovementCollector,
			Consumer<Penalty> penaltyCollector) {
		if (flightLoads == null || flightLoads.isEmpty()) {
			logger.atInfo().log("No flight loads to process for session ID: {} at time {}:{}",
					evaluationSession.getId(), day, hour);
			return;
		}

		flightLoadService.saveAll(
				flightLoads.stream().map(fl -> SessionServiceImpl.toFlightLoad(fl, evaluationSession)).toList());

		for (FlightLoadDto update : flightLoads) {
			Flight flight = flightService.getFlightById(update.flightId());

			if (!checkFlightValidity(evaluationSession, update, flight, day, hour, penaltyCollector)) {
				continue;
			}

			// persist stock retrievals
			Optional<KitMovement> kitMovementOpt = kitMovementService.getByFlightAndSession(flight.getId(),
					evaluationSession.getId());
			KitMovement kitMovement;
			if (kitMovementOpt.isEmpty()) {
				kitMovement = new KitMovement();
				kitMovement.setFlightId(flight.getId());
				kitMovement.setEvaluationSessionId(evaluationSession.getId());
				kitMovement.setDay(flight.getScheduledDepartDay());
				kitMovement.setHour(flight.getScheduledDepartHour());
				kitMovement.setAirportId(flight.getOriginAirport().getId());

			} else {
				kitMovement = kitMovementOpt.get();
			}
			kitMovement.setFirstKits(-update.loadedKits().first());
			kitMovement.setBusinessKits(-update.loadedKits().business());
			kitMovement.setPremiumEconomyKits(-update.loadedKits().premiumEconomy());
			kitMovement.setEconomyKits(-update.loadedKits().economy());
			kitMovement.setCost(update.loadedKits().first()
					* (flight.getOriginAirport().getFirstLoadingCost() + flight.getActualDistance()
							* flight.getActualAircraftType().getCostPerKgPerKm() * KitType.A_FIRST_CLASS.weightKg())
					+ update.loadedKits().business() * (flight.getOriginAirport().getBusinessLoadingCost()
							+ flight.getActualDistance() * flight.getActualAircraftType().getCostPerKgPerKm()
									* KitType.B_BUSINESS.weightKg())
					+ update.loadedKits().premiumEconomy() * (flight.getOriginAirport().getPremiumEconomyLoadingCost()
							+ flight.getActualDistance() * flight.getActualAircraftType().getCostPerKgPerKm()
									* KitType.C_PREMIUM_ECONOMY.weightKg())
					+ update.loadedKits().economy() * (flight.getOriginAirport().getEconomyLoadingCost()
							+ flight.getActualDistance() * flight.getActualAircraftType().getCostPerKgPerKm()
									* KitType.D_ECONOMY.weightKg()));

			kitMovementCollector.accept(kitMovement);

		}

	}

	private List<Penalty> addPassengerWithoutKitsPenalties(int day, int hour, EvaluationSession evaluationSession,
			KitMovement km) {

		List<Penalty> flightLoadPenalties = new LinkedList<>();
		Flight flight = flightService.getFlightById(km.getFlightId());

		if (flight.getActualFirstPassengers() > -km.getFirstKits()) {
			flightLoadPenalties.add(createPenalty(evaluationSession, "FLIGHT_UNFULFILLED_FIRST_CLASS", flight, day,
					hour,
					"Flight " + flight.getFlightNumber() + " has unfulfilled First Class passengers of "
							+ (flight.getActualFirstPassengers() + km.getFirstKits()) + " kits",
					PenaltyFactors.UNFULFILLED_KIT_FACTOR_PER_DISTANCE * KitType.A_FIRST_CLASS.cost()
							* flight.getActualDistance() * (flight.getActualFirstPassengers() + km.getFirstKits())));
		}
		if (flight.getActualBusinessPassengers() > -km.getBusinessKits()) {
			flightLoadPenalties
					.add(createPenalty(evaluationSession, "FLIGHT_UNFULFILLED_BUSINESS_CLASS", flight, day, hour,
							"Flight " + flight.getFlightNumber() + " has unfulfilled Business Class passengers of "
									+ (flight.getActualBusinessPassengers() + km.getBusinessKits()) + " kits",
							PenaltyFactors.UNFULFILLED_KIT_FACTOR_PER_DISTANCE * KitType.B_BUSINESS.cost()
									* flight.getActualDistance()
									* (flight.getActualBusinessPassengers() + km.getBusinessKits())));
		}
		if (flight.getActualPremiumEconomyPassengers() > -km.getPremiumEconomyKits()) {
			flightLoadPenalties.add(createPenalty(evaluationSession, "FLIGHT_UNFULFILLED_PREMIUM_ECONOMY_CLASS", flight,
					day, hour,
					"Flight " + flight.getFlightNumber() + " has unfulfilled Premium Economy Class passengers of "
							+ (flight.getActualPremiumEconomyPassengers() + km.getPremiumEconomyKits()) + " kits",
					PenaltyFactors.UNFULFILLED_KIT_FACTOR_PER_DISTANCE * flight.getActualDistance()
							* KitType.C_PREMIUM_ECONOMY.cost()
							* (flight.getActualPremiumEconomyPassengers() + km.getPremiumEconomyKits())));
		}

		if (flight.getActualEconomyPassengers() > -km.getEconomyKits()) {
			flightLoadPenalties
					.add(createPenalty(evaluationSession, "FLIGHT_UNFULFILLED_ECONOMY_CLASS", flight, day, hour,
							"Flight " + flight.getFlightNumber() + " has unfulfilled Economy Class passengers of "
									+ (flight.getActualEconomyPassengers() + km.getEconomyKits()) + " kits",
							PenaltyFactors.UNFULFILLED_KIT_FACTOR_PER_DISTANCE * KitType.D_ECONOMY.cost()
									* flight.getActualDistance()
									* (flight.getActualEconomyPassengers() + km.getEconomyKits())));
		}
		return flightLoadPenalties;
	}

	private List<Penalty> addOverloadPenalties(int day, int hour, EvaluationSession evaluationSession, KitMovement km) {

		var penaltyPattern = "Flight {0} is overloaded in {1} Class by {2} kits. Capacity of aircraft of type {3} is {4}, loaded kits are {5}.";

		List<Penalty> flightLoadPenalties = new LinkedList<>();
		Flight flight = flightService.getFlightById(km.getFlightId());
		AircraftType type = flight.getActualAircraftType();

		if (type.getFirstClassKitsCapacity() < -km.getFirstKits()) {
			flightLoadPenalties.add(createPenalty(evaluationSession, "FLIGHT_OVERLOADED_FIRST_CLASS", flight, day, hour,
					MessageFormat.format(penaltyPattern, flight.getFlightNumber(), "First",
							(-km.getFirstKits() - type.getFirstClassKitsCapacity()), type.getTypeName(),
							type.getFirstClassKitsCapacity(), -km.getFirstKits()),
					PenaltyFactors.FLIGHT_OVERLOAD_FACTOR_PER_DISTANCE * KitType.A_FIRST_CLASS.cost()
							* flight.getActualDistance() * (-km.getFirstKits() - type.getFirstClassKitsCapacity())));
		}
		if (type.getBusinessKitsCapacity() < -km.getBusinessKits()) {
			flightLoadPenalties
					.add(createPenalty(evaluationSession, "FLIGHT_OVERLOADED_BUSINESS_CLASS", flight, day, hour,
							MessageFormat.format(penaltyPattern, flight.getFlightNumber(), "Business",
									(-km.getBusinessKits() - type.getBusinessKitsCapacity()), type.getTypeName(),
									type.getBusinessKitsCapacity(), -km.getBusinessKits()),
							PenaltyFactors.FLIGHT_OVERLOAD_FACTOR_PER_DISTANCE * KitType.B_BUSINESS.cost()
									* flight.getActualDistance()
									* (-km.getBusinessKits() - type.getBusinessKitsCapacity())));
		}
		if (type.getPremiumEconomyKitsCapacity() < -km.getPremiumEconomyKits()) {
			flightLoadPenalties.add(createPenalty(evaluationSession, "FLIGHT_OVERLOADED_PREMIUM_ECONOMY_CLASS", flight,
					day, hour,
					MessageFormat.format(penaltyPattern, flight.getFlightNumber(), "Premium Economy",
							(-km.getPremiumEconomyKits() - type.getPremiumEconomyKitsCapacity()), type.getTypeName(),
							type.getPremiumEconomyKitsCapacity(), -km.getPremiumEconomyKits()),
					PenaltyFactors.FLIGHT_OVERLOAD_FACTOR_PER_DISTANCE * KitType.C_PREMIUM_ECONOMY.cost()
							* flight.getActualDistance()
							* (-km.getPremiumEconomyKits() - type.getPremiumEconomyKitsCapacity())));
		}
		if (type.getEconomyKitsCapacity() < -km.getEconomyKits()) {
			flightLoadPenalties
					.add(createPenalty(evaluationSession, "FLIGHT_OVERLOADED_ECONOMY_CLASS", flight, day, hour,
							MessageFormat.format(penaltyPattern, flight.getFlightNumber(), "Economy",
									(-km.getEconomyKits() - type.getEconomyKitsCapacity()), type.getTypeName(),
									type.getEconomyKitsCapacity(), -km.getEconomyKits()),
							PenaltyFactors.FLIGHT_OVERLOAD_FACTOR_PER_DISTANCE * KitType.D_ECONOMY.cost()
									* flight.getActualDistance()
									* (-km.getEconomyKits() - type.getEconomyKitsCapacity())));
		}
		return flightLoadPenalties;
	}

	private KitMovement buildKitMovementForOrder(int day, int hour, int amount, ObjIntConsumer<KitMovement> setter,
			EvaluationSession evaluationSession) {
		KitMovement kitMovement = new KitMovement();
		kitMovement.setEvaluationSessionId(evaluationSession.getId());
		kitMovement.setDay(day);
		kitMovement.setHour(hour);
		kitMovement.setFirstKits(0);
		kitMovement.setBusinessKits(0);
		kitMovement.setPremiumEconomyKits(0);
		kitMovement.setEconomyKits(0);
		kitMovement.setAirportId(airportService.getByCode(HUB_AIRPORT_CODE).getId());
		setter.accept(kitMovement, amount);

		return kitMovement;
	}

	private static int negativeToZero(int value) {
		return Math.abs(Math.min(value, 0));
	}

	@Override
	@Transactional
	public HourResponseDto stopSession(UUID apiKey) {
		var evaluationSession = sessionRepository.findByApiKey(apiKey).orElseThrow(SessionNotFoundException::new);
		if (evaluationSession.getEndTime() != null) {
			throw new BadRequestException("SESS-002", "Session already ended");
		}

		var missingHours = Math
				.abs(numberOfHours - (evaluationSession.getCurrentDay() * 24 + evaluationSession.getCurrentHour()));
		if (missingHours < 24) {
			missingHours = missingHours * 10;
		}

		evaluationSession.setEndTime(LocalDateTime.now());

		var penalties = endOfGame(evaluationSession, PenaltyFactors.EARLY_END_OF_GAME * missingHours);

		var endHour = ReferenceHour.addHours(0, 0, numberOfHours);

		var evaluationTrack = new EvaluationTrack();
		evaluationTrack.setTeam(evaluationSession.getTeam());
		evaluationTrack.setSessionId(evaluationSession.getId());
		evaluationTrack.setLatest(true);
		evaluationTrack.setTimeReceived(LocalDateTime.now());
		evaluationTrack.setProdDay(endHour.day());
		evaluationTrack.setProdHour(endHour.hour());
		evaluationTrack.setProductionCost(0);
		evaluationTrack.setPenaltyCost(penalties.stream().mapToDouble(Penalty::getCost).sum());

		evaluationSession.setCost(evaluationSession.getCost() + evaluationTrack.getPenaltyCost());

		evaluationTrack.setTotalCost(evaluationSession.getCost());

		penaltiesRepository.saveAll(penalties);
		sessionRepository.save(evaluationSession);
		evaluationTrackRepository.save(evaluationTrack);

		return new HourResponseDto(endHour.day(), endHour.hour(), null,
				penalties.stream().map(this::convertToPenaltyDto).toList(), evaluationSession.getCost());
	}

	private PenaltyDto convertToPenaltyDto(Penalty penalty) {
		return new PenaltyDto(penalty.getType(), penalty.getFlight() != null ? penalty.getFlight().getId() : null,
				penalty.getFlight() != null ? penalty.getFlight().getFlightNumber() : null, penalty.getDay(),
				penalty.getHour(), penalty.getCost(), penalty.getMessage());
	}

	public Collection<Penalty> endOfGame(EvaluationSession evaluationSession, double factor) {
		List<Penalty> penalties = new LinkedList<>();

		// get all kit inventories and apply end of game penalties for remaining stock
		var kitInventories = kitInventoryService.getCurrentInventories(evaluationSession.getId());
		kitInventories.forEach(inventory -> {
			var airport = airportService.getById(inventory.getAirportId());
			double normalizedStock = inventory.getAvailableKits();
			if (inventory.getAvailableKits() < 0) {
				normalizedStock = Math.abs(inventory.getAvailableKits()) * PenaltyFactors.NEGATIVE_INVENTORY;
			}
			if (inventory.getAvailableKits() > inventory.getCapacity()) {
				normalizedStock = (inventory.getAvailableKits() - inventory.getCapacity())
						* PenaltyFactors.OVER_CAPACITY_STOCK;
			}
			penalties.add(createPenalty(evaluationSession, "END_OF_GAME_REMAINING_STOCK", null,
					evaluationSession.getCurrentDay(), evaluationSession.getCurrentHour(),
					"End of game penalty for remaining stock at airport " + airport.getCode() + " kit type "
							+ inventory.getKitType(),
					PenaltyFactors.END_OF_GAME_REMAINING_STOCK * normalizedStock * inventory.getKitType().cost()
							* factor));

		});

		// get all pending kit processings and apply end of game penalties
		var pendingKitProcessings = kitProcessingService.getPendingKitProcessings(evaluationSession.getId());
		pendingKitProcessings.forEach(kp -> {
			var airport = airportService.getById(kp.getAirportId());
			penalties.add(createPenalty(evaluationSession, "END_OF_GAME_PENDING_KIT_PROCESSING", null,
					evaluationSession.getCurrentDay(), evaluationSession.getCurrentHour(),
					"End of game penalty for pending kit processing at airport " + airport.getCode() + " kit type "
							+ kp.getKitType(),
					PenaltyFactors.END_OF_GAME_PENDING_KIT_PROCESSING * kp.getRemainingQuantity()
							* kp.getKitType().cost() * factor));
		});

		// get all flights that still have unfulfilled kits and apply end of game
		// penalties

		var upcomingFlights = flightService.getUpcomingFlights(evaluationSession.getCurrentDay(),
				evaluationSession.getCurrentHour());
		upcomingFlights.forEach(flight -> {
			penalties.add(createPenalty(evaluationSession, "END_OF_GAME_UNFULFILLED_FLIGHT_KITS", flight,
					evaluationSession.getCurrentDay(), evaluationSession.getCurrentHour(),
					"End of game penalty for unfulfilled kits on flight " + flight.getFlightNumber(),
					(PenaltyFactors.END_OF_GAME_UNFULFILLED_FLIGHT_KITS * flight.getDistance()
							* (flight.getPlannedFirstPassengers() * KitType.A_FIRST_CLASS.cost()
									* KitType.A_FIRST_CLASS.weightKg()
									+ flight.getPlannedBusinessPassengers() * KitType.B_BUSINESS.cost()
											* KitType.B_BUSINESS.weightKg()
									+ flight.getPlannedPremiumEconomyPassengers() * KitType.C_PREMIUM_ECONOMY.cost()
											* KitType.C_PREMIUM_ECONOMY.weightKg()
									+ flight.getPlannedEconomyPassengers() * KitType.D_ECONOMY.cost()
											* KitType.D_ECONOMY.weightKg())
							* factor)));
		});

		return penalties;

	}

	private Penalty createPenalty(EvaluationSession session, String type, Flight flight, int day, int hour,
			String message, double cost) {
		Penalty penalty = new Penalty();
		penalty.setType(type);
		penalty.setMessage(message);
		penalty.setCost(cost);
		penalty.setDay(day);
		penalty.setHour(hour);
		penalty.setFlight(flight);
		penalty.setSession(session);
		return penalty;
	}

	private FlightEvent toScheduledFlight(Flight flight) {
		return new FlightEvent(FlightEventType.SCHEDULED, flight.getFlightNumber(), flight.getId(),
				flight.getOriginAirport().getCode(), flight.getDestinationAirport().getCode(),
				new ReferenceHour(flight.getScheduledDepartDay(), flight.getScheduledDepartHour()),
				new ReferenceHour(flight.getScheduledArrivalDay(), flight.getScheduledArrivalHour()),
				new PerClassAmount(flight.getPlannedFirstPassengers(), flight.getPlannedBusinessPassengers(),
						flight.getPlannedPremiumEconomyPassengers(), flight.getPlannedEconomyPassengers()),
				flight.getScheduledAircraftType().getTypeName());
	}

	private FlightEvent toCheckedInFlight(Flight flight) {
		return new FlightEvent(FlightEventType.CHECKED_IN, flight.getFlightNumber(), flight.getId(),
				flight.getOriginAirport().getCode(), flight.getDestinationAirport().getCode(),
				new ReferenceHour(flight.getScheduledDepartDay(), flight.getScheduledDepartHour()),
				new ReferenceHour(flight.getScheduledArrivalDay(), flight.getScheduledArrivalHour()),
				new PerClassAmount(flight.getActualFirstPassengers(), flight.getActualBusinessPassengers(),
						flight.getActualPremiumEconomyPassengers(), flight.getActualEconomyPassengers()),
				flight.getActualAircraftType().getTypeName());
	}

	private FlightEvent toLandedFlight(Flight flight) {
		return new FlightEvent(FlightEventType.LANDED, flight.getFlightNumber(), flight.getId(),
				flight.getOriginAirport().getCode(), flight.getDestinationAirport().getCode(),
				new ReferenceHour(flight.getScheduledDepartDay(), flight.getScheduledDepartHour()),
				new ReferenceHour(flight.getActualArrivalDay(), flight.getActualArrivalHour()),
				new PerClassAmount(flight.getActualFirstPassengers(), flight.getActualBusinessPassengers(),
						flight.getActualPremiumEconomyPassengers(), flight.getActualEconomyPassengers()),
				flight.getActualAircraftType().getTypeName());
	}

	private static FlightLoad toFlightLoad(FlightLoadDto dto, EvaluationSession session) {
		var fl = new FlightLoad();
		fl.setFlightId(dto.flightId());
		fl.setEvaluationSession(session);
		fl.setFirstKits(dto.loadedKits().first());
		fl.setBusinessKits(dto.loadedKits().business());
		fl.setPremiumEconomyKits(dto.loadedKits().premiumEconomy());
		fl.setEconomyKits(dto.loadedKits().economy());
		return fl;
	}
}