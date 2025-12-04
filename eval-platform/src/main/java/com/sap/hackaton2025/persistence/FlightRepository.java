package com.sap.hackaton2025.persistence;

import java.util.List;
import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import com.sap.hackaton2025.model.Flight;

@Repository
public interface FlightRepository extends JpaRepository<Flight, UUID> {

	@Query("""
			SELECT f FROM Flight f
			WHERE (f.scheduledDepartDay > :scheduledDepartDayStart or ( f.scheduledDepartDay = :scheduledDepartDayStart AND f.scheduledDepartHour >= :scheduledDepartHourStart))
			  AND (f.scheduledDepartDay < :scheduledDepartDayEnd or ( f.scheduledDepartDay = :scheduledDepartDayEnd AND f.scheduledDepartHour <= :scheduledDepartHourEnd))
			""")
	List<Flight> findByScheduledDepartDayAndScheduledDepartHour(int scheduledDepartDayStart, int scheduledDepartHourStart, int scheduledDepartDayEnd, int scheduledDepartHourEnd);

	List<Flight> findByActualArrivalDayAndActualArrivalHour(int actualArrivalDay, int actualArrivalHour);

	@Query("""
			SELECT f FROM Flight f
			WHERE (f.actualArrivalDay > :day)
			   OR (f.actualArrivalDay = :day AND f.actualArrivalHour >= :hour)
			""")
	List<Flight> findAllUpcoming(int day, int hour);

}