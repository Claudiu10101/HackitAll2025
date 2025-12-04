package com.sap.hackaton2025.controller;

import org.springframework.context.annotation.Primary;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.http.HttpStatus;
import org.springframework.http.ProblemDetail;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.reactive.result.method.annotation.ResponseEntityExceptionHandler;

import com.sap.hackaton2025.exception.BadRequestException;
import com.sap.hackaton2025.exception.BusinessException;
import com.sap.hackaton2025.exception.SessionAlreadyExistsException;
import com.sap.hackaton2025.exception.SessionNotFoundException;

@RestControllerAdvice
@Primary
@Order(Ordered.HIGHEST_PRECEDENCE)
public class GlobalExceptionHandler extends ResponseEntityExceptionHandler {

	@ExceptionHandler(BadRequestException.class)
	@ResponseStatus(HttpStatus.BAD_REQUEST)
	protected ProblemDetail handleBadRequestException(BadRequestException ex) {
		return handleBusinessException(ex);
	}

	@ExceptionHandler(SessionAlreadyExistsException.class)
	@ResponseStatus(HttpStatus.CONFLICT)
	protected ProblemDetail handleSessionAlreadyExistsException(SessionAlreadyExistsException ex) {
		return handleBusinessException(ex);
	}

	@ExceptionHandler(SessionNotFoundException.class)
	@ResponseStatus(HttpStatus.NOT_FOUND)
	protected ProblemDetail handleSessionNotFoundException(SessionNotFoundException ex) {
		return handleBusinessException(ex);
	}

	@ExceptionHandler(BusinessException.class)
	@ResponseStatus
	protected ProblemDetail handleBusinessException(BusinessException ex) {
		var problemDetail = ProblemDetail.forStatusAndDetail(ex.getStatusCode(), ex.getMessage());
		problemDetail.setProperty("code", ex.getCode());
		return problemDetail;
	}

	@ExceptionHandler(MethodArgumentNotValidException.class)
	@ResponseStatus
	protected ProblemDetail handleValidationException(MethodArgumentNotValidException ex) {
		var problemDetail = ProblemDetail.forStatusAndDetail(HttpStatus.BAD_REQUEST, "Validation failed");
		problemDetail.setProperty("errors", ex.getBindingResult().getAllErrors());
		return problemDetail;
	}

	@ExceptionHandler(Exception.class)
	@ResponseStatus
	protected ProblemDetail handleGenericException(Exception ex) {

		if (ex instanceof MethodArgumentNotValidException) {
			return handleValidationException((MethodArgumentNotValidException) ex);
		}

		var problemDetail = ProblemDetail.forStatusAndDetail(HttpStatus.INTERNAL_SERVER_ERROR,
				"An unexpected error occurred");
		return problemDetail;
	}
}
