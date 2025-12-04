package com.sap.hackaton2025.exception;

import org.springframework.http.HttpStatus;

public class BadRequestException extends BusinessException {

	private static final long serialVersionUID = 1L;

	public BadRequestException(String code, String message) {
		super(HttpStatus.BAD_REQUEST, code, message);
	}

}
