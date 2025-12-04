package com.sap.hackaton2025.security;

public interface ApiKeyStore {
	ApiKeyToken getApiKeyToken(String apiKey);
}
