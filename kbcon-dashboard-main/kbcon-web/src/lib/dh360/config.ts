/**
 * DH360 BFF Configuration Module.
 *
 * Reads server-only environment variables for connecting to the local
 * DH360 Human Interaction FastAPI daemon.
 *
 * Security:
 * - DH360_OPERATOR_TOKEN must NEVER be prefixed with NEXT_PUBLIC_ or leaked to the client.
 */

export interface Dh360Config {
  apiUrl: string;
  operatorToken: string;
}

export class ConfigurationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConfigurationError";
  }
}

let cachedConfig: Dh360Config | null = null;

export function getDh360Config(): Dh360Config {
  if (cachedConfig) {
    return cachedConfig;
  }

  const rawUrl = process.env.DH360_API_URL || "http://127.0.0.1:8001";
  let apiUrl: string;
  try {
    const parsed = new URL(rawUrl);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      throw new Error("Invalid protocol");
    }
    apiUrl = parsed.origin;
  } catch {
    throw new ConfigurationError(
      `DH360_API_URL must be a valid HTTP/HTTPS URL. Received: "${rawUrl}"`
    );
  }

  const operatorToken = (process.env.DH360_OPERATOR_TOKEN || "").trim();
  if (!operatorToken) {
    throw new ConfigurationError(
      "DH360_OPERATOR_TOKEN is not set or empty in server environment. " +
        "Please configure DH360_OPERATOR_TOKEN in .env.local."
    );
  }

  cachedConfig = { apiUrl, operatorToken };
  return cachedConfig;
}

export function resetConfigCacheForTesting(): void {
  cachedConfig = null;
}
