/**
 * DH360 BFF Configuration Module.
 *
 * Reads server-only environment variables for connecting to the
 * DH360 Human Interaction FastAPI daemon.
 *
 * In local development, defaults to http://127.0.0.1:8000.
 * In cloud production deployments (e.g. Vercel), DH360_API_URL must be
 * explicitly set to the public/tunneled HTTPS URL of the backend (e.g. via Cloudflare Tunnel or ngrok)
 * because localhost/127.0.0.1 resolves to the isolated cloud container.
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

  const isVercel = process.env.VERCEL === "1";
  const rawUrl = process.env.DH360_API_URL || (isVercel ? "" : "http://127.0.0.1:8000");

  if (!rawUrl) {
    throw new ConfigurationError(
      "DH360_API_URL is not configured in environment variables. " +
        (isVercel
          ? "In production (Vercel), DH360_API_URL must be set to the secure HTTPS tunnel or public domain of your DH360 backend (e.g. Cloudflare Tunnel or ngrok)."
          : "Please set DH360_API_URL in .env.local.")
    );
  }

  let apiUrl: string;
  try {
    const parsed = new URL(rawUrl);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      throw new Error("Invalid protocol");
    }

    const isLocalhost =
      parsed.hostname === "localhost" ||
      parsed.hostname === "127.0.0.1" ||
      parsed.hostname === "0.0.0.0";

    if (isVercel && isLocalhost) {
      throw new ConfigurationError(
        `DH360_API_URL cannot point to localhost/127.0.0.1 in a Vercel production deployment. ` +
          `The DH360 perception daemon runs locally with webcam/hardware access and must be reachable via a secure HTTPS tunnel ` +
          `(e.g., Cloudflare Tunnel: https://<tunnel>.trycloudflare.com or a custom domain). ` +
          `Please configure DH360_API_URL in Vercel Project Settings > Environment Variables.`
      );
    }

    apiUrl = parsed.origin;
  } catch (err: unknown) {
    if (err instanceof ConfigurationError) {
      throw err;
    }
    throw new ConfigurationError(
      `DH360_API_URL must be a valid HTTP/HTTPS URL. Received: "${rawUrl}"`
    );
  }

  const operatorToken = (process.env.DH360_OPERATOR_TOKEN || "").trim();
  if (!operatorToken) {
    throw new ConfigurationError(
      "DH360_OPERATOR_TOKEN is not set or empty in server environment. " +
        (isVercel
          ? "Please configure DH360_OPERATOR_TOKEN in Vercel Project Settings > Environment Variables."
          : "Please configure DH360_OPERATOR_TOKEN in .env.local.")
    );
  }

  cachedConfig = { apiUrl, operatorToken };
  return cachedConfig;
}

export function resetConfigCacheForTesting(): void {
  cachedConfig = null;
}
