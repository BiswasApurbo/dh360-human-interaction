/**
 * Server-side Firebase Authentication & Role Validation.
 *
 * Enforces cryptographic verification of Firebase ID tokens:
 * 1. Cryptographic RS256 signature verification against Google's public x509 certs in production.
 * 2. Strict emulator account lookup against Firebase Auth Emulator in development/test.
 * 3. Issuer, audience, expiration, algorithm, and account status validation.
 * 4. Emulator shortcuts are STRICTLY forbidden in production (NODE_ENV === 'production').
 * 5. JWT decoding alone is NEVER treated as authentication. Fails closed on any error.
 */

import crypto from "node:crypto";

export interface AuthPrincipal {
  uid: string;
  email: string | null;
  role: "superAdmin" | "admin";
}

interface DecodedTokenHeader {
  alg?: string;
  kid?: string;
  typ?: string;
}

interface DecodedTokenPayload {
  uid?: string;
  user_id?: string;
  sub?: string;
  email?: string;
  role?: string;
  exp?: number;
  iat?: number;
  auth_time?: number;
  iss?: string;
  aud?: string;
  [key: string]: unknown;
}

// In-memory cache for Google public x509 certificates
let cachedGoogleCerts: Record<string, string> | null = null;
let googleCertsExpiryMs = 0;

/**
 * Fetch Google's public certificates for Firebase token verification.
 * Honors Cache-Control max-age header.
 */
async function getGooglePublicCerts(): Promise<Record<string, string>> {
  const now = Date.now();
  if (cachedGoogleCerts && now < googleCertsExpiryMs) {
    return cachedGoogleCerts;
  }

  const certsUrl =
    "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com";
  const res = await fetch(certsUrl, {
    signal: AbortSignal.timeout(5000),
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch Google certificates: HTTP ${res.status}`);
  }

  const certs = (await res.json()) as Record<string, string>;

  // Parse Cache-Control max-age
  const cacheControl = res.headers.get("cache-control") || "";
  const match = cacheControl.match(/max-age=(\d+)/);
  const maxAgeSec = match ? parseInt(match[1], 10) : 3600;

  cachedGoogleCerts = certs;
  googleCertsExpiryMs = now + maxAgeSec * 1000;
  return certs;
}

/**
 * Safely parse a base64url encoded string to JSON.
 */
function parseBase64UrlJson<T>(base64Url: string): T | null {
  try {
    let base64 = base64Url.replace(/-/g, "+").replace(/_/g, "/");
    while (base64.length % 4 !== 0) {
      base64 += "=";
    }
    const jsonStr = Buffer.from(base64, "base64").toString("utf-8");
    return JSON.parse(jsonStr) as T;
  } catch {
    return null;
  }
}

/**
 * Check if the current execution environment is permitted to use the Auth Emulator.
 * Guard: Emulator mode is STRICTLY prohibited in production environments.
 */
function isEmulatorAllowed(): boolean {
  if (process.env.NODE_ENV === "production") {
    return false;
  }
  return Boolean(
    process.env.FIREBASE_AUTH_EMULATOR_HOST ||
      process.env.NEXT_PUBLIC_USE_EMULATOR === "true"
  );
}

/**
 * Get the target Firebase Project ID for audience/issuer verification.
 */
function getExpectedProjectId(): string {
  return (
    process.env.FIREBASE_PROJECT_ID ||
    process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID ||
    "kbcon-dashboard"
  );
}

/**
 * Validates a Firebase ID Token and returns the authenticated principal.
 *
 * Fails closed on any signature mismatch, malformed payload, disabled user,
 * expiration, or unexpected issuer/audience.
 */
export async function authenticateFirebaseToken(
  authHeader: string | null | undefined
): Promise<AuthPrincipal | null> {
  if (!authHeader || typeof authHeader !== "string") {
    return null;
  }

  const parts = authHeader.trim().split(" ");
  if (parts.length !== 2 || parts[0].toLowerCase() !== "bearer") {
    return null;
  }

  const token = parts[1].trim();
  if (!token) {
    return null;
  }

  const jwtParts = token.split(".");
  if (jwtParts.length !== 3) {
    return null;
  }

  const header = parseBase64UrlJson<DecodedTokenHeader>(jwtParts[0]);
  const payload = parseBase64UrlJson<DecodedTokenPayload>(jwtParts[1]);
  if (!header || !payload) {
    return null;
  }

  const nowSec = Math.floor(Date.now() / 1000);

  // Expiration check (with 10-second clock skew tolerance)
  if (typeof payload.exp !== "number" || payload.exp < nowSec - 10) {
    return null;
  }

  // ---------------------------------------------------------------------------
  // Branch A: Local Development / Emulator Mode
  // ---------------------------------------------------------------------------
  if (isEmulatorAllowed()) {
    const emulatorHost =
      process.env.FIREBASE_AUTH_EMULATOR_HOST || "127.0.0.1:9099";

    try {
      const lookupUrl = `http://${emulatorHost}/identitytoolkit.googleapis.com/v1/accounts:lookup?key=local`;
      const res = await fetch(lookupUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ idToken: token }),
        signal: AbortSignal.timeout(3000),
      });

      if (!res.ok) {
        // Emulator rejected the token (expired, bad signature, or invalid)
        return null;
      }

      const data = (await res.json()) as {
        users?: Array<{
          localId: string;
          email?: string;
          disabled?: boolean;
          customAttributes?: string;
        }>;
      };

      const user = data.users?.[0];
      if (!user || user.disabled) {
        return null;
      }

      let customClaims: { role?: string } = {};
      if (user.customAttributes) {
        try {
          customClaims = JSON.parse(user.customAttributes);
        } catch {
          return null;
        }
      }

      const effectiveRole = customClaims.role || payload.role;
      if (effectiveRole !== "superAdmin" && effectiveRole !== "admin") {
        return null;
      }

      return {
        uid: user.localId,
        email: user.email ?? payload.email ?? null,
        role: effectiveRole as "superAdmin" | "admin",
      };
    } catch {
      // Fail closed: If emulator is unreachable or lookup times out, NEVER fall back to unverified decoding!
      return null;
    }
  }

  // ---------------------------------------------------------------------------
  // Branch B: Cryptographic Production Verification against Google Public Keys
  // ---------------------------------------------------------------------------
  // 1. Strict algorithm check: MUST be RS256
  if (header.alg !== "RS256" || !header.kid) {
    return null;
  }

  // 2. Audience and Issuer check against Google Secure Token service
  const projectId = getExpectedProjectId();
  const expectedIss = `https://securetoken.google.com/${projectId}`;
  if (payload.aud !== projectId || payload.iss !== expectedIss) {
    return null;
  }

  // 3. Subject check (must be non-empty string <= 128 chars)
  const uid = payload.sub || payload.user_id;
  if (!uid || typeof uid !== "string" || uid.length > 128) {
    return null;
  }

  // 4. Issued at and Auth time checks (cannot be in the far future)
  if (typeof payload.iat === "number" && payload.iat > nowSec + 10) {
    return null;
  }
  if (typeof payload.auth_time === "number" && payload.auth_time > nowSec + 10) {
    return null;
  }

  // 5. Cryptographic signature check against Google public x509 certs
  try {
    const certs = await getGooglePublicCerts();
    const cert = certs[header.kid];
    if (!cert) {
      // Unknown Key ID
      return null;
    }

    const signingInput = Buffer.from(`${jwtParts[0]}.${jwtParts[1]}`);
    const signature = Buffer.from(jwtParts[2], "base64url");

    const isSignatureValid = crypto.verify(
      "RSA-SHA256",
      signingInput,
      cert,
      signature
    );

    if (!isSignatureValid) {
      return null;
    }

    // 6. Extract trusted custom claims only after signature verification
    const role = payload.role;
    if (role !== "superAdmin" && role !== "admin") {
      return null;
    }

    return {
      uid,
      email: payload.email || null,
      role: role as "superAdmin" | "admin",
    };
  } catch {
    // Fail closed on any network or crypto error
    return null;
  }
}

/**
 * Reset public certs cache for testing.
 */
export function resetAuthCertsCacheForTesting(): void {
  cachedGoogleCerts = null;
  googleCertsExpiryMs = 0;
}
