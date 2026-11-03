import crypto from "node:crypto";

const CODE_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ";

/**
 * Generate a cryptographically random, human-friendly 12-character pairing code
 * formatted as XXXX-XXXX-XXXX (excluding ambiguous characters: 0, 1, I, O).
 */
export function generatePairingCode(): string {
  const bytes = crypto.randomBytes(12);
  let raw = "";
  for (let i = 0; i < 12; i++) {
    raw += CODE_ALPHABET[bytes[i] % CODE_ALPHABET.length];
  }
  return `${raw.slice(0, 4)}-${raw.slice(4, 8)}-${raw.slice(8, 12)}`;
}

/**
 * Normalize user input pairing code by stripping hyphens and whitespace and converting to uppercase.
 */
export function normalizePairingCode(code: string): string {
  return (code || "").replace(/[^a-zA-Z0-9]/g, "").toUpperCase();
}

/**
 * Hash normalized pairing code with SHA-256 for secure storage and comparison.
 */
export function hashPairingCode(normalizedCode: string): string {
  return crypto.createHash("sha256").update(normalizedCode).digest("hex");
}

/**
 * Hash a high-entropy device token with SHA-256.
 */
export function hashToken(token: string): string {
  return crypto.createHash("sha256").update(token).digest("hex");
}

/**
 * Generate a 32-byte cryptographic challenge string.
 */
export function generateChallenge(): string {
  return crypto.randomBytes(32).toString("hex");
}

export interface PairingSession {
  sessionId: string;
  deviceId: string;
  publicKey: string;
  deviceInfo: Record<string, unknown>;
  codeHash: string;
  challenge: string;
  status: "pending" | "approved" | "consumed" | "expired" | "rejected";
  expiresAt: string;
  createdAt: string;
  robotId?: string | null;
  tenantId?: string | null;
  deviceToken?: string | null;
  robotConfig?: Record<string, unknown> | null;
}

export interface PairedDevice {
  deviceId: string;
  robotId: string;
  tenantId?: string | null;
  publicKey: string;
  deviceTokenHash: string;
  status: "paired" | "revoked";
  pairedAt: string;
  pairedBy: string;
  deviceInfo: Record<string, unknown>;
  lastHeartbeat?: string | null;
  updatedAt: string;
}

// In-memory fallback stores for environments without Firestore Admin credentials
const memoryPairingSessions = new Map<string, PairingSession>();
const memoryDevices = new Map<string, PairedDevice>();

export function getMemoryPairingSessions(): Map<string, PairingSession> {
  return memoryPairingSessions;
}

export function getMemoryDevices(): Map<string, PairedDevice> {
  return memoryDevices;
}
