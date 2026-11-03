import crypto from "node:crypto";

/**
 * Hash a high-entropy device token with SHA-256 for secure storage and comparison.
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

export interface PairedDevice {
  deviceId: string;
  robotId: string;
  robotName?: string | null;
  tenantId?: string | null;
  tenantName?: string | null;
  publicKey: string;
  fingerprint?: string;
  deviceTokenHash: string;
  status: "registered" | "paired" | "revoked" | "unpaired";
  bindingStatus?: "bound" | "unbound";
  pairedAt?: string | null;
  pairedBy?: string | null;
  deviceInfo: Record<string, unknown>;
  lastHeartbeat?: string | null;
  updatedAt: string;
}

// In-memory fallback stores for environments without Firestore Admin credentials
const memoryDevices = new Map<string, PairedDevice>();

export function getMemoryDevices(): Map<string, PairedDevice> {
  return memoryDevices;
}
