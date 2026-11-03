import { NextResponse } from "next/server";
import { hashToken, getMemoryDevices } from "@/lib/device/provisioning";
import { getAdminFirestore, hasServerAdminCredentials } from "@/lib/server-admin";

export async function POST(request: Request) {
  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "BAD_REQUEST", message: "Invalid JSON body." }, { status: 400 });
  }

  // Device-level authenticated heartbeat support
  const deviceIdHeader = request.headers.get("x-device-id");
  const deviceId =
    deviceIdHeader || (typeof body.deviceId === "string" ? body.deviceId.trim() : "");
  const authHeader = request.headers.get("authorization");
  let bearerToken: string | null = null;
  if (authHeader && authHeader.toLowerCase().startsWith("bearer ")) {
    bearerToken = authHeader.slice(7).trim();
  }

  let robotId = typeof body.robotId === "string" ? body.robotId.trim() : "";

  if (!hasServerAdminCredentials()) {
    // Memory fallback check for mock/testing
    if (deviceId && bearerToken) {
      const dev = getMemoryDevices().get(deviceId);
      if (dev) {
        if (dev.status !== "paired" || dev.deviceTokenHash !== hashToken(bearerToken)) {
          return NextResponse.json({ error: "UNAUTHORIZED", message: "Invalid device credentials." }, { status: 401 });
        }
        robotId = dev.robotId;
        dev.lastHeartbeat = new Date().toISOString();
      }
    }
    return NextResponse.json({ success: true, timestamp: new Date().toISOString(), robotId });
  }

  try {
    const adminDb = await getAdminFirestore();

    // If deviceId provided, verify device credentials and use authoritative robotId
    if (deviceId) {
      const deviceRef = adminDb.collection("devices").doc(deviceId);
      const devSnap = await deviceRef.get();
      if (!devSnap.exists) {
        return NextResponse.json(
          { error: "UNAUTHORIZED", message: `Device ${deviceId} not registered.` },
          { status: 401 }
        );
      }
      const devData = devSnap.data() || {};
      if (devData.status !== "paired") {
        return NextResponse.json(
          { error: "UNAUTHORIZED", message: `Device ${deviceId} is not in paired status.` },
          { status: 401 }
        );
      }
      if (bearerToken) {
        const tokenHash = hashToken(bearerToken);
        if (devData.deviceTokenHash && devData.deviceTokenHash !== tokenHash) {
          return NextResponse.json(
            { error: "UNAUTHORIZED", message: "Invalid device authorization token." },
            { status: 401 }
          );
        }
      }
      if (devData.robotId) {
        robotId = devData.robotId;
      }
      await deviceRef.set({ lastHeartbeat: new Date().toISOString() }, { merge: true });
    }

    if (!robotId) {
      return NextResponse.json({ error: "INVALID_ARGUMENT", message: "robotId is required." }, { status: 400 });
    }

    let robotRef = adminDb.collection("robots").doc(robotId);
    let docSnap = await robotRef.get();

    // Canonical resolution: match across case and hyphenation (KB001 <-> Kb-001 <-> KB-001)
    if (!docSnap.exists) {
      const cleanTarget = robotId.replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
      const allRobots = await adminDb.collection("robots").get();
      for (const d of allRobots.docs) {
        if (d.id.replace(/[^a-zA-Z0-9]/g, "").toLowerCase() === cleanTarget) {
          robotRef = d.ref;
          docSnap = d;
          break;
        }
      }
    }

    if (!docSnap.exists) {
      return NextResponse.json({ error: "NOT_FOUND", message: `Robot ${robotId} not found.` }, { status: 404 });
    }

    const currentData = docSnap.data() || {};
    const now = new Date().toISOString();

    const updates: Record<string, unknown> = {
      status: "online",
      lastSeen: now,
      updatedAt: now,
    };

    if (deviceId) {
      updates.deviceId = deviceId;
      updates.deviceStatus = "paired";
    }

    if (body.telemetry && typeof body.telemetry === "object") {
      updates.telemetry = body.telemetry;
    }
    if (typeof body.firmwareVersion === "string") {
      updates.firmwareVersion = body.firmwareVersion;
    }

    await robotRef.set(updates, { merge: true });

    // Return the latest robot configuration for edge state synchronization
    return NextResponse.json({
      success: true,
      timestamp: now,
      robotId,
      config: currentData.config || {},
      mode: currentData.mode || "education",
      language: currentData.language || "en",
      knowledgeBaseId: currentData.knowledgeBaseId || null,
    });
  } catch (err: unknown) {
    return NextResponse.json(
      { error: "INTERNAL_ERROR", message: err instanceof Error ? err.message : "Failed to process heartbeat." },
      { status: 500 }
    );
  }
}
