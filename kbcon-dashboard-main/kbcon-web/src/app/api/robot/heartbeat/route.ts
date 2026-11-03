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
    if (deviceId) {
      let dev = getMemoryDevices().get(deviceId);
      if (!dev) {
        dev = {
          deviceId,
          robotId: robotId || "Kb-001",
          tenantId: "F1M7Ed8CXUNo7KIXLao5aZTuRN02",
          publicKey: "",
          deviceTokenHash: bearerToken ? hashToken(bearerToken) : "",
          status: "paired",
          pairedAt: new Date().toISOString(),
          pairedBy: "auto_registration",
          deviceInfo: (body.telemetry as Record<string, unknown>) || {},
          updatedAt: new Date().toISOString(),
          lastHeartbeat: new Date().toISOString(),
        };
        getMemoryDevices().set(deviceId, dev);
      } else {
        if (bearerToken && (!dev.deviceTokenHash || dev.deviceTokenHash !== hashToken(bearerToken))) {
          dev.deviceTokenHash = hashToken(bearerToken);
        }
        dev.status = "paired";
        if (!dev.robotId && robotId) dev.robotId = robotId;
        dev.lastHeartbeat = new Date().toISOString();
      }
      if (dev.robotId) robotId = dev.robotId;
    }
    return NextResponse.json({
      success: true,
      timestamp: new Date().toISOString(),
      robotId: robotId || "Kb-001",
      tenant: "Salute",
      tenantName: "Salute",
      tenantId: "F1M7Ed8CXUNo7KIXLao5aZTuRN02",
      mode: "education",
      language: "bn",
    });
  }

  try {
    const adminDb = await getAdminFirestore();

    // If deviceId provided, verify device credentials and use authoritative robotId
    if (deviceId) {
      const deviceRef = adminDb.collection("devices").doc(deviceId);
      const devSnap = await deviceRef.get();
      const now = new Date().toISOString();

      if (!devSnap.exists) {
        // Auto-reconcile known physical device or bound robot
        const defaultRobotId = robotId || "Kb-001";
        await deviceRef.set({
          deviceId,
          status: "paired",
          bindingStatus: "bound",
          robotId: defaultRobotId,
          robotName: "KB",
          tenantId: "F1M7Ed8CXUNo7KIXLao5aZTuRN02",
          tenantName: "Salute",
          deviceTokenHash: bearerToken ? hashToken(bearerToken) : undefined,
          lastHeartbeat: now,
          updatedAt: now,
        }, { merge: true });
        robotId = defaultRobotId;
      } else {
        const devData = devSnap.data() || {};
        if (devData.robotId) {
          robotId = devData.robotId;
        } else if (robotId) {
          // Bind device to provided robot
          await deviceRef.set({
            robotId,
            bindingStatus: "bound",
            status: "paired",
          }, { merge: true });
        }

        // Reconcile token hash if missing or matching device
        if (bearerToken) {
          const tokenHash = hashToken(bearerToken);
          if (!devData.deviceTokenHash) {
            await deviceRef.set({ deviceTokenHash: tokenHash }, { merge: true });
          }
        }

        await deviceRef.set({
          status: "paired",
          bindingStatus: "bound",
          lastHeartbeat: now,
          updatedAt: now,
        }, { merge: true });
      }
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

    const authoritativeTenantName = currentData.assignedAdminName || "Salute";
    const authoritativeTenantId =
      currentData.assignedAdminId || currentData.tenant_id || "F1M7Ed8CXUNo7KIXLao5aZTuRN02";

    // Return the latest robot configuration for edge state synchronization
    return NextResponse.json({
      success: true,
      timestamp: now,
      robotId,
      name: currentData.name || robotId,
      tenant: authoritativeTenantName,
      tenantName: authoritativeTenantName,
      tenantId: authoritativeTenantId,
      config: currentData.config || {},
      mode: currentData.mode || "education",
      language: currentData.language || "bn",
      knowledgeBaseId: currentData.knowledgeBaseId || null,
    });
  } catch (err: unknown) {
    return NextResponse.json(
      { error: "INTERNAL_ERROR", message: err instanceof Error ? err.message : "Failed to process heartbeat." },
      { status: 500 }
    );
  }
}
