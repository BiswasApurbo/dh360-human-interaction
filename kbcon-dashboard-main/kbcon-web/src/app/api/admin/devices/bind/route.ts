import { NextResponse } from "next/server";
import crypto from "node:crypto";
import { authenticateFirebaseToken } from "@/lib/dh360/auth";
import { hashToken, getMemoryDevices } from "@/lib/device/provisioning";
import { getAdminFirestore, hasServerAdminCredentials } from "@/lib/server-admin";

export async function POST(request: Request) {
  const authHeader = request.headers.get("authorization");
  const authResult = await authenticateFirebaseToken(authHeader, { checkRevoked: true });

  if (!authResult.success) {
    return NextResponse.json(
      { error: authResult.code, message: authResult.message },
      { status: authResult.status }
    );
  }

  if (authResult.principal.role !== "superAdmin") {
    return NextResponse.json(
      { error: "FORBIDDEN", message: "Only Super Admin can bind physical devices." },
      { status: 403 }
    );
  }

  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "BAD_REQUEST", message: "Invalid JSON body." }, { status: 400 });
  }

  const deviceId = typeof body.deviceId === "string" ? body.deviceId.trim() : "";
  const robotId = typeof body.robotId === "string" ? body.robotId.trim() : "";

  if (!deviceId) {
    return NextResponse.json({ error: "INVALID_ARGUMENT", message: "deviceId is required." }, { status: 400 });
  }

  if (!robotId) {
    return NextResponse.json({ error: "INVALID_ARGUMENT", message: "robotId is required." }, { status: 400 });
  }

  const now = new Date().toISOString();
  const rawToken = crypto.randomBytes(32).toString("hex");
  const tokenHash = hashToken(rawToken);

  if (!hasServerAdminCredentials()) {
    const memDevice = getMemoryDevices().get(deviceId) || {
      deviceId,
      robotId,
      tenantId: "F1M7Ed8CXUNo7KIXLao5aZTuRN02",
      publicKey: "mock_key",
      deviceTokenHash: tokenHash,
      status: "paired",
      pairedAt: now,
      pairedBy: authResult.principal.uid,
      deviceInfo: { arch: "arm64" },
      updatedAt: now,
    };

    memDevice.robotId = robotId;
    memDevice.status = "paired";
    memDevice.pairedAt = now;
    memDevice.deviceTokenHash = tokenHash;
    getMemoryDevices().set(deviceId, memDevice);

    return NextResponse.json({
      success: true,
      message: `Device ${deviceId} successfully bound to robot ${robotId}!`,
      deviceToken: rawToken,
      bindingStatus: "bound",
      robotId,
      tenantId: "F1M7Ed8CXUNo7KIXLao5aZTuRN02",
      tenantName: "Salute",
    });
  }

  try {
    const adminDb = await getAdminFirestore();

    // 1. Fetch target robot
    let robotRef = adminDb.collection("robots").doc(robotId);
    let robotSnap = await robotRef.get();

    if (!robotSnap.exists) {
      // Try canonical resolution
      const cleanTarget = robotId.replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
      const allRobots = await adminDb.collection("robots").get();
      for (const d of allRobots.docs) {
        if (d.id.replace(/[^a-zA-Z0-9]/g, "").toLowerCase() === cleanTarget) {
          robotRef = d.ref;
          robotSnap = d;
          break;
        }
      }
    }

    if (!robotSnap.exists) {
      return NextResponse.json({ error: "NOT_FOUND", message: `Target robot ${robotId} not found.` }, { status: 404 });
    }

    const robotData = robotSnap.data() || {};
    const authoritativeTenantId = robotData.assignedAdminId || robotData.tenant_id || "F1M7Ed8CXUNo7KIXLao5aZTuRN02";
    const authoritativeTenantName = robotData.assignedAdminName || "Salute";

    // 2. Clear any previous device bound to this robot
    if (robotData.deviceId && robotData.deviceId !== deviceId) {
      const prevDevRef = adminDb.collection("devices").doc(robotData.deviceId);
      await prevDevRef.set(
        {
          robotId: null,
          robotName: null,
          tenantId: null,
          tenantName: null,
          status: "unpaired",
          bindingStatus: "unbound",
          updatedAt: now,
        },
        { merge: true }
      );
    }

    // 3. Update device document
    const deviceRef = adminDb.collection("devices").doc(deviceId);
    await deviceRef.set(
      {
        deviceId,
        robotId: robotSnap.id,
        robotName: robotData.name || robotSnap.id,
        tenantId: authoritativeTenantId,
        tenantName: authoritativeTenantName,
        language: robotData.language || "bn",
        mode: robotData.mode || "education",
        status: "paired",
        bindingStatus: "bound",
        deviceTokenHash: tokenHash,
        pairedAt: now,
        pairedBy: authResult.principal.uid,
        updatedAt: now,
      },
      { merge: true }
    );

    // 4. Update robot document
    await robotRef.set(
      {
        deviceId,
        deviceStatus: "paired",
        status: "online",
        updatedAt: now,
      },
      { merge: true }
    );

    return NextResponse.json({
      success: true,
      message: `Device ${deviceId} successfully bound to robot ${robotSnap.id} (${authoritativeTenantName})!`,
      deviceToken: rawToken,
      bindingStatus: "bound",
      robotId: robotSnap.id,
      tenantId: authoritativeTenantId,
      tenantName: authoritativeTenantName,
      robotConfig: robotData.config || {},
      mode: robotData.mode || "education",
      language: robotData.language || "bn",
    });
  } catch (err: unknown) {
    return NextResponse.json(
      { error: "INTERNAL_ERROR", message: err instanceof Error ? err.message : "Failed to bind device." },
      { status: 500 }
    );
  }
}
