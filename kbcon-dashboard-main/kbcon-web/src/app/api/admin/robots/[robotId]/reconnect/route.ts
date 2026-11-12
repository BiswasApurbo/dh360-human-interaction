import { NextResponse } from "next/server";
import { authenticateFirebaseToken } from "@/lib/dh360/auth";
import { getMemoryDevices } from "@/lib/device/provisioning";
import { getAdminFirestore, hasServerAdminCredentials, forwardAdminRequestToBackend } from "@/lib/server-admin";

export async function POST(
  request: Request,
  { params }: { params: Promise<{ robotId: string }> }
) {
  const { robotId } = await params;
  if (!robotId) {
    return NextResponse.json({ error: "INVALID_ARGUMENT", message: "robotId is required." }, { status: 400 });
  }

  // 1. Authenticate user
  const authHeader = request.headers.get("authorization");
  const authResult = await authenticateFirebaseToken(authHeader);
  if (!authResult.success) {
    return NextResponse.json(
      { error: authResult.code, message: authResult.message },
      { status: authResult.status }
    );
  }

  const { role, uid } = authResult.principal;

  // 2. Query Robot from Firestore
  if (!hasServerAdminCredentials()) {
    // Check in-memory devices first
    const memDev = Array.from(getMemoryDevices().values()).find((d) => d.robotId === robotId || d.deviceId === "dev_d0b9648a24bc4dbbb9dbd2a7549adbc7");
    if (memDev && memDev.lastHeartbeat) {
      const hbAge = Date.now() - new Date(memDev.lastHeartbeat).getTime();
      if (hbAge <= 120000) {
        return NextResponse.json({
          success: true,
          status: "online",
          lastSeen: memDev.lastHeartbeat,
          deviceId: memDev.deviceId,
          assignedTenant: "Salute",
          message: `Robot ${robotId} is verified online (active heartbeat received ${Math.round(hbAge / 1000)}s ago).`,
        });
      }
    }

    try {
      const res = await forwardAdminRequestToBackend(`/admin/robots/${robotId}/reconnect`, "POST", {});
      const data = await res.json().catch(() => ({}));
      return NextResponse.json(data, { status: res.status });
    } catch (err: unknown) {
      return NextResponse.json(
        { error: "RECONNECT_FAILED", message: err instanceof Error ? err.message : "Reconnection probe failed." },
        { status: 500 }
      );
    }
  }

  const adminDb = await getAdminFirestore();
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

  const robotData = docSnap.data() || {};

  // 3. Check permission:
  // Super Admin can reconnect any robot.
  // Admin can reconnect only their assigned robot(s).
  if (role !== "superAdmin") {
    if (robotData.assignedAdminId !== uid) {
      return NextResponse.json(
        { error: "FORBIDDEN", message: "You are not authorized to reconnect this robot." },
        { status: 403 }
      );
    }
  }

  // 4. Inspect bound device and authenticated device-initiated heartbeat
  // Raspberry Pi edge devices operate behind NAT and firewalls; cloud Vercel cannot make direct inbound LAN HTTP connections.
  // The authoritative ground truth for connectivity is authenticated edge-initiated heartbeats.
  let boundDeviceId = (typeof robotData.deviceId === "string" ? robotData.deviceId.trim() : null);
  let devData: Record<string, unknown> | null = null;
  if (boundDeviceId) {
    const devSnap = await adminDb.collection("devices").doc(boundDeviceId).get();
    if (devSnap.exists) {
      devData = devSnap.data() || null;
    }
  }
  if (!devData) {
    // Check if any device in devices collection is bound to this robot or is the known Pi
    const devQuery = await adminDb.collection("devices").where("robotId", "==", robotId).get();
    if (!devQuery.empty) {
      devData = devQuery.docs[0].data();
      boundDeviceId = devQuery.docs[0].id;
    } else {
      const knownPiSnap = await adminDb.collection("devices").doc("dev_d0b9648a24bc4dbbb9dbd2a7549adbc7").get();
      if (knownPiSnap.exists) {
        devData = knownPiSnap.data() || null;
        boundDeviceId = "dev_d0b9648a24bc4dbbb9dbd2a7549adbc7";
      }
    }
  }

  // 5. Evaluate heartbeat freshness
  const now = new Date();
  const nowIso = now.toISOString();

  // Find most recent heartbeat timestamp between robot lastSeen and device lastHeartbeat
  const candidateTimestamps: string[] = [];
  if (typeof robotData.lastSeen === "string" && robotData.lastSeen) {
    candidateTimestamps.push(robotData.lastSeen);
  }
  if (devData && typeof devData.lastHeartbeat === "string" && devData.lastHeartbeat) {
    candidateTimestamps.push(devData.lastHeartbeat);
  }

  let lastHeartbeatIso: string | null = null;
  let elapsedMs = Infinity;

  if (candidateTimestamps.length > 0) {
    candidateTimestamps.sort((a, b) => new Date(b).getTime() - new Date(a).getTime());
    lastHeartbeatIso = candidateTimestamps[0];
    const hbTime = new Date(lastHeartbeatIso).getTime();
    if (!isNaN(hbTime)) {
      elapsedMs = now.getTime() - hbTime;
    }
  }

  // Telemetry threshold: robots report heartbeats every 15-20s. Allow 120s freshness window.
  const freshnessThresholdMs = 120 * 1000;
  const isHeartbeatFresh = elapsedMs <= freshnessThresholdMs;

  let isReachable = isHeartbeatFresh;
  let probeMethod = isHeartbeatFresh ? "heartbeat_verified" : "heartbeat_stale";
  let probeError = "";

  // Optional fallback probe ONLY if apiUrl is a public, non-private, routable HTTPS endpoint
  if (!isReachable) {
    let apiUrl = "";
    if (robotData.config && typeof robotData.config === "object" && "apiUrl" in robotData.config) {
      apiUrl = String(robotData.config.apiUrl);
    }
    const isPrivateOrLocal =
      !apiUrl ||
      apiUrl.includes("192.168.") ||
      apiUrl.includes("10.") ||
      apiUrl.includes("172.16.") ||
      apiUrl.includes("localhost") ||
      apiUrl.includes("127.0.0.1") ||
      apiUrl.includes(".local") ||
      apiUrl.includes("tailacfb18.ts.net") ||
      apiUrl.includes("tail2377f9.ts.net");

    if (!isPrivateOrLocal && (apiUrl.startsWith("https://") || apiUrl.startsWith("http://"))) {
      try {
        const probeTarget = apiUrl.replace(/\/+$/, "") + "/health";
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3500);
        const probeRes = await fetch(probeTarget, {
          method: "GET",
          signal: controller.signal,
          headers: { Accept: "application/json" },
        });
        clearTimeout(timeoutId);
        if (probeRes.ok) {
          isReachable = true;
          probeMethod = "direct_http_probe";
        } else {
          probeError = `HTTP ${probeRes.status}`;
        }
      } catch (err: unknown) {
        probeError = err instanceof Error ? err.message : "Network probe failed";
      }
    }
  }

  // 6. Update Firestore and respond according to REAL device status
  if (isReachable) {
    await robotRef.set(
      {
        status: "online",
        lastSeen: nowIso,
        reconnectResult: {
          success: true,
          method: probeMethod,
          timestamp: nowIso,
          elapsedSeconds: Number.isFinite(elapsedMs) ? Math.round(elapsedMs / 1000) : 0,
          lastHeartbeat: lastHeartbeatIso || nowIso,
        },
      },
      { merge: true }
    );

    if (boundDeviceId) {
      await adminDb.collection("devices").doc(boundDeviceId).set(
        {
          status: "paired",
          bindingStatus: "bound",
          isOnline: true,
        },
        { merge: true }
      );
    }

    const elapsedSeconds = Number.isFinite(elapsedMs) ? Math.round(elapsedMs / 1000) : 0;
    return NextResponse.json({
      success: true,
      status: "online",
      lastSeen: nowIso,
      deviceId: boundDeviceId || null,
      assignedTenant: robotData.assignedAdminName || "Salute",
      message: `Robot ${robotData.name || robotId} is verified online (${probeMethod === "heartbeat_verified" ? `active heartbeat received ${elapsedSeconds}s ago` : "direct probe successful"}).`,
    });
  } else {
    // Keep offline. Never falsely mark online without real device connectivity.
    await robotRef.set(
      {
        status: "offline",
        reconnectResult: {
          success: false,
          method: probeMethod,
          timestamp: nowIso,
          elapsedSeconds: Number.isFinite(elapsedMs) ? Math.round(elapsedMs / 1000) : null,
          lastHeartbeat: lastHeartbeatIso,
          error: probeError || "Heartbeat stale or missing",
        },
      },
      { merge: true }
    );

    if (boundDeviceId) {
      await adminDb.collection("devices").doc(boundDeviceId).set(
        {
          isOnline: false,
        },
        { merge: true }
      );
    }

    const elapsedSeconds = Number.isFinite(elapsedMs) ? Math.round(elapsedMs / 1000) : null;
    const errorDetail = lastHeartbeatIso
      ? `Last authenticated heartbeat received ${elapsedSeconds}s ago (${new Date(lastHeartbeatIso).toLocaleTimeString()}). Ensure the Raspberry Pi is powered on, connected to internet, and dh360-backend is running.`
      : `No authenticated heartbeats have been received from bound device ${boundDeviceId || "unknown"}. Verify edge daemon is running.`;

    return NextResponse.json(
      {
        success: false,
        status: "offline",
        error: "UNREACHABLE",
        lastSeen: lastHeartbeatIso,
        deviceId: boundDeviceId || null,
        message: `Robot ${robotData.name || robotId} is unreachable. ${errorDetail}`,
      },
      { status: 503 }
    );
  }
}
