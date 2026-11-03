import { NextResponse } from "next/server";
import { authenticateFirebaseToken } from "@/lib/dh360/auth";
import { getAdminFirestore, hasServerAdminCredentials, forwardAdminRequestToBackend } from "@/lib/server-admin";
import { getDh360Config } from "@/lib/dh360/config";

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

  // 4. Resolve edge API URL safely (anti-SSRF and no private LAN probing from cloud)
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
    apiUrl.includes("trycloudflare.com");

  if (isPrivateOrLocal) {
    try {
      const config = getDh360Config();
      apiUrl = config.apiUrl;
    } catch {
      apiUrl = "https://raspberrypi.tailacfb18.ts.net";
    }
  }

  // 5. Perform genuine health probe to the robot
  const now = new Date().toISOString();
  const probeTarget = apiUrl.replace(/\/+$/, "") + "/health";
  let isReachable = false;
  let probeError = "";

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 4500); // 4.5s probe timeout

    const probeRes = await fetch(probeTarget, {
      method: "GET",
      signal: controller.signal,
      headers: { Accept: "application/json" },
    });
    clearTimeout(timeoutId);

    if (probeRes.ok) {
      isReachable = true;
    } else {
      probeError = `Robot returned HTTP ${probeRes.status}`;
    }
  } catch (err: unknown) {
    if (err instanceof Error) {
      if (err.name === "AbortError") {
        probeError = "Connection timed out after 4.5s.";
      } else {
        probeError = err.message;
      }
    } else {
      probeError = "Network connection failed.";
    }
  }

  // 6. Update Firestore and respond according to REAL device status
  let host = apiUrl;
  try {
    host = new URL(probeTarget).hostname || apiUrl;
  } catch {
    // Keep host
  }

  if (isReachable) {
    await robotRef.set(
      {
        status: "online",
        lastSeen: now,
        reconnectResult: {
          success: true,
          timestamp: now,
          probeTarget,
        },
      },
      { merge: true }
    );

    return NextResponse.json({
      success: true,
      status: "online",
      lastSeen: now,
      message: `Robot ${robotData.name || robotId} reconnected successfully. Device is healthy.`,
    });
  } else {
    // Keep offline. Never falsely mark online without real device connectivity.
    await robotRef.set(
      {
        reconnectResult: {
          success: false,
          error: probeError,
          timestamp: now,
          probeTarget,
        },
      },
      { merge: true }
    );

    return NextResponse.json(
      {
        success: false,
        status: "offline",
        error: "UNREACHABLE",
        message: `Robot at ${host} is unreachable (${probeError}). Verify that the robot is powered on, connected to the local network, and dh360-backend is running.`,
      },
      { status: 503 }
    );
  }
}
