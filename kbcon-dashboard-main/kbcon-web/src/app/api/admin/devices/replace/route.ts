import { NextResponse } from "next/server";
import { authenticateFirebaseToken } from "@/lib/dh360/auth";
import { POST as bindPost } from "../bind/route";
import { POST as unbindPost } from "../unbind/route";

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
      { error: "FORBIDDEN", message: "Only Super Admin can replace paired devices." },
      { status: 403 }
    );
  }

  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: "BAD_REQUEST", message: "Invalid JSON body." },
      { status: 400 }
    );
  }

  const oldDeviceId = typeof body.oldDeviceId === "string" ? body.oldDeviceId.trim() : "";
  const newDeviceId = typeof (body.newDeviceId || body.deviceId) === "string"
    ? String(body.newDeviceId || body.deviceId).trim()
    : "";
  const robotId = typeof body.robotId === "string" ? body.robotId.trim() : "";

  if (!newDeviceId || !robotId) {
    return NextResponse.json(
      { error: "INVALID_ARGUMENT", message: "New deviceId and target robotId are required." },
      { status: 400 }
    );
  }

  // 1. If oldDeviceId provided, unbind it first
  if (oldDeviceId) {
    const unbindReq = new Request("http://localhost/api/admin/devices/unbind", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: authHeader || "",
      },
      body: JSON.stringify({ deviceId: oldDeviceId }),
    });
    await unbindPost(unbindReq);
  }

  // 2. Bind new device directly
  const bindReq = new Request("http://localhost/api/admin/devices/bind", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: authHeader || "",
    },
    body: JSON.stringify({ deviceId: newDeviceId, robotId }),
  });

  return bindPost(bindReq);
}
