import { NextResponse } from "next/server";
import { authenticateFirebaseToken } from "@/lib/dh360/auth";
import { POST as pairPost } from "../pair/route";
import { POST as revokePost } from "../revoke/route";

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
  const code = typeof body.code === "string" ? body.code.trim() : "";
  const robotId = typeof body.robotId === "string" ? body.robotId.trim() : "";

  if (!code || !robotId) {
    return NextResponse.json(
      { error: "INVALID_ARGUMENT", message: "Pairing code and target robotId are required." },
      { status: 400 }
    );
  }

  // 1. If oldDeviceId provided, revoke it first
  if (oldDeviceId) {
    const revokeReq = new Request("http://localhost/api/admin/devices/revoke", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: authHeader || "",
      },
      body: JSON.stringify({ deviceId: oldDeviceId }),
    });
    await revokePost(revokeReq);
  }

  // 2. Pair new device with pairing code
  const pairReq = new Request("http://localhost/api/admin/devices/pair", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: authHeader || "",
    },
    body: JSON.stringify({ code, robotId }),
  });

  return pairPost(pairReq);
}
