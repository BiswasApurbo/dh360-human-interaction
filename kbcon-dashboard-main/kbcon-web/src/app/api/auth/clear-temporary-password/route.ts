import { NextResponse } from "next/server";
import { authenticateFirebaseToken } from "@/lib/dh360/auth";
import {
  forwardAdminRequestToBackend,
  getAdminAuth,
  getAdminFirestore,
  hasServerAdminCredentials,
} from "@/lib/server-admin";

export async function POST(request: Request) {
  const authHeader = request.headers.get("authorization");
  const authResult = await authenticateFirebaseToken(authHeader);

  if (!authResult.success) {
    return NextResponse.json(
      { error: authResult.code, message: authResult.message },
      { status: authResult.status }
    );
  }

  const { uid } = authResult.principal;

  // 1. Direct execution if server credentials are local
  if (hasServerAdminCredentials()) {
    try {
      const adminAuth = getAdminAuth();
      const adminDb = getAdminFirestore();

      const user = await adminAuth.getUser(uid);
      const existingClaims = (user.customClaims as Record<string, unknown>) || {};

      delete existingClaims.mustChangePassword;
      await adminAuth.setCustomUserClaims(uid, existingClaims);

      await adminDb.collection("users").doc(uid).set(
        {
          mustChangePassword: false,
          updatedAt: new Date().toISOString(),
        },
        { merge: true }
      );

      return NextResponse.json({
        success: true,
        message: "Temporary bootstrap status cleared successfully.",
      });
    } catch (err: unknown) {
      console.warn("Local execution failed, attempting backend host fallback:", err);
    }
  }

  // 2. Delegate to DH360 backend on the Raspberry Pi
  try {
    const backendRes = await forwardAdminRequestToBackend(
      "/auth/clear-temporary-password",
      "POST",
      { uid }
    );

    if (backendRes.ok) {
      const data = await backendRes.json();
      return NextResponse.json(data);
    }

    const errJson = await backendRes.json().catch(() => ({}));
    return NextResponse.json(
      {
        error: "UPSTREAM_ERROR",
        message: errJson.detail || "Backend host failed to clear bootstrap status.",
      },
      { status: backendRes.status }
    );
  } catch (err: unknown) {
    return NextResponse.json(
      {
        error: "INTERNAL_ERROR",
        message: err instanceof Error ? err.message : "Failed to clear bootstrap status.",
      },
      { status: 500 }
    );
  }
}
