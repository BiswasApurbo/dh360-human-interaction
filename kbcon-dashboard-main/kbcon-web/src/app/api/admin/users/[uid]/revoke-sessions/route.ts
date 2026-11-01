import { NextResponse } from "next/server";
import { authenticateFirebaseToken } from "@/lib/dh360/auth";
import {
  forwardAdminRequestToBackend,
  getAdminAuth,
  hasServerAdminCredentials,
} from "@/lib/server-admin";

interface RouteParams {
  params: Promise<{ uid: string }> | { uid: string };
}

export async function POST(request: Request, context: RouteParams) {
  const { uid } = await Promise.resolve(context.params);
  const authHeader = request.headers.get("authorization");
  const authResult = await authenticateFirebaseToken(authHeader);

  if (!authResult.success) {
    return NextResponse.json(
      { error: authResult.code, message: authResult.message },
      { status: authResult.status }
    );
  }

  // Enforce mandatory password change for bootstrap accounts
  if (authResult.principal.mustChangePassword) {
    return NextResponse.json(
      {
        error: "PASSWORD_CHANGE_REQUIRED",
        message: "Mandatory password change required before accessing administrative endpoints.",
      },
      { status: 403 }
    );
  }

  if (authResult.principal.role !== "superAdmin") {
    return NextResponse.json(
      { error: "FORBIDDEN", message: "Only Super Admin can revoke sessions." },
      { status: 403 }
    );
  }

  if (!hasServerAdminCredentials()) {
    try {
      const res = await forwardAdminRequestToBackend(`/admin/users/${uid}/revoke-sessions`, "POST");
      const data = await res.json();
      return NextResponse.json(data, { status: res.status });
    } catch (err: unknown) {
      return NextResponse.json(
        { error: "REVOCATION_FAILED", message: err instanceof Error ? err.message : "Failed to revoke sessions." },
        { status: 500 }
      );
    }
  }

  try {
    const adminAuth = await getAdminAuth();
    await adminAuth.revokeRefreshTokens(uid);

    return NextResponse.json({
      success: true,
      message: `All refresh tokens and sessions revoked for user ${uid}.`,
    });
  } catch (err: unknown) {
    return NextResponse.json(
      { error: "REVOCATION_FAILED", message: err instanceof Error ? err.message : "Failed to revoke sessions." },
      { status: 500 }
    );
  }
}
