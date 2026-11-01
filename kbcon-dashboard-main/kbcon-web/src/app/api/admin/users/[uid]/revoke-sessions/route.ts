import { NextResponse } from "next/server";
import { authenticateFirebaseToken } from "@/lib/dh360/auth";
import { getAdminAuth } from "@/lib/server-admin";

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

  if (authResult.principal.role !== "superAdmin") {
    return NextResponse.json(
      { error: "FORBIDDEN", message: "Only Super Admin can revoke sessions." },
      { status: 403 }
    );
  }

  try {
    const adminAuth = getAdminAuth();
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
