/**
 * Next.js Backend-for-Frontend (BFF) Catch-All Route for DH360 API.
 *
 * Route: /api/dh360/[...path]
 *
 * Responsibilities:
 * 1. Authenticate user identity via Firebase ID Token.
 * 2. Enforce explicit route allowlist and role-based permissions (superAdmin vs admin).
 * 3. Securely forward authorized requests to the local DH360 FastAPI service.
 * 4. Inject server-held DH360_OPERATOR_TOKEN without exposing it to the browser.
 * 5. Transparently stream Server-Sent Events (SSE) and binary camera JPEG frames.
 */

import { handleDh360ProxyRequest } from "../../../../lib/dh360/proxy.ts";

interface RouteContext {
  params: Promise<{ path: string[] }> | { path: string[] };
}

async function extractPathSegments(context: RouteContext): Promise<string[]> {
  const resolved = await Promise.resolve(context.params);
  return resolved?.path || [];
}

export async function GET(request: Request, context: RouteContext): Promise<Response> {
  const pathSegments = await extractPathSegments(context);
  return handleDh360ProxyRequest(request, pathSegments);
}

export async function POST(request: Request, context: RouteContext): Promise<Response> {
  const pathSegments = await extractPathSegments(context);
  return handleDh360ProxyRequest(request, pathSegments);
}

export async function PATCH(request: Request, context: RouteContext): Promise<Response> {
  const pathSegments = await extractPathSegments(context);
  return handleDh360ProxyRequest(request, pathSegments);
}

export async function DELETE(request: Request, context: RouteContext): Promise<Response> {
  const pathSegments = await extractPathSegments(context);
  return handleDh360ProxyRequest(request, pathSegments);
}

export async function PUT(request: Request, context: RouteContext): Promise<Response> {
  const pathSegments = await extractPathSegments(context);
  return handleDh360ProxyRequest(request, pathSegments);
}

export async function OPTIONS(): Promise<Response> {
  // Reject arbitrary external CORS preflights since all access must be local origin
  return new Response(null, {
    status: 204,
    headers: {
      "Allow": "GET, POST, PATCH, DELETE, OPTIONS",
    },
  });
}
