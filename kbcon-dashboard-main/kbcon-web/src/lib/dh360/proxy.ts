/**
 * DH360 Request Forwarding & Streaming Proxy Engine.
 *
 * Implements:
 * 1. Server-side token validation & role-based access control.
 * 2. SSRF prevention and strict path boundary checks.
 * 3. Operator Bearer token injection (held exclusively on server).
 * 4. Streaming pipes for Server-Sent Events (SSE) and binary JPEG frames.
 * 5. Sanitized error handling preventing credential leakage.
 */

import { authenticateFirebaseToken } from "./auth.ts";
import { getDh360Config } from "./config.ts";
import { checkPermission } from "./permissions.ts";

const MAX_BODY_BYTES = 1024 * 1024; // 1MB limit for JSON bodies

/**
 * Handle incoming Next.js API requests to the DH360 BFF.
 */
export async function handleDh360ProxyRequest(
  request: Request,
  rawPathSegments: string[]
): Promise<Response> {
  const method = request.method.toUpperCase();

  // 1. Path normalization and traversal protection
  const subpath = "/" + rawPathSegments.join("/").replace(/\/+/g, "/");
  if (subpath.includes("..") || subpath.includes("\\")) {
    return Response.json(
      { error: "BAD_REQUEST", message: "Invalid path segments." },
      { status: 400 }
    );
  }

  // 2. Authenticate the caller's Firebase identity
  const authHeader = request.headers.get("authorization");
  const authResult = await authenticateFirebaseToken(authHeader);
  if (!authResult.success) {
    return Response.json(
      {
        error: authResult.code,
        message: authResult.message,
      },
      {
        status: authResult.status,
        ...(authResult.status === 401
          ? { headers: { "WWW-Authenticate": "Bearer" } }
          : {}),
      }
    );
  }

  const principal = authResult.principal;

  // 3. Inspect request body for PATCH / POST
  let requestBodyObj: Record<string, unknown> | null = null;
  let rawBodyText: string | null = null;

  if (method === "POST" || method === "PATCH") {
    const contentType = request.headers.get("content-type") || "";
    const contentLength = parseInt(request.headers.get("content-length") || "0", 10);

    // If a body is present or advertised, enforce application/json
    if (contentLength > 0 || contentType) {
      if (!contentType.toLowerCase().includes("application/json")) {
        return Response.json(
          {
            error: "UNSUPPORTED_MEDIA_TYPE",
            message: "Content-Type must be application/json for request bodies.",
          },
          { status: 415 }
        );
      }

      try {
        const text = await request.text();
        if (text.length > MAX_BODY_BYTES) {
          return Response.json(
            { error: "PAYLOAD_TOO_LARGE", message: "Request body exceeds 1MB limit." },
            { status: 413 }
          );
        }
        rawBodyText = text;
        if (text.trim().length > 0) {
          requestBodyObj = JSON.parse(text) as Record<string, unknown>;
        }
      } catch {
        return Response.json(
          { error: "INVALID_JSON", message: "Malformed JSON payload." },
          { status: 400 }
        );
      }
    }
  }

  // 4. Validate permissions against the explicit allowlist
  const permCheck = checkPermission(subpath, method, principal, requestBodyObj);
  if (!permCheck.allowed) {
    return Response.json(
      {
        error: permCheck.httpStatus === 403 ? "FORBIDDEN" : "NOT_FOUND",
        message: permCheck.reason || "Operation not permitted.",
      },
      { status: permCheck.httpStatus || 403 }
    );
  }

  // 5. Construct upstream destination URL safely
  const config = getDh360Config();
  const incomingUrl = new URL(request.url);
  const upstreamUrl = new URL(subpath, config.apiUrl);
  upstreamUrl.search = incomingUrl.search; // Preserve query parameters safely

  // Verify that destination does not deviate from the configured API origin (Anti-SSRF)
  if (upstreamUrl.origin !== config.apiUrl) {
    return Response.json(
      { error: "SECURITY_VIOLATION", message: "Target origin mismatch." },
      { status: 400 }
    );
  }

  // 6. Build upstream request headers
  const upstreamHeaders = new Headers();
  upstreamHeaders.set("Authorization", `Bearer ${config.operatorToken}`);

  // Forward client Accept header (crucial for text/event-stream and image/jpeg)
  const accept = request.headers.get("accept");
  if (accept) {
    upstreamHeaders.set("Accept", accept);
  }

  // Forward Content-Type if present
  if (request.headers.get("content-type")) {
    upstreamHeaders.set("Content-Type", request.headers.get("content-type")!);
  }

  // Forward Last-Event-ID for SSE resumption
  const lastEventId = request.headers.get("last-event-id");
  if (lastEventId) {
    upstreamHeaders.set("Last-Event-ID", lastEventId);
  }

  // 7. Execute upstream fetch with appropriate timeout and client disconnect linking
  const isSse = subpath === "/interaction/events" || accept?.includes("text/event-stream");
  const timeoutController = new AbortController();
  const timeoutMs = isSse ? 0 : 15000;
  let timeoutId: NodeJS.Timeout | null = null;

  if (timeoutMs > 0) {
    timeoutId = setTimeout(() => timeoutController.abort(), timeoutMs);
  }

  // Abort upstream immediately if either the timeout fires or client disconnects
  const fetchSignal = request.signal
    ? AbortSignal.any([request.signal, timeoutController.signal])
    : timeoutController.signal;

  try {
    const upstreamResponse = await fetch(upstreamUrl.toString(), {
      method,
      headers: upstreamHeaders,
      body: rawBodyText !== null ? rawBodyText : undefined,
      signal: fetchSignal,
    });

    if (timeoutId) clearTimeout(timeoutId);

    // 8. Stream or forward response based on content-type
    const contentType = upstreamResponse.headers.get("content-type") || "";

    // SSE Stream forwarding
    if (contentType.includes("text/event-stream")) {
      return new Response(upstreamResponse.body, {
        status: upstreamResponse.status,
        headers: {
          "Content-Type": "text/event-stream; charset=utf-8",
          "Cache-Control": "no-cache, no-transform",
          "Connection": "keep-alive",
          "X-Accel-Buffering": "no",
        },
      });
    }

    // Binary Image Stream forwarding (JPEG preview)
    if (contentType.includes("image/jpeg") || contentType.includes("image/")) {
      return new Response(upstreamResponse.body, {
        status: upstreamResponse.status,
        headers: {
          "Content-Type": contentType,
          "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        },
      });
    }

    // Handle null-body HTTP statuses (204 No Content, 205 Reset Content, 304 Not Modified)
    if (upstreamResponse.status === 204 || upstreamResponse.status === 205 || upstreamResponse.status === 304) {
      return new Response(null, {
        status: upstreamResponse.status,
      });
    }

    // Standard JSON / text forwarding
    const responseBody = await upstreamResponse.text();
    return new Response(responseBody, {
      status: upstreamResponse.status,
      headers: {
        "Content-Type": contentType || "application/json",
      },
    });
  } catch (err: unknown) {
    if (timeoutId) clearTimeout(timeoutId);

    const isAbort =
      err instanceof Error &&
      (err.name === "AbortError" || err.name === "TimeoutError");

    if (isAbort) {
      return Response.json(
        {
          error: "GATEWAY_TIMEOUT",
          message: "DH360 upstream service timed out.",
        },
        { status: 504 }
      );
    }

    // Upstream connection error / service unreachable
    return Response.json(
      {
        error: "UPSTREAM_UNAVAILABLE",
        message: "DH360 Human Interaction service is currently unreachable.",
      },
      { status: 503 }
    );
  }
}
