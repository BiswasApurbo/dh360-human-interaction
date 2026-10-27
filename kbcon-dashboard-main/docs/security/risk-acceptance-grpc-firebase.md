# [PROPOSED / PENDING APPROVAL] Security Risk Acceptance & Waiver Record

> **STATUS:** **PROPOSED / PENDING APPROVAL**  
> **DOCUMENT TYPE:** Security Risk Acceptance & Waiver Proposal  
> **RECORD ID:** SEC-WAIVER-2026-001  
> **TARGET PROJECT:** `kbcon-dashboard-main/kbcon-web`  
> **CREATED DATE:** 2026-10-26  
> **REVIEW / EXPIRATION DATE:** 2026-12-26 (60 days from creation, or upon upstream release)  
> **OWNER:** KBCON / DH360 Security & Integration Engineering  

---

## 1. Context & Scope

Following the successful application and verification of the Next.js (`next@16.4.3`) and Shadcn/tooling (`shadcn@4.22.1`, `body-parser@2.4.0`, `undici@7.30.1`, `fast-uri@3.1.9`) security patches, `npm audit` reports exactly **4 High severity findings** in `kbcon-web`. 

All four findings propagate exclusively through a single transitive dependency chain rooted in `firebase@13.0.0`. This document records the technical investigation, proposed risk acceptance rationale, residual risks, compensating controls, and sunset criteria.

---

## 2. Vulnerability Details & Dependency Mapping

### 2.1 Dependency Chain
```text
kbcon-web@0.1.0
└── firebase@13.0.0 (Direct production dependency)
    ├─┬ @firebase/firestore-compat@0.4.15
    │ └── @firebase/firestore@4.18.0 (deduped)
    └─┬ @firebase/firestore@4.18.0
      └── @grpc/grpc-js@1.9.16 (Vulnerable component)
```

### 2.2 Security Advisories
1. **GHSA-m9gg-hp2v-232j (CVE-2026-101916)**
   * **Package:** `@grpc/grpc-js`
   * **Installed Version:** `1.9.16`
   * **Vulnerable Range:** `< 1.13.6`
   * **Severity / CVSS:** **High** (CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:H/A:N — Base Score: 7.4)
   * **CWE:** CWE-295 (Improper Certificate Validation)
   * **Description:** When gRPC server credentials are created with `requireClientCertificate: false`, `getAuthContext` fails to properly distinguish between authorized and unauthorized peer certificates, potentially allowing authentication/authorization bypass on services using mTLS with custom RBAC verification.
   * **Patched In:** `@grpc/grpc-js >= 1.13.6` or `>= 1.14.5`

2. **GHSA-f596-whhp-79r4 (CVE-2026-101915)**
   * **Package:** `@grpc/grpc-js`
   * **Installed Version:** `1.9.16`
   * **Vulnerable Range:** `< 1.13.6`
   * **Severity / CVSS:** **Low** (CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:L/I:N/A:N — Base Score: 3.7)
   * **CWE:** CWE-550 (Information Disclosure)
   * **Description:** In gRPC server implementations, unhandled exceptions thrown by method handlers may inadvertently serialize internal error messages and stack traces into status messages transmitted back to connecting clients.
   * **Patched In:** `@grpc/grpc-js >= 1.13.6` or `>= 1.14.5`

---

## 3. Technical Assessment & Rationale for Waiver

### 3.1 Application Topology vs. Threat Vector
* **Server vs. Client Functionality:** Both advisories explicitly describe vulnerabilities in **gRPC Server** components (`new grpc.Server()`, incoming mTLS certificate evaluation via `getAuthContext`, and server method exception handling).
* **KBCON Dashboard Topology:** 
  * The KBCON dashboard is a Next.js web application and Backend-for-Frontend (BFF).
  * In browser client bundles, the Firebase JS SDK communicates via WebChannel (HTTPS/WebSockets); `@grpc/grpc-js` is not bundled into client-side code.
  * In Node.js server environments, `@firebase/firestore` acts solely as an **outbound gRPC client** communicating with Google Cloud Firestore endpoints (`firestore.googleapis.com:443`).
  * **The dashboard never initializes or binds a gRPC server**, exposes zero incoming gRPC ports, and implements no gRPC method handlers.

### 3.2 Infeasibility of Upstream Resolution at Present
1. **Automatic Fix is Destructive:** `npm audit fix --force` attempts to downgrade `firebase` to `9.14.0` (released in 2022). This would break React 19 compatibility, Next.js 16.4.3 Turbopack module resolution, and all modern modular Firebase v13 imports across the dashboard.
2. **Latest Firebase SDK Still Affected:** Manifest inspection of `firebase@13.1.0` and `@firebase/firestore@4.19.0` confirms that the official upstream Firebase SDK still strictly pins `"@grpc/grpc-js": "~1.9.0"`. No official Firebase JS SDK version has yet upgraded this dependency constraint.
3. **Manual Overrides Introduce Stability Risk:** Upstream bug trackers report that forcing global npm overrides for `@grpc/grpc-js` (e.g., `^1.13.7` or `^1.14.5`) causes duplicate package nesting, type definition mismatches, and connection channel initialization failures within `@firebase/firestore`.

---

## 4. Technical Uncertainties & Residual Risk

While analysis indicates low practical exploitability in standard client operations, the following uncertainties are noted:
* **Transitive Node.js Execution:** In server-side rendering (SSR) or API route handlers where Firebase is imported, the `@grpc/grpc-js` codebase is loaded into Node's process memory. While server-specific initialization methods are not invoked by our code, we cannot guarantee that no internal transitive helper utility touches affected code paths.
* **Lack of 1.9.x Backport:** The gRPC maintainers did not backport the fixes to the 1.9.x release train; the vulnerability will remain permanently flagged on `1.9.16` until `@firebase/firestore` adopts a newer minor version.

---

## 5. Compensating Security Controls

1. **Deny-by-Default BFF Boundary (Verified):** All access to the DH360 backend is mediated by the Next.js BFF (`/api/dh360/[...path]`). The BFF requires verified Firebase ID tokens and strictly enforces role-based access control (Super Admin vs. Admin vs. Viewer). Verified by 23 passing tests in `tests/test_dh360_bff.mjs`.
2. **Cryptographic Authentication via Trusted APIs (Verified):** Firebase authentication is cryptographically verified using Google's public x509 certificates and REST/identitytoolkit token verification endpoints rather than gRPC.
3. **Application Topology: No Inbound gRPC Listener Identified (Verified); Network-Level Ingress Restrictions (Not Verified):** Inspection of `kbcon-web` confirms it operates as a standard Next.js web application and defines no gRPC server listeners or service bindings. However, network-level ingress firewall rules, port forwarding, or security groups have not been inspected or confirmed within the repository and are deployment-dependent.
4. **Egress Isolation (Not Verified / Planned Deployment Requirement):** No egress filtering, outbound proxy, or firewall rules have been verified in the local workspace or repository configuration. The Node.js runtime currently has standard unrestricted outbound network access. Egress isolation is tracked as a planned deployment hardening requirement rather than an active compensating control.

---

## 6. Sunset Criteria & Action Trigger

This risk acceptance proposal is strictly time-limited and will be reviewed upon either of the following events:
1. **Upstream Release:** Google releases an official update to `@firebase/firestore` (or `firebase`) that updates its dependency constraint to `@grpc/grpc-js >= 1.13.6`.
2. **Review Date (2026-12-26):** If no upstream update has been released within 60 days, the security owner must re-evaluate whether an isolated, thoroughly-tested override can be safely introduced without breaking Firestore client runtime behavior.

---

## 7. Separate Security Track: Credential Rotation & History Scrubbing

The following remediation items remain tracked as a **separate, independent security workstream** and are not covered by this dependency waiver:

1. **Provider API Key Rotation:**
   * Rotate **Google Gemini / Google AI Studio API Key** on the provider console.
   * Rotate **Soniox Speech Recognition API Key** on the provider console.
   * Rotate **Firebase Service Account / Private Keys** on the Google Cloud / Firebase console if previously committed.
2. **Git History Scrubbing:**
   * Perform a coordinated maintenance pass (using `git-filter-repo` or BFG) to rewrite historical commit objects that contained hardcoded credentials prior to the Phase 0.5 remediation.
   * Coordinate with any remote repository forks before force-updating branch references.

*(Secret values are strictly excluded from all project documentation).*

---

## 8. Approval & Sign-Off (Pending)

| Role | Name / Title | Decision | Date |
| :--- | :--- | :---: | :---: |
| **Security Engineer** | Antigravity AI Assistant | Proposed | 2026-10-26 |
| **Project Owner / Tech Lead** | User / Project Administrator | **[ PENDING APPROVAL ]** | — |
