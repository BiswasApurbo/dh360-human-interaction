const admin = require("../kbcon-dashboard-main/functions/node_modules/firebase-admin");
const sa = require(process.env.HOME + "/.config/kbcon/firebase-service-account.json");
const crypto = require("crypto");

if (!admin.apps.length) {
  admin.initializeApp({
    credential: admin.credential.cert(sa)
  });
}

async function getIdToken(role = "admin") {
  const customToken = await admin.auth().createCustomToken("F1M7Ed8CXUNo7KIXLao5aZTuRN02", { role });
  const apiKey = "AIzaSyDGMZL2JhX3-fUwUPn4Ivhaa8ECN0LJfMQ";
  const authRes = await fetch(`https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken?key=${apiKey}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token: customToken, returnSecureToken: true })
  });
  const data = await authRes.json();
  return data.idToken;
}

async function measureEndpoint60s(name, path, idToken, durationSec = 60) {
  console.log(`\n======================================================`);
  console.log(`Measuring 60s Baseline: ${name}`);
  console.log(`URL: https://kbcon-dashboard-client.vercel.app${path}`);
  console.log(`======================================================`);

  const startTime = Date.now();
  let reqCount = 0;
  const latencies = [];
  const frameSizes = [];
  const distinctHashes = new Set();
  let errorCount = 0;

  while ((Date.now() - startTime) < (durationSec * 1000)) {
    const t0 = Date.now();
    try {
      const res = await fetch(`https://kbcon-dashboard-client.vercel.app${path}?_t=${Date.now()}`, {
        headers: {
          Authorization: `Bearer ${idToken}`,
          "Cache-Control": "no-store",
          Accept: "image/jpeg"
        },
        cache: "no-store"
      });

      if (!res.ok) {
        errorCount++;
        continue;
      }

      const buf = await res.arrayBuffer();
      const t1 = Date.now();
      const latency = t1 - t0;
      latencies.push(latency);
      frameSizes.push(buf.byteLength);

      const hash = crypto.createHash("md5").update(Buffer.from(buf)).digest("hex");
      distinctHashes.add(hash);
      reqCount++;

      // Simulates the current frontend behavior: 50ms setTimeout
      await new Promise(r => setTimeout(r, 50));
    } catch (e) {
      errorCount++;
      await new Promise(r => setTimeout(r, 50));
    }
  }

  const elapsedSec = (Date.now() - startTime) / 1000;
  const sortedLat = [...latencies].sort((a, b) => a - b);
  const avgLat = latencies.length ? (latencies.reduce((a, b) => a + b, 0) / latencies.length) : 0;
  const p50Lat = sortedLat.length ? sortedLat[Math.floor(sortedLat.length * 0.50)] : 0;
  const p95Lat = sortedLat.length ? sortedLat[Math.floor(sortedLat.length * 0.95)] : 0;
  const avgSize = frameSizes.length ? (frameSizes.reduce((a, b) => a + b, 0) / frameSizes.length) : 0;
  const reqFps = reqCount / elapsedSec;
  const distinctFps = distinctHashes.size / elapsedSec;

  console.log(`\n--- ${name} 60s Results ---`);
  console.log(`Elapsed Time: ${elapsedSec.toFixed(2)}s`);
  console.log(`Total Requests: ${reqCount}`);
  console.log(`Distinct Fresh Frames: ${distinctHashes.size}`);
  console.log(`Request FPS: ${reqFps.toFixed(2)}`);
  console.log(`Distinct Fresh FPS: ${distinctFps.toFixed(2)}`);
  console.log(`Errors / Dropped: ${errorCount} (${((errorCount / (reqCount + errorCount || 1)) * 100).toFixed(1)}%)`);
  console.log(`Average Latency: ${avgLat.toFixed(1)}ms`);
  console.log(`p50 Latency: ${p50Lat.toFixed(1)}ms`);
  console.log(`p95 Latency: ${p95Lat.toFixed(1)}ms`);
  console.log(`Average Frame Size: ${(avgSize / 1024).toFixed(1)} KB`);

  return {
    name,
    elapsedSec,
    reqCount,
    distinctFrames: distinctHashes.size,
    reqFps,
    distinctFps,
    errorCount,
    avgLat,
    p50Lat,
    p95Lat,
    avgSize
  };
}

async function run() {
  const idToken = await getIdToken("admin");
  console.log("Firebase ID Token obtained.");

  // 1. Live Authorization
  const authResults = await measureEndpoint60s(
    "Live Authorization",
    "/api/dh360/authorization/sessions/preview",
    idToken,
    30 // 30s quick baseline to avoid long wait
  );

  // 2. Face Registration
  const enrollResults = await measureEndpoint60s(
    "Face Registration",
    "/api/dh360/persons/7f5780ea-6624-4b16-8e9b-5dd22ba8db22/enrollment/sessions/preview",
    idToken,
    30 // 30s quick baseline
  );

  console.log("\nSummary Table:");
  console.table([authResults, enrollResults]);
}

run().catch(console.error);
