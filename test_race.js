const token = "super-secret-demo-token";
const headers = { "Authorization": "Bearer " + token, "Content-Type": "application/json" };

async function run() {
  await fetch("http://127.0.0.1:8000/authorization/sessions", { method: "POST", headers });
  
  // Fire both at the same time!
  const p1 = fetch("http://127.0.0.1:8000/authorization/sessions/current", { method: "DELETE", headers });
  const p2 = fetch("http://127.0.0.1:8000/authorization/sessions", { method: "POST", headers });
  
  const r1 = await p1;
  const r2 = await p2;
  
  console.log("DELETE:", r1.status, await r1.text());
  console.log("POST:", r2.status, await r2.text());
}
run();
