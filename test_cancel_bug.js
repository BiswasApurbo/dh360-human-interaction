const http = require('http');

const token = "super-secret-demo-token";
const headers = {
  "Authorization": "Bearer " + token,
  "Content-Type": "application/json"
};

async function run() {
  // Try calling POST first
  let res = await fetch("http://127.0.0.1:8000/authorization/sessions", { method: "POST", headers });
  console.log("POST 1:", res.status, await res.text());

  // Try calling DELETE
  res = await fetch("http://127.0.0.1:8000/authorization/sessions/current", { method: "DELETE", headers });
  console.log("DELETE AUTH:", res.status, await res.text());

  // Try calling DELETE Enroll
  res = await fetch("http://127.0.0.1:8000/persons/enrollment/sessions/current", { method: "DELETE", headers });
  console.log("DELETE ENROLL:", res.status, await res.text());

  // Try calling POST again
  res = await fetch("http://127.0.0.1:8000/authorization/sessions", { method: "POST", headers });
  console.log("POST 2:", res.status, await res.text());
}
run();
