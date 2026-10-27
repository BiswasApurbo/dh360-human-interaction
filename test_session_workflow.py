import httpx
import time

URL = "http://127.0.0.1:8000"
HEADERS = {"Authorization": "Bearer super-secret-demo-token"}

def report(scenario, result, evidence):
    print(f"\n[{'PASS' if result else 'FAIL'}] {scenario}")
    print(f"Evidence: {evidence}")

def main():
    time.sleep(1) # wait for server to be ready
    with httpx.Client(base_url=URL, headers=HEADERS) as client:
        # Initial cleanup just in case
        client.delete("/authorization/sessions/current")
        client.delete("/persons/enrollment/sessions/current")

        # Verify appropriate behavior when there is no active session to cancel.
        res = client.delete("/authorization/sessions/current")
        report("Verify appropriate behavior when there is no active session to cancel (Auth)",
               res.status_code == 409 and "No active" in res.json()["detail"],
               f"Status: {res.status_code}, Body: {res.text}")

        res = client.delete("/persons/enrollment/sessions/current")
        report("Verify appropriate behavior when there is no active session to cancel (Enroll)",
               res.status_code == 409 and "No active" in res.json()["detail"],
               f"Status: {res.status_code}, Body: {res.text}")

        # Start authorization.
        res1 = client.post("/authorization/sessions")
        report("Start authorization",
               res1.status_code == 201,
               f"Status: {res1.status_code}, Body: {res1.text}")

        # Replace an existing authorization session with a new one.
        res_del = client.delete("/authorization/sessions/current")
        res_post = client.post("/authorization/sessions")
        report("Replace an existing authorization session with a new one",
               res_del.status_code == 200 and res_post.status_code == 201,
               f"Delete Status: {res_del.status_code}, Post Status: {res_post.status_code}, Post Body: {res_post.text}")

        # Cancel an active enrollment session.
        # First start an enrollment session (requires a person and consent). Let's create a temp person.
        person_res = client.post("/persons", json={"display_name": "Test Person"})
        person_id = person_res.json()["person_id"]
        # Grant consent
        client.patch(f"/persons/{person_id}", json={"consent_status": "granted"})

        enroll_start_res = client.post(f"/persons/{person_id}/enrollment/sessions", json={})
        
        enroll_cancel_res = client.delete("/persons/enrollment/sessions/current")
        report("Cancel an active enrollment session",
               enroll_start_res.status_code == 201 and enroll_cancel_res.status_code == 200,
               f"Start Status: {enroll_start_res.status_code}, Cancel Status: {enroll_cancel_res.status_code}")

        # Start authorization after cancellation.
        # Wait, if an enrollment session was cancelled, start authorization.
        # Ensure previous auth is cancelled first.
        client.delete("/authorization/sessions/current")
        res_auth_after_cancel = client.post("/authorization/sessions")
        report("Start authorization after cancellation",
               res_auth_after_cancel.status_code == 201,
               f"Status: {res_auth_after_cancel.status_code}")

        # Repeat requests to verify stale sessions do not block new sessions.
        res_stale_1 = client.delete("/authorization/sessions/current")
        res_stale_2 = client.post("/authorization/sessions")
        res_stale_3 = client.delete("/authorization/sessions/current")
        res_stale_4 = client.post("/authorization/sessions")
        report("Repeat requests to verify stale sessions do not block new sessions",
               all(x in (200, 201, 409) for x in (res_stale_1.status_code, res_stale_2.status_code, res_stale_3.status_code, res_stale_4.status_code)) and res_stale_4.status_code == 201,
               f"Statuses: {res_stale_1.status_code}, {res_stale_2.status_code}, {res_stale_3.status_code}, {res_stale_4.status_code}")

        # Cleanup person
        client.delete(f"/persons/{person_id}")

if __name__ == '__main__':
    main()
