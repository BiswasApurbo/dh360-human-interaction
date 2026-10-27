  
    let abortController = null;
    let isConnecting = false;
    let isPreflightMode = false;
    let currentPersonId = null;
    let currentAuthId = null;
    let enrollmentIntervalId = null;
    let isEnrollmentCommitted = false;
    let isStepInProgress = false;

    const DIRECTION_INSTRUCTIONS = {
      FRONT: "Look FRONT",
      RIGHT: "Turn RIGHT",
      LEFT: "Turn LEFT"
    };

    function getNowTime() {
      const d = new Date();
      return d.toTimeString().split(' ')[0];
    }

    function appendLog(msg) {
      const box = document.getElementById("log-box");
      const line = `[${getNowTime()}] ${msg}\n`;
      box.textContent += line;
      box.scrollTop = box.scrollHeight;
    }

    function showNotice(msg, type = "error") {
      const n = document.getElementById("notice-box");
      n.textContent = msg;
      n.className = "notice notice-" + type;
      n.style.display = "block";
    }

    function clearNotice() {
      const n = document.getElementById("notice-box");
      n.style.display = "none";
    }

    function escapeHtml(str) {
      if (!str) return "";
      return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    function updateSessionUI(session) {
      if (!session) return;

      const st = (session.status || "IDLE").toUpperCase();
      const statusEl = document.getElementById("auth-status");
      statusEl.textContent = st;
      statusEl.className = "badge badge-" + st.toLowerCase();

      const pName = session.display_name || session.person_id || "-";
      document.getElementById("auth-person").textContent = pName;
      document.getElementById("auth-id").textContent = session.authorization_id || "-";
      document.getElementById("auth-started").textContent = session.started_at || session.created_at || "-";
      document.getElementById("auth-updated").textContent = session.updated_at || "-";
      document.getElementById("auth-ended").textContent = session.ended_at || "-";
    }

    function setSseIndicator(state, text) {
      const dot = document.getElementById("sse-dot");
      const statusTxt = document.getElementById("sse-status");
      dot.className = "sse-indicator sse-" + state;
      statusTxt.textContent = text;
    }

    function addQARow() {
      const tbody = document.querySelector("#qa-table tbody");
      const row = document.createElement("tr");
      row.innerHTML = `
        <td style="padding: 2px 4px;"><input type="text" class="qa-q" placeholder="Question" style="width:100%; font-size:0.8rem; padding:0.3rem 0.5rem;" /></td>
        <td style="padding: 2px 4px;"><input type="text" class="qa-a" placeholder="Answer" style="width:100%; font-size:0.8rem; padding:0.3rem 0.5rem;" /></td>
        <td style="padding: 2px 4px; text-align:center;"><button type="button" onclick="removeQARow(this)" style="background:#dc2626; font-size:0.75rem; padding:0.25rem 0.5rem;">Remove</button></td>
      `;
      tbody.appendChild(row);
    }

    function removeQARow(btn) {
      const tr = btn.closest("tr");
      if (tr) tr.remove();
    }

    async function handleRegister() {
      clearNotice();
      const token = document.getElementById("token-input").value.trim();
      if (!token) {
        showNotice("Please enter an operator Bearer token first.", "error");
        return;
      }

      const displayName = document.getElementById("display-name").value.trim();
      if (!displayName) {
        showNotice("Display Name is required.", "error");
        return;
      }

      const preferredName = document.getElementById("preferred-name").value.trim();
      const role = document.getElementById("role").value.trim();
      const department = document.getElementById("department").value.trim();
      const officeRoom = document.getElementById("office-room").value.trim();
      const workingHours = document.getElementById("working-hours").value.trim();
      const profileDescription = document.getElementById("profile-description").value.trim();

      const qaPairs = [];
      const rows = document.querySelectorAll("#qa-table tbody tr");
      for (const row of rows) {
        const qEl = row.querySelector(".qa-q");
        const aEl = row.querySelector(".qa-a");
        if (qEl && aEl) {
          const q = qEl.value.trim();
          const a = aEl.value.trim();
          if (q && a) {
            qaPairs.push({ question: q, answer: a });
          }
        }
      }

      const payload = {
        display_name: displayName,
        preferred_name: preferredName || null,
        role: role || null,
        department: department || null,
        office_room: officeRoom || null,
        working_hours: workingHours || null,
        profile_description: profileDescription || null,
        qa_pairs: qaPairs.length > 0 ? qaPairs : null
      };

      const registerBtn = document.getElementById("register-btn");
      registerBtn.disabled = true;

      try {
        const resp = await fetch("/persons", {
          method: "POST",
          headers: {
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json"
          },
          body: JSON.stringify(payload)
        });

        if (resp.status === 201) {
          const data = await resp.json();
          currentPersonId = data.person_id;
          appendLog(`person created: ${currentPersonId} (${data.display_name})`);
          document.getElementById("registration-section").style.display = "none";
          document.getElementById("enrollment-section").style.display = "block";
          showNotice(`Person registered (${data.display_name}). Starting guided biometric enrollment...`, "info");
          
          const enrollOk = await startEnrollment(token, currentPersonId);
          if (!enrollOk) {
            appendLog(`rolling back newly created person: ${currentPersonId}`);
            await fetch(`/persons/${currentPersonId}`, { method: "DELETE", headers: { "Authorization": "Bearer " + token } });
            currentPersonId = null;
            document.getElementById("enrollment-section").style.display = "none";
            document.getElementById("registration-section").style.display = "block";
            registerBtn.disabled = false;
          }
        } else {
          const err = await resp.json().catch(() => ({}));
          const msg = err.detail || err.message || `HTTP ${resp.status}`;
          showNotice(`Person registration failed: ${msg}`, "error");
          appendLog(`person registration failed: HTTP ${resp.status}`);
          registerBtn.disabled = false;
        }
      } catch (err) {
        showNotice(`Network error during registration: ${err.message}`, "error");
        appendLog(`network error during registration: ${err.message}`);
        registerBtn.disabled = false;
      }
    }

    async function startEnrollment(token, personId) {
      try {
        const consentResp = await fetch(`/persons/${personId}`, {
          method: "PATCH",
          headers: {
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ consent_status: "granted" })
        });

        if (!consentResp.ok) {
          const cErr = await consentResp.json().catch(() => ({}));
          const cMsg = cErr.detail || cErr.message || `HTTP ${consentResp.status}`;
          showNotice(`Failed to set biometric consent: ${cMsg}.`, "error");
          appendLog(`consent update failed: HTTP ${consentResp.status}`);
          return false;
        }

        const resp = await fetch(`/persons/${personId}/enrollment/sessions`, {
          method: "POST",
          headers: {
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ mode: "add_samples" })
        });

        if (resp.status === 201) {
          appendLog(`enrollment session started for: ${personId}`);
          isEnrollmentCommitted = false;
          pollEnrollmentStep(token, personId);
          return true;
        } else {
          const err = await resp.json().catch(() => ({}));
          const msg = err.detail || err.message || `HTTP ${resp.status}`;
          showNotice(`Enrollment session start failed: ${msg}.`, "error");
          appendLog(`enrollment session start failed: HTTP ${resp.status}`);
          return false;
        }
      } catch (err) {
        showNotice(`Network error starting enrollment: ${err.message}.`, "error");
        appendLog(`network error starting enrollment: ${err.message}`);
        return false;
      }
    }

    function updateEnrollmentProgressUI(data) {
      const dir = (data.current_direction || "FRONT").toUpperCase();
      const instr = DIRECTION_INSTRUCTIONS[dir] || `Look ${dir}`;
      document.getElementById("current-direction").textContent = dir;
      document.getElementById("current-instruction").textContent = instr;

      const counts = data.direction_counts || {};
      const targets = data.direction_targets || {};
      const getVal = (obj, key) => {
        if (!obj) return undefined;
        if (obj[key] !== undefined) return obj[key];
        if (obj[key.toLowerCase()] !== undefined) return obj[key.toLowerCase()];
        if (obj[key.toUpperCase()] !== undefined) return obj[key.toUpperCase()];
        return undefined;
      };
      const parts = ["FRONT", "RIGHT", "LEFT"].map(d => {
        const c = getVal(counts, d) ?? 0;
        const t = getVal(targets, d) ?? 10;
        return `${d} ${c}/${t}`;
      });
      document.getElementById("direction-progress").textContent = parts.join(" | ");

      const total = data.samples_collected !== undefined ? data.samples_collected : (data.total_samples !== undefined ? data.total_samples : 0);
      const target = data.target_samples !== undefined ? data.target_samples : 30;
      document.getElementById("total-progress").textContent = `${total}/${target}`;
    }

    function pollEnrollmentStep(token, personId) {
      if (enrollmentIntervalId) {
        clearInterval(enrollmentIntervalId);
      }

      enrollmentIntervalId = setInterval(async () => {
        if (isEnrollmentCommitted || isStepInProgress) return;
        isStepInProgress = true;

        try {
          const resp = await fetch(`/persons/${personId}/enrollment/sessions/step`, {
            method: "POST",
            headers: {
              "Authorization": "Bearer " + token,
              "Content-Type": "application/json"
            }
          });

          if (resp.status === 200) {
            const data = await resp.json();
            updateEnrollmentProgressUI(data);

            const isComplete = Boolean(data.is_complete || (data.state && data.state.toUpperCase() === "COMPLETED"));
            if (isComplete && !isEnrollmentCommitted) {
              isEnrollmentCommitted = true;
              clearInterval(enrollmentIntervalId);
              enrollmentIntervalId = null;
              await commitEnrollment(token, personId);
            }
          } else {
            const err = await resp.json().catch(() => ({}));
            appendLog(`enrollment step error: HTTP ${resp.status} (${err.detail || err.message || ""})`);
          }
        } catch (err) {
          appendLog(`enrollment step network error: ${err.message}`);
        } finally {
          isStepInProgress = false;
        }
      }, 500);
    }

    async function commitEnrollment(token, personId) {
      try {
        const resp = await fetch(`/persons/${personId}/enrollment/sessions/commit`, {
          method: "POST",
          headers: {
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json"
          }
        });

        if (resp.status === 200) {
          const data = await resp.json();
          const persistedCount = data.samples_enrolled !== undefined ? data.samples_enrolled : (data.samples_persisted || 30);
          appendLog(`enrollment committed: ${persistedCount} embeddings saved`);
          showNotice(`Registration complete. ${persistedCount} embeddings saved.`, "info");
          document.getElementById("current-instruction").textContent = "Enrollment Complete";
          document.getElementById("current-instruction").style.color = "#15803d";
          const target = document.getElementById("total-progress").textContent.split("/")[1] || "30";
          document.getElementById("total-progress").textContent = `${persistedCount}/${target}`;
          document.getElementById("auth-action-section").style.display = "block";
        } else {
          const err = await resp.json().catch(() => ({}));
          const msg = err.detail || err.message || `HTTP ${resp.status}`;
          showNotice(`Enrollment commit failed: ${msg}`, "error");
          appendLog(`enrollment commit failed: HTTP ${resp.status}`);
        }
      } catch (err) {
        showNotice(`Network error committing enrollment: ${err.message}`, "error");
        appendLog(`network error committing enrollment: ${err.message}`);
      }
    }

    async function cancelActiveSession(token) {
      try {
        const resp = await fetch("/authorization/sessions/current", {
          method: "DELETE",
          headers: {
            "Authorization": "Bearer " + token
          }
        });

        let shouldClear = false;
        if (resp.status === 200) {
          appendLog("cancelled previous active authorization session");
          shouldClear = true;
        } else if (resp.status === 409) {
          const err = await resp.json().catch(() => ({}));
          if (err.detail && err.detail.includes("No active")) {
            shouldClear = true;
          } else {
            appendLog(`failed to cancel session (conflict): ${err.detail}`);
            return false;
          }
        } else {
          appendLog(`failed to cancel session: HTTP ${resp.status}`);
          return false;
        }

        if (shouldClear) {
          if (abortController) {
            abortController.abort();
            abortController = null;
          }
          isConnecting = false;
          setSseIndicator("error", "Disconnected");
          clearContextUI();

          const statusEl = document.getElementById("auth-status");
          if (statusEl) {
            statusEl.textContent = "IDLE";
            statusEl.className = "badge badge-none";
          }
          ["auth-person", "auth-id", "auth-started", "auth-updated", "auth-ended"].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.textContent = "-";
          });
        }
        return true;
      } catch(err) {
        appendLog(`network error cancelling session: ${err.message}`);
        return false;
      }
    }

    async function cancelActiveEnrollmentSession(token) {
      try {
        const resp = await fetch("/persons/enrollment/sessions/current", {
          method: "DELETE",
          headers: {
            "Authorization": "Bearer " + token
          }
        });
        if (resp.status === 200) {
          appendLog("cancelled previous active enrollment session");
          return true;
        } else if (resp.status === 409) {
          const err = await resp.json().catch(() => ({}));
          if (err.detail && err.detail.includes("No active enrollment session")) {
            return true;
          } else {
            appendLog(`failed to cancel enrollment session (conflict): ${err.detail}`);
            return false;
          }
        }
        return false;
      } catch(err) {
        appendLog(`network error cancelling enrollment session: ${err.message}`);
        return false;
      }
    }

    async function showRegistrationForm() {
      clearNotice();
      const token = document.getElementById("token-input").value.trim();
      if (!token) {
        showNotice("Please enter an operator Bearer token with required scopes to register.", "error");
        return;
      }

      const regBtn = document.getElementById('show-register-btn');
      regBtn.disabled = true;

      const okAuth = await cancelActiveSession(token);
      const okEnroll = await cancelActiveEnrollmentSession(token);
      if (!okAuth || !okEnroll) {
        showNotice("Failed to cancel existing sessions. Registration is blocked.", "error");
        regBtn.disabled = false;
        return;
      }

      // Start Preflight Authorization Check
      isPreflightMode = true;
      appendLog("starting biometric preflight to check for duplicate registration...");

      try {
        const resp = await fetch("/authorization/sessions", {
          method: "POST",
          headers: {
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json"
          }
        });

        if (resp.status === 201) {
          const data = await resp.json();
          currentAuthId = data.authorization_id;
          updateSessionUI(data);
          connectSSE(token);
          showNotice("Looking for face to verify if you are already registered...", "info");
        } else {
          showNotice("Failed to start preflight check.", "error");
          regBtn.disabled = false;
        }
      } catch (err) {
        showNotice(`Network error during preflight: ${err.message}`, "error");
        regBtn.disabled = false;
      }
    }
    async function startAuthorization() {
      clearNotice();
      isPreflightMode = false;
      const token = document.getElementById("token-input").value.trim();
      const startBtn = document.getElementById("start-btn");

      if (!token) {
        showNotice("Please enter an operator Bearer token with required scopes.", "error");
        return;
      }

      startBtn.disabled = true;

      // Cancel ANY existing sessions safely
      const okAuth = await cancelActiveSession(token);
      const okEnroll = await cancelActiveEnrollmentSession(token);

      if (!okAuth || !okEnroll) {
        showNotice("Failed to cancel existing sessions cleanly. Cannot start authorization.", "error");
        startBtn.disabled = false;
        return;
      }

      // Hide registration if it was open
      document.getElementById('registration-section').style.display='none';
      document.getElementById('show-register-btn').style.display='inline-block';

      try {
        const resp = await fetch("/authorization/sessions", {
          method: "POST",
          headers: {
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json"
          }
        });

        if (resp.status === 201) {
          const data = await resp.json();
          currentAuthId = data.authorization_id;
          appendLog(`authorization session started: ${data.authorization_id}`);
          updateSessionUI(data);
          connectSSE(token);
        } else if (resp.status === 401 || resp.status === 403) {
          const err = await resp.json().catch(() => ({}));
          showNotice(`Authentication failed (${resp.status}): ${err.message || err.detail || 'Unauthorized'}`, "error");
          appendLog(`authorization request failed: HTTP ${resp.status}`);
        } else if (resp.status === 503) {
          showNotice("Authorization service unavailable (runtime not running)", "error");
          appendLog("authorization request failed: HTTP 503 service unavailable");
        } else {
          showNotice(`Unexpected error: HTTP ${resp.status}`, "error");
          appendLog(`authorization request failed: HTTP ${resp.status}`);
        }
      } catch (err) {
        showNotice(`Network error: ${err.message}`, "error");
        appendLog(`network error: ${err.message}`);
      } finally {
        startBtn.disabled = false;
      }
    }
    async function connectSSE(token) {
      if (isConnecting || abortController) {
        return;
      }

      isConnecting = true;
      setSseIndicator("connecting", "Connecting...");
      appendLog("connecting to /interaction/events SSE stream...");

      const currentAbortController = new AbortController();
      abortController = currentAbortController;

      try {
        const resp = await fetch("/interaction/events", {
          headers: {
            "Authorization": "Bearer " + token
          },
          signal: currentAbortController.signal
        });

        if (!resp.ok) {
          setSseIndicator("error", `HTTP ${resp.status}`);
          appendLog(`SSE stream connection failed: HTTP ${resp.status}`);
          isConnecting = false;
          abortController = null;
          return;
        }

        setSseIndicator("connected", "Connected");
        appendLog("SSE stream connected.");
        isConnecting = false;

        const reader = resp.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split("\n\n");
          buffer = parts.pop();

          for (const block of parts) {
            await handleSseBlock(block);
          }
        }
      } catch (err) {
        if (err.name !== "AbortError") {
          setSseIndicator("error", "Disconnected");
          appendLog(`SSE stream error: ${err.message}`);
        }
      } finally {
        if (abortController === currentAbortController) {
          isConnecting = false;
          abortController = null;
          setSseIndicator("error", "Disconnected");
        }
      }
    }

    async function handleSseBlock(raw) {
      if (!raw.trim()) return;

      let eventType = "message";
      let dataStr = "";

      const lines = raw.split("\n");
      for (const line of lines) {
        if (line.startsWith("event:")) {
          eventType = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataStr = line.slice(5).trim();
        }
      }

      if (!dataStr) return;

      try {
        const payload = JSON.parse(dataStr);
        if (eventType === "authorization_updated" || payload.event_type === "authorization_updated") {
          const who = payload.display_name || payload.person_id || "unidentified";
          appendLog(`authorization ${payload.status} (${who})`);
          updateSessionUI(payload);
          if (payload.authorization_id) {
            currentAuthId = payload.authorization_id;
          }
          const st = (payload.status || "").toUpperCase();
          if (isPreflightMode) {
            if (st === "AUTHORIZED") {
              const token = document.getElementById("token-input").value.trim();
              await cancelActiveSession(token);
              showNotice(`This person is already registered as ${who}.`, "error");
              document.getElementById('registration-section').style.display='none';
              document.getElementById('show-register-btn').disabled = false;
            } else if (st === "UNKNOWN") {
              const token = document.getElementById("token-input").value.trim();
              await cancelActiveSession(token);
              clearNotice();
              isPreflightMode = false;
              document.getElementById('registration-section').style.display='block';
              document.getElementById('show-register-btn').style.display='none';
              document.getElementById('show-register-btn').disabled = false;
            } else if (st === "UNCERTAIN") {
              showNotice("Identity could not be confirmed. Please adjust lighting and face the camera directly.", "error");
            } else if (st === "NO_FACE" || st === "LOST") {
              showNotice("Please face the camera so we can verify if you are already registered.", "info");
            } else if (st === "TIMEOUT") {
              showNotice("Preflight check timed out. Please try again.", "error");
              isPreflightMode = false;
              document.getElementById('show-register-btn').disabled = false;
            }
          } else {
            if (st === "AUTHORIZED") {
              const token = document.getElementById("token-input").value.trim();
              fetchAuthorizedContext(token, payload.authorization_id || currentAuthId);
            } else if (st === "UNKNOWN") {
              clearContextUI();
              showNotice("Person not registered. Please register.", "error");
            } else if (st === "UNCERTAIN") {
              clearContextUI();
              showNotice("Identity could not be confirmed.", "error");
            } else {
              clearContextUI();
            }
          }
        } else if (eventType === "context") {
          appendLog("received context snapshot from edge runtime");
        } else {
          appendLog(`event: ${eventType}`);
        }
      } catch (e) {
        // Non-JSON SSE frame
      }
    }

    async function fetchAuthorizedContext(token, authorizationId) {
      if (!token || !authorizationId) return;

      try {
        const resp = await fetch(`/authorization/sessions/${authorizationId}/context`, {
          headers: {
            "Authorization": "Bearer " + token
          }
        });

        if (resp.status === 200) {
          const data = await resp.json();
          appendLog(`retrieved authorized context for: ${data.display_name || data.person_id}`);
          displayContextUI(data);
        } else {
          appendLog(`authorized context retrieval HTTP ${resp.status}`);
          clearContextUI();
        }
      } catch (err) {
        appendLog(`error fetching authorized context: ${err.message}`);
        clearContextUI();
      }
    }

    function clearContextUI() {
      const panel = document.getElementById("context-panel");
      if (panel) {
        panel.style.display = "none";
      }
    }

    function displayContextUI(ctx) {
      const panel = document.getElementById("context-panel");
      if (!panel) return;
      panel.style.display = "block";

      const nameEl = document.getElementById("ctx-name");
      nameEl.textContent = ctx.preferred_name
        ? `${ctx.display_name} (${ctx.preferred_name})`
        : (ctx.display_name || "-");

      document.getElementById("ctx-role").textContent = ctx.role || "-";
      document.getElementById("ctx-department").textContent = ctx.department || "-";
      document.getElementById("ctx-room").textContent = ctx.office_room || "-";
      document.getElementById("ctx-hours").textContent = ctx.working_hours || "-";
      document.getElementById("ctx-desc").textContent = ctx.profile_description || "-";

      const qaContainer = document.getElementById("ctx-qa");
      qaContainer.innerHTML = "";
      if (Array.isArray(ctx.qa_pairs) && ctx.qa_pairs.length > 0) {
        for (const pair of ctx.qa_pairs) {
          const d = document.createElement("div");
          d.style.marginBottom = "0.4rem";
          d.style.padding = "0.25rem 0.5rem";
          d.style.background = "#ffffff";
          d.style.borderRadius = "4px";
          d.style.border = "1px solid #dcfce7";
          d.innerHTML = `<div><strong>Q:</strong> ${escapeHtml(pair.question)}</div><div><strong>A:</strong> ${escapeHtml(pair.answer)}</div>`;
          qaContainer.appendChild(d);
        }
      } else {
        qaContainer.textContent = "None";
      }
    }
  
