"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import {
  AlertCircle,
  Bot,
  CheckCircle2,
  Globe,
  Loader2,
  Mic,
  MicOff,
  Radio,
  RefreshCw,
  Send,
  Shield,
  ShieldCheck,
  Sparkles,
  Square,
  Trash2,
  User,
  Volume2,
  VolumeX,
} from "lucide-react";
import { useAssistantAuth } from "./use-assistant-auth";
import type {
  AssistantLanguage,
  ChatMessage,
  DialogueTurnResponse,
  DialogueVoiceTurnResponse,
} from "./types";

export function AssistantPageContent() {
  const t = useTranslations("Assistant");

  // Language & Audio state
  const [language, setLanguage] = useState<AssistantLanguage>("bn");
  const [audioEnabled, setAudioEnabled] = useState(true);

  // Conversational state
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [isListeningRobot, setIsListeningRobot] = useState(false);
  const [isListeningBrowser, setIsListeningBrowser] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [turnError, setTurnError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const browserRecognitionRef = useRef<any>(null);

  // Privacy isolation callback: instantly clear personal messages when face or authorization is lost
  const handleAuthorizationLost = useCallback((_lostPersonId: string | null) => {
    setMessages((prev) => {
      const sanitized = prev.filter((msg) => !msg.isPersonal);
      if (sanitized.length < prev.length) {
        return [
          ...sanitized,
          {
            id: `system-lost-${Date.now()}`,
            role: "system",
            text: "Personal session concluded. Private conversation records have been safely cleared.",
            timestamp: Date.now() / 1000,
            language: "en",
          },
        ];
      }
      return sanitized;
    });
  }, []);

  const {
    session,
    isAuthorized,
    currentDisplayName,
    scanning,
    rescan,
    getHeaders,
  } = useAssistantAuth({
    onAuthorizationLost: handleAuthorizationLost,
  });

  // Auto-scroll chat view to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isProcessing, isListeningRobot, scrollToBottom]);

  // Handle client-side browser speech synthesis if audio enabled
  const playLocalSpeech = useCallback(
    (text: string, lang: AssistantLanguage) => {
      if (!audioEnabled || typeof window === "undefined" || !("speechSynthesis" in window)) {
        return;
      }
      try {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = lang === "bn" ? "bn-BD" : "en-US";
        utterance.rate = 1.0;
        utterance.onstart = () => setIsSpeaking(true);
        utterance.onend = () => setIsSpeaking(false);
        utterance.onerror = () => setIsSpeaking(false);
        window.speechSynthesis.speak(utterance);
      } catch {
        setIsSpeaking(false);
      }
    },
    [audioEnabled]
  );

  // Stop active speech playback / interrupt robot backend
  const interruptAudio = useCallback(async () => {
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    setIsSpeaking(false);
    setIsListeningRobot(false);
    setIsListeningBrowser(false);

    try {
      const headers = await getHeaders();
      await fetch("/api/dh360/dialogue/interrupt", {
        method: "POST",
        headers,
      });
    } catch {
      // Best-effort interrupt
    }
  }, [getHeaders]);

  // Submit a typed question to the DH360 dialogue coordinator
  const handleSendText = async (textToSend?: string) => {
    const query = (textToSend ?? input).trim();
    if (!query || isProcessing) return;

    setInput("");
    setTurnError(null);

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      text: query,
      timestamp: Date.now() / 1000,
      language,
      isPersonal: isAuthorized,
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsProcessing(true);

    try {
      const headers = await getHeaders();
      const res = await fetch("/api/dh360/dialogue/turn", {
        method: "POST",
        headers,
        body: JSON.stringify({
          text: query,
          language,
          speak_response: audioEnabled,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || errData.message || `HTTP ${res.status}`);
      }

      const data: DialogueTurnResponse = await res.json();

      const modelMessage: ChatMessage = {
        id: `model-${Date.now()}`,
        role: "model",
        text: data.text,
        timestamp: data.timestamp,
        language,
        isPersonal: isAuthorized,
      };

      setMessages((prev) => [...prev, modelMessage]);
      playLocalSpeech(data.text, language);
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : t("errorOccurred");
      setTurnError(errMsg);
      setMessages((prev) => [
        ...prev,
        {
          id: `error-${Date.now()}`,
          role: "system",
          text: `${t("errorOccurred")}: ${errMsg}`,
          timestamp: Date.now() / 1000,
          language,
          error: errMsg,
        },
      ]);
    } finally {
      setIsProcessing(false);
    }
  };

  // Trigger robot microphone listening via Soniox STT pipeline
  const handleRobotVoiceTurn = async () => {
    if (isProcessing || isListeningRobot) return;

    setTurnError(null);
    setIsListeningRobot(true);

    try {
      const headers = await getHeaders();
      const res = await fetch("/api/dh360/dialogue/turn/voice", {
        method: "POST",
        headers,
        body: JSON.stringify({
          language,
          speak_response: audioEnabled,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || errData.message || `HTTP ${res.status}`);
      }

      const data: DialogueVoiceTurnResponse = await res.json();

      if (data.user_text) {
        setMessages((prev) => [
          ...prev,
          {
            id: `user-voice-${Date.now()}`,
            role: "user",
            text: data.user_text,
            timestamp: data.timestamp - 1,
            language,
            isPersonal: isAuthorized,
            isVoice: true,
          },
          {
            id: `model-${Date.now()}`,
            role: "model",
            text: data.text,
            timestamp: data.timestamp,
            language,
            isPersonal: isAuthorized,
            isVoice: true,
          },
        ]);
        playLocalSpeech(data.text, language);
      }
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : t("errorOccurred");
      setTurnError(errMsg);
    } finally {
      setIsListeningRobot(false);
    }
  };

  // Browser speech recognition (for remote dashboard operators)
  const handleBrowserVoiceTurn = () => {
    if (typeof window === "undefined") return;
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      alert("Browser Speech Recognition is not supported on this browser. Use Robot Mic.");
      return;
    }

    if (isListeningBrowser && browserRecognitionRef.current) {
      browserRecognitionRef.current.stop();
      setIsListeningBrowser(false);
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      recognition.lang = language === "bn" ? "bn-BD" : "en-US";
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;

      recognition.onstart = () => setIsListeningBrowser(true);
      recognition.onend = () => setIsListeningBrowser(false);
      recognition.onerror = () => setIsListeningBrowser(false);

      recognition.onresult = (event: any) => {
        const transcript = event.results[0]?.[0]?.transcript;
        if (transcript) {
          void handleSendText(transcript);
        }
      };

      browserRecognitionRef.current = recognition;
      recognition.start();
    } catch (err) {
      console.warn("Speech recognition error:", err);
      setIsListeningBrowser(false);
    }
  };

  // Sample quick queries
  const sampleQueries = isAuthorized
    ? [
        language === "bn" ? "আমার প্রোফাইলের তথ্য বলো" : "Tell me my profile details",
        language === "bn" ? "আমার সময়সূচি কি?" : "What is my schedule?",
        language === "bn" ? "তুমি কে?" : "Who are you?",
        language === "bn" ? "মেলার বিস্তারিত বলো" : "Tell me about the Innovation Fair",
      ]
    : [
        language === "bn" ? "তুমি কে?" : "Who are you?",
        language === "bn" ? "বাংলাদেশ ইনোভেশন ফেয়ার ২০২৬ সম্পর্কে বলো" : "Tell me about the Bangladesh Innovation Fair 2026",
        language === "bn" ? "রোবোটেক ভ্যালি কি কাজ করে?" : "What does Robotech Valley do?",
        language === "bn" ? "মানুষের রক্তে কতভাগ পানি থাকে?" : "What percentage of blood is water?",
      ];

  // Render camera authorization pill badge
  const renderAuthBadge = () => {
    if (isAuthorized) {
      return (
        <div className="flex items-center gap-2 rounded-2xl bg-emerald-50 px-3.5 py-1.5 ring-1 ring-emerald-300">
          <ShieldCheck className="size-4 text-emerald-600" />
          <span className="text-xs font-bold text-emerald-900">
            {t("authorizedUser", { name: currentDisplayName || "User" })}
          </span>
        </div>
      );
    }

    const status = session?.status;
    let label = t("scanActive");
    let bg = "bg-sky-50 ring-sky-200 text-sky-800";

    if (status === "no_face") {
      label = t("noFace");
      bg = "bg-amber-50 ring-amber-200 text-amber-800";
    } else if (status === "unknown") {
      label = t("unknown");
      bg = "bg-rose-50 ring-rose-200 text-rose-800";
    } else if (status === "uncertain") {
      label = t("uncertain");
      bg = "bg-purple-50 ring-purple-200 text-purple-800";
    } else if (status === "lost") {
      label = t("lost");
      bg = "bg-amber-50 ring-amber-200 text-amber-800";
    } else if (status === "timeout" || status === "cancelled") {
      label = t("timeout");
      bg = "bg-slate-100 ring-slate-200 text-slate-700";
    }

    return (
      <div className={`flex items-center gap-2 rounded-2xl px-3.5 py-1.5 ring-1 text-xs font-semibold ${bg}`}>
        <Radio className="size-3.5 animate-pulse" />
        <span>{label}</span>
        {status && ["timeout", "cancelled", "lost"].includes(status) && (
          <button
            type="button"
            onClick={rescan}
            className="ml-1 flex items-center gap-1 rounded-lg bg-white px-2 py-0.5 text-[11px] font-bold text-sky-700 shadow-sm ring-1 ring-sky-300 hover:bg-sky-50"
          >
            <RefreshCw className="size-3" />
            {t("rescan")}
          </button>
        )}
      </div>
    );
  };

  return (
    <div className="mx-auto flex h-[calc(100vh-8.5rem)] max-w-5xl flex-col gap-3">
      {/* Top Banner: Mode & Authorization Status */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-3xl border border-sky-200 bg-white/90 p-4 shadow-sm backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div
            className={`flex size-11 items-center justify-center rounded-2xl shadow-sm ring-1 ${
              isAuthorized
                ? "bg-emerald-50 text-emerald-600 ring-emerald-300"
                : "bg-sky-50 text-sky-600 ring-sky-300"
            }`}
          >
            {isAuthorized ? <ShieldCheck className="size-6" /> : <Bot className="size-6" />}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-sky-950">{t("title")}</h1>
              <span
                className={`inline-flex items-center gap-1 rounded-xl px-2.5 py-0.5 text-xs font-extrabold ring-1 ${
                  isAuthorized
                    ? "bg-emerald-500 text-white ring-emerald-600"
                    : "bg-sky-600 text-white ring-sky-700"
                }`}
              >
                <Sparkles className="size-3" />
                {isAuthorized ? t("authorizedModeBadge") : t("generalModeBadge")}
              </span>
            </div>
            <p className="text-xs text-sky-800/80">
              {isAuthorized ? t("authorizedModeDesc") : t("generalModeDesc")}
            </p>
          </div>
        </div>

        {/* Live status badge & Global action controls */}
        <div className="flex flex-wrap items-center gap-2">
          {renderAuthBadge()}

          {/* Audio speech toggle */}
          <button
            type="button"
            onClick={() => setAudioEnabled(!audioEnabled)}
            className={`flex items-center gap-1.5 rounded-2xl px-3 py-1.5 text-xs font-semibold ring-1 transition-colors ${
              audioEnabled
                ? "bg-sky-50 text-sky-800 ring-sky-300 hover:bg-sky-100"
                : "bg-slate-100 text-slate-600 ring-slate-300 hover:bg-slate-200"
            }`}
            title={audioEnabled ? t("audioOn") : t("audioOff")}
          >
            {audioEnabled ? <Volume2 className="size-4 text-sky-600" /> : <VolumeX className="size-4" />}
            <span className="hidden sm:inline">{audioEnabled ? t("audioOn") : t("audioOff")}</span>
          </button>

          {/* Language selector */}
          <div className="flex items-center rounded-2xl bg-slate-100 p-0.5 ring-1 ring-slate-200">
            <button
              type="button"
              onClick={() => setLanguage("bn")}
              className={`rounded-xl px-2.5 py-1 text-xs font-bold transition-all ${
                language === "bn"
                  ? "bg-white text-sky-800 shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              বাংলা
            </button>
            <button
              type="button"
              onClick={() => setLanguage("en")}
              className={`rounded-xl px-2.5 py-1 text-xs font-bold transition-all ${
                language === "en"
                  ? "bg-white text-sky-800 shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              EN
            </button>
          </div>

          {/* Clear conversation button */}
          {messages.length > 0 && (
            <button
              type="button"
              onClick={() => setMessages([])}
              className="rounded-2xl p-2 text-slate-500 ring-1 ring-slate-200 hover:bg-red-50 hover:text-red-600 hover:ring-red-200"
              title={t("clearChat")}
            >
              <Trash2 className="size-4" />
            </button>
          )}
        </div>
      </div>

      {/* Main Conversation Stream */}
      <div className="flex flex-1 flex-col overflow-hidden rounded-3xl border border-sky-200 bg-white/95 p-4 shadow-md backdrop-blur-sm sm:p-6">
        <div className="flex-1 space-y-4 overflow-y-auto pr-2">
          {/* Welcome state if no messages */}
          {messages.length === 0 && (
            <div className="mx-auto my-auto flex max-w-lg flex-col items-center justify-center space-y-4 py-8 text-center">
              <div
                className={`flex size-16 items-center justify-center rounded-3xl shadow-lg ring-1 ${
                  isAuthorized
                    ? "bg-emerald-100 text-emerald-700 ring-emerald-300"
                    : "bg-sky-100 text-sky-700 ring-sky-300"
                }`}
              >
                {isAuthorized ? <ShieldCheck className="size-9" /> : <Bot className="size-9" />}
              </div>
              <div>
                <h2 className="text-lg font-bold text-sky-950">
                  {isAuthorized
                    ? t("welcomeAuthorized", { name: currentDisplayName || "User" })
                    : t("welcomeTitle")}
                </h2>
                <p className="mt-1.5 text-sm text-sky-800/80 leading-relaxed">
                  {isAuthorized
                    ? t("welcomeAuthorized", { name: currentDisplayName || "User" })
                    : t("welcomeGeneral")}
                </p>
              </div>

              {/* Sample Prompt Pills */}
              <div className="mt-2 flex flex-wrap justify-center gap-2">
                {sampleQueries.map((q, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => void handleSendText(q)}
                    className="rounded-2xl bg-sky-50 px-3.5 py-1.5 text-xs font-semibold text-sky-800 ring-1 ring-sky-200 hover:bg-sky-100 hover:ring-sky-300 transition-all text-left"
                  >
                    💬 {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Render Dialogue Messages */}
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex items-start gap-3 ${
                msg.role === "user" ? "flex-row-reverse" : "flex-row"
              }`}
            >
              {/* Avatar */}
              <div
                className={`flex size-9 shrink-0 items-center justify-center rounded-2xl shadow-sm text-white ${
                  msg.role === "user"
                    ? "bg-sky-600"
                    : msg.role === "system"
                    ? "bg-slate-500"
                    : "bg-gradient-to-tr from-sky-500 to-indigo-600"
                }`}
              >
                {msg.role === "user" ? (
                  <User className="size-4" />
                ) : msg.role === "system" ? (
                  <Shield className="size-4" />
                ) : (
                  <Bot className="size-4" />
                )}
              </div>

              {/* Bubble */}
              <div
                className={`max-w-[80%] rounded-3xl px-4 py-3 text-sm leading-relaxed shadow-sm ${
                  msg.role === "user"
                    ? "bg-sky-600 text-white rounded-tr-none"
                    : msg.role === "system"
                    ? "bg-slate-100 text-slate-700 ring-1 ring-slate-200 text-xs"
                    : "bg-slate-50/90 text-slate-800 ring-1 ring-sky-100 rounded-tl-none font-medium"
                }`}
              >
                {/* Personal context & voice badges */}
                <div className="mb-1 flex flex-wrap items-center gap-1.5">
                  {msg.isPersonal && (
                    <span className="inline-flex items-center gap-1 rounded-lg bg-emerald-100 px-2 py-0.5 text-[10px] font-bold text-emerald-800">
                      <ShieldCheck className="size-2.5" />
                      {t("personalDataBadge")}
                    </span>
                  )}
                  {msg.isVoice && (
                    <span className="inline-flex items-center gap-1 rounded-lg bg-sky-100 px-2 py-0.5 text-[10px] font-bold text-sky-800">
                      <Mic className="size-2.5" />
                      {t("voiceTurnBadge")}
                    </span>
                  )}
                </div>

                <p className="whitespace-pre-wrap">{msg.text}</p>
              </div>
            </div>
          ))}

          {/* Active processing / listening indicator */}
          {(isProcessing || isListeningRobot || isListeningBrowser) && (
            <div className="flex items-center gap-3">
              <div className="flex size-9 items-center justify-center rounded-2xl bg-gradient-to-tr from-sky-500 to-indigo-600 text-white shadow-sm">
                <Bot className="size-4" />
              </div>
              <div className="flex items-center gap-2 rounded-2xl bg-sky-50 px-4 py-2.5 text-xs font-semibold text-sky-800 ring-1 ring-sky-200">
                <Loader2 className="size-4 animate-spin text-sky-600" />
                <span>
                  {isListeningRobot
                    ? t("listening") + " (Soniox)"
                    : isListeningBrowser
                    ? t("listening") + " (Browser Mic)"
                    : t("processing")}
                </span>
                <button
                  type="button"
                  onClick={interruptAudio}
                  className="ml-2 flex items-center gap-1 rounded-lg bg-red-600 px-2 py-0.5 text-[11px] font-bold text-white shadow-sm hover:bg-red-700"
                >
                  <Square className="size-3" />
                  {t("interrupt")}
                </button>
              </div>
            </div>
          )}

          {/* Speaking Audio indicator */}
          {isSpeaking && (
            <div className="flex items-center justify-between rounded-2xl bg-emerald-50 px-4 py-2 text-xs font-semibold text-emerald-800 ring-1 ring-emerald-200">
              <div className="flex items-center gap-2">
                <Volume2 className="size-4 animate-bounce text-emerald-600" />
                <span>{t("speaking")}</span>
              </div>
              <button
                type="button"
                onClick={interruptAudio}
                className="flex items-center gap-1 rounded-lg bg-red-600 px-2.5 py-1 text-xs font-bold text-white hover:bg-red-700"
              >
                <Square className="size-3" />
                {t("interrupt")}
              </button>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar & Controls */}
        <div className="mt-3 border-t border-sky-100 pt-3">
          {turnError && (
            <div className="mb-2 flex items-center gap-2 rounded-2xl bg-red-50 p-2.5 text-xs font-medium text-red-700 ring-1 ring-red-200">
              <AlertCircle className="size-4 shrink-0 text-red-600" />
              <span>{turnError}</span>
            </div>
          )}

          <form
            onSubmit={(e) => {
              e.preventDefault();
              void handleSendText();
            }}
            className="flex items-center gap-2"
          >
            {/* Robot Physical Microphone Button (Soniox) */}
            <button
              type="button"
              onClick={handleRobotVoiceTurn}
              disabled={isProcessing || isListeningRobot}
              className={`flex items-center gap-1.5 rounded-2xl px-3.5 py-3 text-xs font-bold shadow-sm transition-all disabled:opacity-50 ${
                isListeningRobot
                  ? "bg-red-600 text-white animate-pulse"
                  : "bg-sky-50 text-sky-800 ring-1 ring-sky-300 hover:bg-sky-100"
              }`}
              title={t("robotMicTooltip")}
            >
              <Mic className="size-4" />
              <span className="hidden md:inline">{t("robotMic")}</span>
            </button>

            {/* Browser Microphone Button */}
            <button
              type="button"
              onClick={handleBrowserVoiceTurn}
              disabled={isProcessing || isListeningRobot}
              className={`flex items-center gap-1.5 rounded-2xl px-3.5 py-3 text-xs font-bold shadow-sm transition-all disabled:opacity-50 ${
                isListeningBrowser
                  ? "bg-purple-600 text-white animate-pulse"
                  : "bg-purple-50 text-purple-800 ring-1 ring-purple-300 hover:bg-purple-100"
              }`}
              title={t("browserMicTooltip")}
            >
              <Radio className="size-4" />
              <span className="hidden md:inline">{t("browserMic")}</span>
            </button>

            {/* Query Input */}
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={t("inputPlaceholder")}
              disabled={isProcessing || isListeningRobot}
              className="flex-1 rounded-2xl border border-sky-200 bg-sky-50/50 px-4 py-3 text-sm text-sky-950 placeholder-sky-400 shadow-inner outline-none transition focus:border-sky-400 focus:bg-white focus:ring-2 focus:ring-sky-300 disabled:opacity-60"
            />

            {/* Send Button */}
            <button
              type="submit"
              disabled={!input.trim() || isProcessing}
              className="flex size-11 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-r from-sky-500 to-sky-600 text-white shadow-md shadow-sky-300/60 hover:from-sky-600 hover:to-sky-700 disabled:opacity-50"
              aria-label={t("send")}
            >
              {isProcessing ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Send className="size-4" />
              )}
            </button>
          </form>

          {/* Privacy Footnote */}
          <p className="mt-2 text-center text-[11px] text-sky-700/70">
            {t("privacyNotice")}
          </p>
        </div>
      </div>
    </div>
  );
}
