/**
 * POST /chat/stream SSE bridge for htmx-ext-sse 2.x.
 *
 * The SSE extension opens EventSource (GET) for sse-connect; our chat endpoint
 * streams over POST with form data. This script intercepts the chat form request,
 * reads the SSE body via fetch(), and swaps each `message` event into the target
 * declared on the dedicated sse-swap listener.
 */
(function () {
  const CHAT_FORM_ID = "chat-form";
  const CHAT_STREAM_PATH = "/chat/stream";
  const CHAT_STREAM_TIMEOUT_MS = 90_000;

  function chatDebugEnabled(form) {
    return form?.dataset?.chatDebug === "true";
  }

  function chatDebugLog(form, label, detail) {
    if (!chatDebugEnabled(form)) {
      return;
    }
    if (detail === undefined) {
      console.info(`[chat] ${label}`);
      return;
    }
    console.info(`[chat] ${label}`, detail);
  }

  const originalCreateEventSource = htmx.createEventSource;

  htmx.createEventSource = function createChatSafeEventSource(url) {
    if (url === CHAT_STREAM_PATH || url.endsWith(CHAT_STREAM_PATH)) {
      return {
        url,
        withCredentials: true,
        readyState: 2,
        CONNECTING: 0,
        OPEN: 1,
        CLOSED: 2,
        close() {},
        addEventListener() {},
        removeEventListener() {},
        dispatchEvent() {
          return true;
        },
        onopen: null,
        onmessage: null,
        onerror: null,
      };
    }
    return originalCreateEventSource(url);
  };

  function findChatForm() {
    return document.getElementById(CHAT_FORM_ID);
  }

  function findSseListener(form) {
    return form.querySelector("[sse-swap]");
  }

  function prepareChatForm(form) {
    const path = form.getAttribute("sse-connect") || CHAT_STREAM_PATH;
    form.dataset.chatStreamPath = path;
    // HTMX must not own this form — our fetch/SSE bridge replaces sse-connect POST.
    form.removeAttribute("hx-post");
    form.removeAttribute("hx-trigger");
    form.removeAttribute("sse-connect");
    form.removeAttribute("hx-ext");
  }

  function initChatForm() {
    const form = findChatForm();
    if (!form || form.dataset.chatSseReady === "true") {
      return;
    }
    prepareChatForm(form);
    form.dataset.chatSseReady = "true";
  }

  function parseSseChunk(buffer) {
    const events = [];
    const parts = buffer.split("\n\n");
    const remainder = parts.pop() ?? "";

    for (const part of parts) {
      if (!part.trim()) {
        continue;
      }
      let eventName = "message";
      const dataLines = [];
      for (const line of part.split("\n")) {
        if (line.startsWith("event:")) {
          eventName = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).replace(/^\s/, ""));
        }
      }
      if (dataLines.length) {
        events.push({ eventName, data: dataLines.join("\n") });
      }
    }

    return { events, remainder };
  }

  function swapListenerHtml(listener, html) {
    const targetSelector = listener.getAttribute("hx-target");
    const target = targetSelector ? document.querySelector(targetSelector) : null;
    if (!target) {
      return;
    }
    const swapStyle = listener.getAttribute("hx-swap") || "beforeend";
    const childCountBefore = target.children.length;
    htmx.swap(target, html, { swapStyle });
    // Activate hx-* on streamed fragments only (not the whole message log / form).
    for (const node of Array.from(target.children).slice(childCountBefore)) {
      htmx.process(node);
    }
    document.body.dispatchEvent(
      new CustomEvent("htmx:afterSwap", { detail: { target } }),
    );
  }

  function swapStatusHtml(html) {
    // Status events OOB-update the pending assistant bubble without appending.
    htmx.swap(document.body, html, { swapStyle: "none" });
    document.body.dispatchEvent(
      new CustomEvent("htmx:afterSwap", { detail: { target: document.body } }),
    );
  }

  function toggleRequestState(form, active) {
    const indicator = document.getElementById("chat-loading");
    const submitButton = form.querySelector('button[type="submit"]');
    const messageInput = form.querySelector("#chat-message");
    if (active) {
      form.classList.add("htmx-request");
      indicator?.classList.add("htmx-request", "chat-loading");
      if (submitButton) {
        submitButton.disabled = true;
      }
      if (messageInput) {
        messageInput.readOnly = true;
        messageInput.value = "";
      }
    } else {
      form.classList.remove("htmx-request");
      indicator?.classList.remove("htmx-request", "chat-loading");
      if (submitButton) {
        submitButton.disabled = false;
      }
      if (messageInput) {
        messageInput.readOnly = false;
      }
    }
  }

  function removePendingAssistantBubbles() {
    for (const el of document.querySelectorAll('[id^="assistant-stream-"]')) {
      el.remove();
    }
  }

  function registerAfterRequestBackup() {
    document.addEventListener("htmx:afterRequest", (event) => {
      const elt = event.detail?.elt;
      if (!elt || elt.id !== CHAT_FORM_ID) {
        return;
      }
      toggleRequestState(elt, false);
    });
  }

  async function streamChatPost(form, formData) {
    const listener = findSseListener(form);
    if (!listener) {
      throw new Error("Missing chat SSE listener element");
    }

    const acceptedEvents = (listener.getAttribute("sse-swap") || "message")
      .split(",")
      .map((name) => name.trim())
      .filter(Boolean);
    const connectPath = form.dataset.chatStreamPath || CHAT_STREAM_PATH;
    const outboundMessage = formData.get("message");
    chatDebugLog(form, "send", {
      path: connectPath,
      message: outboundMessage,
    });

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), CHAT_STREAM_TIMEOUT_MS);

      const response = await fetch(connectPath, {
        method: "POST",
        body: formData,
        headers: { "HX-Request": "true" },
        credentials: "same-origin",
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        chatDebugLog(form, "http error", { status: response.status });
        throw new Error(`Chat stream failed (${response.status})`);
      }

      chatDebugLog(form, "stream open", { status: response.status });

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("Chat stream body unavailable");
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const parsed = parseSseChunk(buffer);
        buffer = parsed.remainder;

        for (const event of parsed.events) {
          if (!acceptedEvents.includes(event.eventName)) {
            continue;
          }
          chatDebugLog(form, "sse event", {
            event: event.eventName,
            htmlChars: event.data?.length ?? 0,
          });
          if (event.eventName === "status") {
            swapStatusHtml(event.data);
          } else {
            swapListenerHtml(listener, event.data);
          }
        }
      }

      if (buffer.trim()) {
        const parsed = parseSseChunk(`${buffer}\n\n`);
        for (const event of parsed.events) {
          if (!acceptedEvents.includes(event.eventName)) {
            continue;
          }
          if (event.eventName === "status") {
            swapStatusHtml(event.data);
          } else {
            swapListenerHtml(listener, event.data);
          }
        }
      }

      chatDebugLog(form, "stream complete");
      document.body.dispatchEvent(
        new CustomEvent("htmx:afterRequest", {
          detail: { elt: form, successful: true },
        }),
      );
      form.reset();
    } catch (error) {
      removePendingAssistantBubbles();
      chatDebugLog(form, "stream failed", error);
      toggleRequestState(form, false);
      document.body.dispatchEvent(
        new CustomEvent("htmx:afterRequest", {
          detail: { elt: form, successful: false },
        }),
      );
      throw error;
    } finally {
      toggleRequestState(form, false);
    }
  }

  initChatForm();
  registerAfterRequestBackup();

  document.addEventListener(
    "submit",
    (event) => {
      const form = event.target instanceof HTMLFormElement ? event.target : null;
      if (!form || form.id !== CHAT_FORM_ID) {
        return;
      }

      event.preventDefault();
      if (form.classList.contains("htmx-request")) {
        return;
      }
      const formData = new FormData(form);
      toggleRequestState(form, true);
      void streamChatPost(form, formData).catch((error) => {
        console.error("chat SSE stream failed", error);
      });
    },
    true,
  );
})();
