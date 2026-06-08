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

  function toggleRequestState(form, active) {
    const indicator = document.getElementById("chat-loading");
    if (active) {
      form.classList.add("htmx-request");
      indicator?.classList.add("htmx-request");
    } else {
      form.classList.remove("htmx-request");
      indicator?.classList.remove("htmx-request");
    }
  }

  async function streamChatPost(form) {
    const listener = findSseListener(form);
    if (!listener) {
      throw new Error("Missing chat SSE listener element");
    }

    const acceptedEvents = (listener.getAttribute("sse-swap") || "message")
      .split(",")
      .map((name) => name.trim())
      .filter(Boolean);
    const connectPath = form.dataset.chatStreamPath || CHAT_STREAM_PATH;
    const formData = new FormData(form);

    toggleRequestState(form, true);

    try {
      const response = await fetch(connectPath, {
        method: "POST",
        body: formData,
        headers: { "HX-Request": "true" },
        credentials: "same-origin",
      });

      if (!response.ok) {
        throw new Error(`Chat stream failed (${response.status})`);
      }

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
          swapListenerHtml(listener, event.data);
        }
      }

      if (buffer.trim()) {
        const parsed = parseSseChunk(`${buffer}\n\n`);
        for (const event of parsed.events) {
          if (acceptedEvents.includes(event.eventName)) {
            swapListenerHtml(listener, event.data);
          }
        }
      }

      document.body.dispatchEvent(
        new CustomEvent("htmx:afterRequest", {
          detail: { elt: form, successful: true },
        }),
      );
      form.reset();
    } catch (error) {
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

  document.addEventListener(
    "submit",
    (event) => {
      const form = event.target instanceof HTMLFormElement ? event.target : null;
      if (!form || form.id !== CHAT_FORM_ID) {
        return;
      }

      event.preventDefault();
      void streamChatPost(form).catch((error) => {
        console.error("chat SSE stream failed", error);
      });
    },
    true,
  );
})();
