/**
 * New Session flow — prompt to keep or clear cart, then POST /chat/new.
 */
(function () {
  const CHAT_FORM_ID = "chat-form";
  const NEW_CHAT_BUTTON_SELECTOR = '[data-testid="new-chat-button"]';
  const MODAL_ID = "new-session-modal";

  function findModal() {
    return document.getElementById(MODAL_ID);
  }

  function showModal() {
    const modal = findModal();
    if (!modal) {
      return;
    }
    modal.hidden = false;
    modal.classList.remove("pointer-events-none");
    modal.setAttribute("aria-hidden", "false");
    const keepButton = modal.querySelector("[data-new-session-keep]");
    keepButton?.focus();
  }

  function hideModal() {
    const modal = findModal();
    if (!modal) {
      return;
    }
    modal.hidden = true;
    modal.classList.add("pointer-events-none");
    modal.setAttribute("aria-hidden", "true");
  }

  function clearChatMessagesOptimistic() {
    const messages = document.getElementById("chat-messages");
    const template = document.getElementById("chat-empty-state-template");
    if (!messages || !template) {
      return;
    }
    messages.replaceChildren(template.content.cloneNode(true));
    document.body.dispatchEvent(
      new CustomEvent("htmx:afterSwap", { detail: { target: messages } }),
    );
  }

  function resetComposer() {
    const form = document.getElementById(CHAT_FORM_ID);
    if (!form) {
      return;
    }
    form.classList.remove("htmx-request");
    form.reset();

    const messageInput = form.querySelector("#chat-message");
    if (messageInput) {
      messageInput.readOnly = false;
      messageInput.value = "";
      messageInput.style.height = "44px";
    }

    const submitButton = form.querySelector('button[type="submit"]');
    if (submitButton) {
      submitButton.disabled = false;
    }

    const indicator = document.getElementById("chat-loading");
    if (indicator) {
      indicator.classList.remove("htmx-request", "chat-loading");
      indicator.hidden = true;
      indicator.setAttribute("aria-hidden", "true");
      const span =
        indicator.querySelector('[data-testid="chat-loading-text"]') ||
        indicator.querySelector("span");
      if (span) {
        span.textContent = "";
      }
      indicator.removeAttribute("aria-label");
    }
  }

  async function startNewSession(keepCart) {
    hideModal();

    if (typeof window.abortChatStream === "function") {
      window.abortChatStream("new-session");
    }

    clearChatMessagesOptimistic();
    resetComposer();

    const url = `/chat/new?keep_cart=${keepCart ? "true" : "false"}`;
    try {
      // /chat/new returns OOB fragments (chat-messages + cart-panel), not a full page.
      // Use HTMX ajax so OOB swaps apply; swapStyle none avoids replacing <body>.
      if (window.htmx && typeof window.htmx.ajax === "function") {
        await new Promise((resolve, reject) => {
          const onSettle = (event) => {
            cleanup();
            resolve(event);
          };
          const onError = (event) => {
            cleanup();
            reject(event);
          };
          const cleanup = () => {
            document.body.removeEventListener("htmx:afterSettle", onSettle);
            document.body.removeEventListener("htmx:responseError", onError);
          };
          document.body.addEventListener("htmx:afterSettle", onSettle);
          document.body.addEventListener("htmx:responseError", onError);
          window.htmx.ajax("POST", url, {
            target: "body",
            swap: "none",
            headers: { "HX-Request": "true" },
          });
        });
      } else {
        const response = await fetch(url, {
          method: "POST",
          headers: { "HX-Request": "true" },
          credentials: "same-origin",
        });
        if (!response.ok) {
          throw new Error(`New session failed (${response.status})`);
        }
        const html = await response.text();
        htmx.swap(document.body, html, { swapStyle: "none" });
      }
      document.body.dispatchEvent(
        new CustomEvent("htmx:afterSwap", { detail: { target: document.body } }),
      );
    } catch (error) {
      console.error("new session failed", error);
    }
  }

  function bindNewSessionControls() {
    const button = document.querySelector(NEW_CHAT_BUTTON_SELECTOR);
    const modal = findModal();
    if (!button || !modal || button.dataset.newSessionReady === "true") {
      return;
    }

    button.dataset.newSessionReady = "true";
    button.addEventListener("click", (event) => {
      event.preventDefault();
      showModal();
    });

    modal.querySelector("[data-new-session-keep]")?.addEventListener("click", () => {
      void startNewSession(true);
    });
    modal.querySelector("[data-new-session-clear]")?.addEventListener("click", () => {
      void startNewSession(false);
    });
    modal.querySelector("[data-new-session-cancel]")?.addEventListener("click", () => {
      hideModal();
    });
    modal.querySelector("[data-new-session-backdrop]")?.addEventListener("click", () => {
      hideModal();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindNewSessionControls);
  } else {
    bindNewSessionControls();
  }
})();
