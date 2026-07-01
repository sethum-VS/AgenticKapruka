/**
 * Alpine.js chatHelpers — auto-scroll message list and refocus input after HTMX swaps.
 */
document.addEventListener("alpine:init", () => {
  Alpine.data("chatHelpers", () => ({
    init() {
      document.body.addEventListener("htmx:afterSwap", (event) => {
        const target = event.detail?.target;
        if (target?.id === "chat-messages") {
          this.scrollToBottom();
        }
      });

      document.body.addEventListener("htmx:afterRequest", (event) => {
        const elt = event.detail?.elt;
        if (elt?.id === "chat-form" && event.detail.successful) {
          this.focusInput();
        }
      });

      document.body.addEventListener("click", (event) => {
        const chip = event.target.closest("[data-chat-suggestion]");
        if (!chip) {
          return;
        }
        const suggestion = chip.getAttribute("data-chat-suggestion");
        if (!suggestion) {
          return;
        }
        const form = document.getElementById("chat-form");
        const input = form?.querySelector("#chat-message");
        if (!form || !input) {
          return;
        }
        input.value = suggestion;
        form.requestSubmit();
      });

      const form = document.getElementById("chat-form");
      const input = form?.querySelector("#chat-message");
      if (form && input) {
        input.addEventListener("keydown", (event) => {
          if (event.key !== "Enter" || event.shiftKey) {
            return;
          }
          event.preventDefault();
          if (form.classList.contains("htmx-request")) {
            return;
          }
          form.requestSubmit();
        });
      }
    },

    scrollToBottom() {
      const container = this.$refs.messages;
      if (!container) {
        return;
      }
      container.scrollTop = container.scrollHeight;
    },

    focusInput() {
      const input = this.$refs.input;
      if (!input) {
        return;
      }
      input.focus();
    },
  }));
});
