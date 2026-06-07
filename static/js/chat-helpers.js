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
