/**
 * Alpine.js cartDrawer — slide-over cart panel with badge synced from HTMX cart swaps.
 */
document.addEventListener("alpine:init", () => {
  Alpine.data("cartDrawer", (initialCount = 0) => ({
    open: false,
    itemCount: Number(initialCount) || 0,

    init() {
      document.body.addEventListener("htmx:afterSwap", (event) => {
        this.syncCountFromPanel(event);
      });
    },

    openDrawer() {
      this.open = true;
    },

    close() {
      this.open = false;
    },

    proceedToCheckout() {
      const form = document.getElementById("chat-form");
      const input = document.getElementById("chat-message");
      if (!form || !input) {
        return;
      }
      input.value = "Proceed to checkout";
      form.requestSubmit();
      this.close();
    },

    syncCountFromPanel(event) {
      const target = event.detail?.target;
      if (!target || target.id !== "cart-panel") {
        return;
      }
      const raw = target.getAttribute("data-item-count") ?? "0";
      const count = parseInt(raw, 10);
      this.itemCount = Number.isNaN(count) ? 0 : count;
    },
  }));
});
