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
