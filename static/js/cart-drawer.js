/**
 * Alpine.js cartDrawer — slide-over cart panel with badge synced from HTMX cart swaps.
 */
document.addEventListener("alpine:init", () => {
  Alpine.data("cartDrawer", (initialCount = 0) => ({
    open: false,
    sidebarOpen: false,
    itemCount: Number(initialCount) || 0,

    init() {
      document.body.addEventListener("htmx:afterSwap", (event) => {
        this.syncCountFromPanel(event);
      });
      // HTMX outerHTML swaps do not activate Alpine @click on injected cart partials.
      document.body.addEventListener("click", (event) => {
        const target = event.target;
        if (!(target instanceof Element)) {
          return;
        }
        if (!target.closest('[data-testid="cart-proceed-checkout"]')) {
          return;
        }
        this.proceedToCheckout();
      });
    },

    openDrawer() {
      const panel = document.getElementById("cart-panel");
      if (!panel || !window.htmx) {
        this.open = true;
        return;
      }

      let opened = false;
      const finish = (event) => {
        if (opened) {
          return;
        }
        opened = true;
        cleanup();
        if (event) {
          this.syncCountFromPanel(event);
        }
        this.open = true;
      };

      const onSettle = (event) => {
        if (event.detail?.target?.id !== "cart-panel") {
          return;
        }
        finish(event);
      };

      const onError = () => finish(null);

      const cleanup = () => {
        document.body.removeEventListener("htmx:afterSettle", onSettle);
        document.body.removeEventListener("htmx:responseError", onError);
      };

      document.body.addEventListener("htmx:afterSettle", onSettle);
      document.body.addEventListener("htmx:responseError", onError);
      window.htmx.ajax("GET", "/cart/panel", {
        target: "#cart-panel",
        swap: "outerHTML",
      });
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
      // outerHTML swap: detail.target is the removed node (stale count); read live panel.
      const panel = document.getElementById("cart-panel");
      const raw = panel?.getAttribute("data-item-count") ?? "0";
      const count = parseInt(raw, 10);
      this.itemCount = Number.isNaN(count) ? 0 : count;
    },
  }));
});
