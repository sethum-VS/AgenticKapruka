/**
 * Alpine.js cartDrawer — slide-over cart panel with badge synced from HTMX cart swaps.
 */
const PROCEED_CHECKOUT_MESSAGE = "Proceed to checkout";
let proceedCheckoutInFlight = false;

function isChatFormInFlight() {
  const form = document.getElementById("chat-form");
  return Boolean(form?.classList.contains("htmx-request"));
}

function chatComposerShell() {
  const form = document.getElementById("chat-form");
  return form?.parentElement?.parentElement ?? null;
}

function updateCartDrawerComposerClearance() {
  const backdrop = document.querySelector('[data-testid="cart-backdrop"]');
  const panel = document.querySelector('[data-testid="cart-drawer-panel"]');
  const shell = chatComposerShell();
  const clearance =
    shell && shell.getBoundingClientRect
      ? Math.max(0, window.innerHeight - shell.getBoundingClientRect().top)
      : 0;
  const bottom = clearance > 0 ? `${clearance}px` : "";
  if (backdrop) {
    backdrop.style.bottom = bottom;
  }
  if (panel) {
    panel.style.bottom = bottom;
  }
}

function resetCartDrawerComposerClearance() {
  const backdrop = document.querySelector('[data-testid="cart-backdrop"]');
  const panel = document.querySelector('[data-testid="cart-drawer-panel"]');
  if (backdrop) {
    backdrop.style.bottom = "";
  }
  if (panel) {
    panel.style.bottom = "";
  }
}

function focusChatComposer() {
  const input = document.getElementById("chat-message");
  if (input instanceof HTMLElement) {
    input.focus();
  }
}

function closeCartDrawer() {
  const drawerRoot = document.querySelector('[data-testid="cart-drawer"]');
  if (!drawerRoot || !window.Alpine) {
    return;
  }
  const data = Alpine.$data(drawerRoot);
  if (data && typeof data.close === "function") {
    data.close();
  }
}

function proceedToCheckoutFromDrawer() {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-message");
  if (!form || !input) {
    return;
  }
  if (proceedCheckoutInFlight || isChatFormInFlight()) {
    return;
  }
  proceedCheckoutInFlight = true;
  input.value = PROCEED_CHECKOUT_MESSAGE;
  form.requestSubmit();
  closeCartDrawer();
  requestAnimationFrame(() => {
    focusChatComposer();
  });
}

document.addEventListener("alpine:init", () => {
  Alpine.data("cartDrawer", (initialCount = 0) => ({
    open: false,
    sidebarOpen: false,
    itemCount: Number(initialCount) || 0,

    init() {
      document.body.addEventListener("htmx:afterSwap", (event) => {
        this.syncCountFromPanel(event);
      });
      window.addEventListener("resize", () => {
        if (this.open) {
          updateCartDrawerComposerClearance();
        }
      });
    },

    openDrawer() {
      const panel = document.getElementById("cart-panel");
      if (!panel || !window.htmx) {
        this.open = true;
        this.$nextTick(() => updateCartDrawerComposerClearance());
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
        this.$nextTick(() => updateCartDrawerComposerClearance());
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
      resetCartDrawerComposerClearance();
    },

    proceedToCheckout() {
      proceedToCheckoutFromDrawer();
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

// Single delegated handler — avoids duplicate submits when multiple cartDrawer roots mount.
document.body.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof Element)) {
    return;
  }
  if (!target.closest('[data-testid="cart-proceed-checkout"]')) {
    return;
  }
  event.preventDefault();
  proceedToCheckoutFromDrawer();
});

document.body.addEventListener("htmx:afterRequest", (event) => {
  const elt = event.detail?.elt;
  if (elt?.id === "chat-form") {
    proceedCheckoutInFlight = false;
  }
});
