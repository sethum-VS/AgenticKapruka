/**
 * Alpine.js paymentCountdown — MM:SS timer until Kapruka checkout link expires.
 */
document.addEventListener("alpine:init", () => {
  Alpine.data("paymentCountdown", (expiresAtIso) => ({
    expiresAt: null,
    remainingSeconds: 0,
    expired: false,
    warning: false,
    display: "00:00",
    intervalId: null,

    init() {
      this.expiresAt = new Date(expiresAtIso);
      this.tick();
      this.intervalId = setInterval(() => this.tick(), 1000);
    },

    tick() {
      const diff = Math.max(
        0,
        Math.floor((this.expiresAt.getTime() - Date.now()) / 1000),
      );
      this.remainingSeconds = diff;
      this.expired = diff === 0;
      this.warning = diff > 0 && diff < 600;
      const minutes = Math.floor(diff / 60);
      const seconds = diff % 60;
      this.display = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
    },

    destroy() {
      if (this.intervalId !== null) {
        clearInterval(this.intervalId);
        this.intervalId = null;
      }
    },
  }));
});
