/**
 * Alpine.js rateLimitCountdown — seconds-until-retry timer for 429 alert banners.
 */
document.addEventListener("alpine:init", () => {
  Alpine.data("rateLimitCountdown", (retryAfterRaw) => ({
    remainingSeconds: Math.max(0, parseInt(retryAfterRaw, 10) || 60),
    dismissed: false,
    display: "00:00",
    intervalId: null,

    init() {
      this.tick();
      this.intervalId = setInterval(() => this.tick(), 1000);
    },

    tick() {
      if (this.remainingSeconds <= 0) {
        this.dismissed = true;
        this.display = "00:00";
        this.destroy();
        return;
      }

      const minutes = Math.floor(this.remainingSeconds / 60);
      const seconds = this.remainingSeconds % 60;
      this.display = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
      this.remainingSeconds -= 1;
    },

    destroy() {
      if (this.intervalId !== null) {
        clearInterval(this.intervalId);
        this.intervalId = null;
      }
    },
  }));
});
