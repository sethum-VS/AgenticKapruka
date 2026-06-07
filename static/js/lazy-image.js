/**
 * Alpine.js lazyImage — intersection-observer lazy load with skeleton and fade-in.
 */
document.addEventListener("alpine:init", () => {
  Alpine.data("lazyImage", () => ({
    src: "",
    loaded: false,
    inView: false,
    _observer: null,

    init() {
      this.src = this.$el.dataset.src || "";
      if (!this.src) {
        this.loaded = true;
        return;
      }

      this._observer = new IntersectionObserver(
        (entries) => {
          for (const entry of entries) {
            if (entry.isIntersecting) {
              this.inView = true;
              this._observer?.disconnect();
              break;
            }
          }
        },
        { rootMargin: "64px", threshold: 0.01 },
      );
      this._observer.observe(this.$el);
    },

    onLoad() {
      this.loaded = true;
    },

    destroy() {
      this._observer?.disconnect();
    },
  }));
});
