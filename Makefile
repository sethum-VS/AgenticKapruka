TAILWIND_BIN := bin/tailwindcss
INPUT_CSS := static/css/input.css
OUTPUT_CSS := static/css/app.css

.PHONY: css css-watch install-tailwind clean-css

css: $(OUTPUT_CSS)

$(OUTPUT_CSS): $(INPUT_CSS) tailwind.config.js $(TAILWIND_BIN)
	$(TAILWIND_BIN) -i $(INPUT_CSS) -o $(OUTPUT_CSS) --minify

css-watch: $(TAILWIND_BIN)
	$(TAILWIND_BIN) -i $(INPUT_CSS) -o $(OUTPUT_CSS) --watch

install-tailwind: $(TAILWIND_BIN)

$(TAILWIND_BIN):
	@./scripts/install-tailwind.sh

clean-css:
	rm -f $(OUTPUT_CSS)
