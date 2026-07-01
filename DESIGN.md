# AgenticKapruka — Design System

Kapruka Concierge UI: app-shell chat workspace with purple sidebar, gold CTAs, and calm commerce surfaces.

## Classifier

**APP UI** (task-focused concierge workspace). Not a marketing landing page.

## Typography

- **Font:** Inter (400 body, 500–600 headings, 600–700 labels)
- **Scale:** `headline-xl` 40/48, `headline-lg` 32/40, `headline-md` 24/32, `body-lg` 18/28, `body-md` 16/24, `body-sm` 14/20, `label-md` 14/16, `label-sm` 12/14
- **Icons:** Material Symbols Outlined (Google Fonts)

## Color tokens (CSS / Tailwind)

| Token | Hex | Use |
|-------|-----|-----|
| `primary` | `#2c125c` | Brand purple, headings, send button |
| `primary-container` | `#422b73` | Sidebar active state |
| `sidebar-bg` | `#422B73` | Fixed nav rail |
| `secondary-container` | `#fcb812` | Primary CTA (New Session, checkout) |
| `canvas` | `#FFFFFF` | Main workspace background |
| `surface` | `#f8f9fa` | User bubbles, order summary |
| `surface-muted` | `#F0EEFA` | Chips, product image fallback |
| `on-surface` | `#191c1d` | Body text |
| `on-surface-variant` | `#494550` | Secondary text |
| `text-main` | `#333333` | Assistant message body |
| `outline-variant` | `#cbc4d1` | Borders |

Legacy aliases (`kapruka-*`, `commerce-*`) map to these tokens for checkout partials.

## Layout

- **Sidebar:** 280px fixed left, full viewport height
- **Main:** `ml-[280px]` on `md+`, hamburger overlay on mobile
- **Chat column:** `max-w-4xl` centered, `pb-[140px]` for fixed composer
- **Composer:** Fixed bottom, blurred white bar, 44px min touch targets

## Components

- **User bubble:** `bg-surface`, `rounded-xl rounded-tr-none`, border `outline-variant/30`
- **Assistant bubble:** avatar + `bg-canvas` border `surface-muted`, `rounded-xl rounded-tl-none`
- **Product grid:** `grid-cols-1 md:grid-cols-2` in assistant messages (not decorative card mosaic)
- **Suggestion chips:** `rounded-full bg-surface-muted text-primary`
- **Order summary:** `bg-surface` inset card with gold checkout CTA

## Interaction states

| Feature | Loading | Empty | Error |
|---------|---------|-------|-------|
| Chat | Spinner above composer | Welcome + suggestion chips | Red inline alert bubble |
| Products | Skeleton in card | Hidden (no carousel) | N/A |
| Cart drawer | HTMX swap | "Cart is empty" | N/A |

## Accessibility

- `role="log"` on message list, `aria-live="polite"`
- Visible labels on composer (`sr-only` ok with placeholder backup)
- 44px minimum touch targets on send, nav items, product actions
- `focus-visible` rings on interactive elements
- `prefers-reduced-motion`: disable carousel scroll animation

## NOT in scope (deferred)

- Dark mode polish (tokens exist, not fully audited)
- Settings / Recent Sessions routes (nav placeholders)
- User profile persistence (sidebar shows static placeholder)
