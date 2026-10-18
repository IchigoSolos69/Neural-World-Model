# Neural World Model application design

## Direction and brief

This is a Windows-local research instrument for comparing physical predictions. The primary user supplies an intervention or an observed video, runs a reproducible job, and inspects the result. The dominant tone is precise and investigative. Measured evidence must remain visually separate from experimental generation.

The direction is **scientific optical bench**: a compact command center with a left inspector, a broad playback surface and frame ruler, and a job ledger. The playback surface is the memorable element. The shell is quiet and practical, with no marketing hero, generic statistic cards, gradients, decorative shadows, or implied confidence.

## Tokens before code

- Palette: bench `#edf1f5`, paper `#ffffff`, ink `#172a3b`, secondary `#516575`, rule `#c6d1db`, instrument blue `#1263a3`. Ball identities are red `#bc3e3e`, green `#237851`, blue `#285db2`; warning `#8a5314` and failure `#a32932` are reserved for actual states.
- Type: Segoe UI, Windows system UI, then sans-serif. This follows the host platform and keeps scientific controls legible. Scale 12 / 13 / 14 / 16 / 22px; numeric readings use tabular figures. Monospace is limited to job IDs.
- Spacing: compact 4 / 8 / 12 / 16 / 24 / 32px rhythm. One 286px inspector and a flexible main pane; 1080px changes to stacked panes, 600px gives full-width controls and horizontal tables.
- Radius: 4px inputs and buttons, 0px work surfaces. Elevation: none, except the native focus ring. Borders separate operating regions rather than framing repeated cards.
- Icons: custom inline line icons only where a shape adds recognition; actions keep text labels. No emoji. Status always includes words.
- Motion: only native video playback and progress; no entrance, hover, or ambient animations. Reduced motion has no functional cost.
- Accessibility: native labeled controls, visible 3px focus, 44px minimum main action target, semantic tables with horizontal overflow, text status, keyboard-operable navigation and scrubber.

## Layout

```text
Product / local connection                 worker + storage
Simulation Lab | Video Forecast | Video Backtest
Inspector (scenario / clip) | playback comparison / empty instructions
                           | synchronized timeline / forecast boundary
                           | measured table / curves / result downloads
Job ledger: time, mode, state, progress, cancel, retry, delete
```

Simulation uses supplied impulses and physical parameters. Forecast and Backtest share upload/prefix controls; only Backtest exposes measured errors. The observation/forecast boundary receives a solid vertical marker. No illustration stands in for a model result.

## Review against the brief

The subject supplies the signature interaction: synchronized physical rollouts and an editable impulse schedule. System typography is deliberate for a local Windows tool; ball colors describe identities rather than a generic accent palette. A repeated card dashboard was rejected because it hides the relation between the experiment, trajectory and measurements.

## Quality gate and remaining trade-offs

Target AI Slop Score: 1/10 (restrained conventional blue); distinctiveness: 8/10; implementation readiness: 10/10. Check keyboard focus, narrow layout, absent model, upload validation, cancelled/error jobs, and an empty ledger before delivery. Browser playback depends on installed codec support; actual unavailable artifacts must show a clear empty state. No mock data ships in the interface.
