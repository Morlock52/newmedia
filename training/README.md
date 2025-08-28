# Training Materials Starter Kit

Professionally formatted, modular training materials you can customize for any topic. This kit includes slide templates, instructor and learner docs, and helper scripts for serving/exporting content.

## What’s Inside

- Structure: course-wide templates and module folders
- Slides: modern, responsive Markdown-to-slides via remark.js
- Docs: instructor guide, learner workbook, assessments, handouts
- Styling: cohesive typography, spacing, color system for a professional look
- Scripts: quick local server and module scaffolder

## Structure

```
training/
  README.md
  checklist.md
  metadata.yaml
  templates/
    styles.css           # Shared doc styles (print-friendly)
    slides.css           # Slide theme
    slides-base.html     # Base HTML to render slides.md with remark.js
  modules/
    Module 01 - Introduction/
      slides.md
      instructor-guide.md
      workbook.md
      assessment.md
      handouts/
        quick-reference.md
  bin/
    serve-training.sh    # Simple HTTP server for slides
    new-module.sh        # Scaffold a new module
```

## Quick Start

1) Serve locally (recommended for slides):

```
cd training
./bin/serve-training.sh
# open http://localhost:8080/training/modules/Module%2001%20-%20Introduction/slides.html
```

2) Edit content:

- Update `training/metadata.yaml` for title, audience, duration
- Fill in `modules/Module 01 - Introduction/*`
- Duplicate the module folder for more modules or run `./bin/new-module.sh "Module 02 - Deep Dive"`

3) Export (optional):

You can print slides to PDF from the browser (set margins to none, background graphics on). For DOC/PDF versions of guides/workbooks, use your preferred Markdown-to-PDF tool (e.g., Pandoc), or print from a Markdown-capable viewer.

## Authoring Guidance

- Objectives: Start each module with 3–5 measurable outcomes
- Agenda: Time-box sections and include breaks
- Visual rhythm: Alternate concept slides with demos and short activities
- Accessibility: Use high-contrast palettes, meaningful headings, and alt text
- Assessment: Include short checks per section; end with a summary task
- Reuse: Keep examples in `handouts/` to reuse across modules

## Branding & Customization

- Colors and fonts are set in `templates/styles.css` and `templates/slides.css`
- Swap palette variables and font stacks to match your brand
- If you have a logo, reference it in `slides-base.html` and handouts

## Tips for Facilitation

- Start with a quick baseline poll to tailor depth/speed
- Use a timer for activities; show time remaining on slides if helpful
- Encourage pair work to reduce pressure and boost engagement
- Close each module with: recap → questions → next steps → assessment

## FAQ

- Why remark.js? It’s lightweight, works from static files, and keeps authors in Markdown.
- Can we use Marp/Reveal/etc.? Yes—swap `slides-base.html` or add an alternate template.

## License

Internal use by your team; customize freely.

