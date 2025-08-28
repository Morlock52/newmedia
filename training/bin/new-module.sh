#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 \"Module NN - Title\"" >&2
  exit 1
fi

MODULE_NAME="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MOD_DIR="${ROOT_DIR}/training/modules/${MODULE_NAME}"

mkdir -p "${MOD_DIR}/handouts"

# Slides HTML
cat >"${MOD_DIR}/slides.html" <<'HTML'
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Slides</title>
    <link rel="stylesheet" href="../../templates/slides.css" />
  </head>
  <body>
    <textarea id="source" style="display:none"></textarea>
    <script src="https://remarkjs.com/downloads/remark-latest.min.js"></script>
    <script>
      (async function () {
        try {
          const res = await fetch('slides.md');
          const md = await res.text();
          document.getElementById('source').value = md;
          window.slideshow = remark.create({ highlightStyle: 'monokai', ratio: '16:9' });
        } catch (e) {
          document.body.innerHTML = '<pre style="padding:24px">Could not load slides.md.\nUse training/bin/serve-training.sh</pre>';
        }
      })();
    </script>
  </body>
  </html>
HTML

# Slides MD
cat >"${MOD_DIR}/slides.md" <<MD
class: title-slide, center, middle

# ${MODULE_NAME}
## Course Title

---

## Objectives

- Objective 1
- Objective 2
- Objective 3

---

## Agenda

- Topic A
- Topic B
- Activity

---

## Content

Your content here.

MD

# Instructor guide
cat >"${MOD_DIR}/instructor-guide.md" <<'MD'
<link rel="stylesheet" href="../../templates/styles.css">

# Instructor Guide

Add timing, talking points, demo steps, and risks/mitigations.
MD

# Workbook
cat >"${MOD_DIR}/workbook.md" <<'MD'
<link rel="stylesheet" href="../../templates/styles.css">

# Learner Workbook

Exercises and reflection prompts for this module.
MD

# Assessment
cat >"${MOD_DIR}/assessment.md" <<'MD'
<link rel="stylesheet" href="../../templates/styles.css">

# Assessment

Short knowledge check and a practical task.
MD

# Quick reference handout
cat >"${MOD_DIR}/handouts/quick-reference.md" <<'MD'
<link rel="stylesheet" href="../../../templates/styles.css">

# Quick Reference

Key steps, commands, and troubleshooting tips.
MD

echo "Created module: ${MOD_DIR}"

