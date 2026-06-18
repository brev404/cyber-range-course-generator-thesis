# Pedagogical Ranking Criteria (Cyber Range Course)

Used by the **Pedagogical** reviewer in the ranking agent. Run **after** technical ranking passes (or in sequence). Threshold (e.g. 9.0 or 9.5) determines pass or re-assess/repair.

## Goal

Evaluate whether the course reads as **cyber range course material**: clear learning progression, human language and context, expression appropriate for training (not generic writeup tone).

## Dimensions (based on MRBench, Bloom's taxonomy, and course writeup guidelines)

1. **Sections and structure** — Abstract/TL;DR (~50 words), Objectives, Technical skills, Definitions and concepts, Reproducibility (Step 0), Thought process, Step-by-step resolution, Solution script, Conclusion, Extra resources (ATT&CK, CWE, OWASP where relevant).
2. **Cognitive load (CLT)** — Low extraneous load; clarity and chunking; "easy to follow" and "explains why."
3. **Scaffolding / reproducibility** — Step 0 present (metadata, resources); steps reproducible; code formatted and commented.
4. **Relevance / curriculum** — Where relevant, reference ATT&CK, CWE, OWASP WSTG; consistency with challenge category.
5. **Skill-level awareness** — Appropriate depth for audience (novice = full worked example; intermediate = completion-style; advanced = core vuln and pointers).
6. **Human language and context** — Does it express itself as a **course** for a cyber range (training, learning objectives, clear narrative)? Not generic "writeup" tone; appropriate context and audience.

## Scoring (1–10)

- **9–10**: Pass threshold. Clear course tone, all sections present and coherent, low cognitive load, curriculum anchors, appropriate language and context.
- **7–8**: Adequate but below pass. Good structure but language/context could be more "course-like" or some sections thin.
- **4–6**: Needs work. Missing sections, high cognitive load, or does not read as cyber range course material.
- **1–3**: Poor. Generic writeup tone, unclear objectives, or not suitable as course content.

## Threshold and repair

- **Pedagogical threshold**: e.g. 9.0 or 9.5 (configurable). If score &lt; threshold → re-assess pedagogical focus (human language, context, course expression) or trigger pedagogical repair.
- Repair path: feedback to content generation emphasizing "express as a course for cyber range training" and weak dimensions (e.g. objectives, narrative, definitions).
