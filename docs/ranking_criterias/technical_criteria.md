# Technical Ranking Criteria (Cyber Range Course)

Used by the **Technical** reviewer in the ranking agent. Run **first**; threshold (e.g. 9.0 or 9.5) determines pass or re-assess/repair.

## Dimensions (from objective_ranking.md and ABSEval / correctness-focused evaluation)

1. **Correctness** — Does the solution actually solve the challenge? Are commands and code accurate? Any edge cases or false assumptions?
2. **Completeness** — Are all steps needed to reproduce the solution present? No implicit "and then you get the flag" without showing how.
3. **Technical accuracy** — Are vulnerability explanations and exploit steps technically sound? Is the solver script consistent with the course and environment?
4. **Code quality** — Is the solution script readable, commented where helpful, and following reasonable practices?
5. **Logical / process validity** — Do steps follow in a feasible order? (ABSEval-style: no "Open the can" before "Pick up the can opener".)

## Scoring (1–10)

- **9–10**: Pass threshold. Correct, complete, technically accurate; script matches course; no logical gaps.
- **7–8**: Adequate but below pass. Minor gaps or assumptions; may need repair before acceptance.
- **4–6**: Needs work. Notable errors, missing steps, or script inconsistent with course.
- **1–3**: Poor. Wrong approach, major errors, or unreproducible.

## Threshold and repair

- **Technical threshold**: e.g. 9.0 or 9.5 (configurable). If score &lt; threshold → re-assess technical focus or trigger technical repair (regenerate/repair course with technical feedback).
- Repair path: either re-run technical review with focused prompt on weak dimensions, or send feedback to content generation for targeted repair.
