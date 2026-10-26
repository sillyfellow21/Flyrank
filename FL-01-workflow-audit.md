# FL-01 Phase Setup — AI Fluency Workflow Audit

**Name/Role:** CS / Engineering university student
**Focus:** Finishing coursework/exams + internship prep (30–90 days)
**Tools:** Google Docs / Sheets / Slides, VS Code, GitHub
**Tone for Claude Project:** Direct and concise
**Frameworks:** Ethan Mollick — *On-boarding your AI Intern* (give the intern well-scoped tasks + full context, always check its work). Anthropic Academy — *AI Fluency: Framework & Foundations*. Claude Projects for reusable context.

---

## 1. Workflow Audit Table — 12 Recurring Tasks

| # | Recurring task | Frequency / time cost | Classification | One-line rationale |
|---|---|---|---|---|
| 1 | Debug errors in Python/Java coding assignments | 3x/week, 2–3h | Collaborate with AI | AI suggests fixes fast, I must understand logic to pass exams. |
| 2 | Write boilerplate code + unit tests for labs | 2x/week, 1.5h | Delegate to AI with review | AI drafts starter code quickly, I run and verify correctness. |
| 3 | Summarize lecture slides into exam study notes | 4x/week, 4h | Collaborate with AI | AI condenses slides, I add gaps and memorize key derivations. |
| 4 | Practice DSA/LeetCode for internship prep | Daily, 1h | Collaborate with AI | AI gives hint + complexity analysis, I code solution solo. |
| 5 | Draft lab reports in Google Docs | 2x/week, 3h | Delegate to AI with review | AI builds outline from my data, I insert results and check claims. |
| 6 | Build slide deck for project presentations | 1x / 2 weeks, 2h | Delegate to AI with review | AI drafts structure/bullets, I add visuals and verify technicals. |
| 7 | Track deadlines/grades in Google Sheets from syllabus | Weekly, 30m | Fully automate | Rule-based extraction + formatting, no judgment needed. |
| 8 | Tailor resume bullets + cover letters for internships | 2x/week, 2h | Collaborate with AI | AI tightens wording to job post, I ensure 100% truthful. |
| 9 | Mock technical interviews (Q&A + dry run) | 2x/week, 1.5h | Collaborate with AI | AI roleplays interviewer, I practice reasoning aloud. |
| 10 | Take proctored exams / in-class quizzes | Weekly, 2h | Just me | Must prove own knowledge; AI use is academic misconduct. |
| 11 | Final review + submit code: originality check | Per assignment, 30m | Just me | I am responsible for integrity and explaining my code. |
| 12 | Organize Drive/GitHub: rename, backup lab files | Weekly, 30m | Fully automate | Repetitive filing; template/checklist handles it. |

---

## 2. Three Target Tasks for FL-02 through FL-04

### T1 — Debug coding assignments
- **Why:** Biggest time sink (6–9h/week); ideal AI-intern task.
- **Done well:** Output = fixed code + 3-line root-cause note. Bar = passes all tests, no warnings. Accuracy = 100% tests pass, I can re-explain fix. Max time = 40 min/bug. Review = run code, check edge case, commit. Evidence = before/after screenshot + test log.

### T2 — Lecture slides to exam notes
- **Why:** 4h/week, directly boosts exam scores.
- **Done well:** Output = 1-page summary per lecture. Bar = definitions + formulas + 1 example, no hallucinations. Accuracy = 95%+ match slides, page numbers cited. Max time = 25 min/lecture. Review = check 3 random facts vs slides. Evidence = notes doc + cited slide numbers.

### T3 — Tailor resume for internship
- **Why:** High leverage for 30–90 day internship goal.
- **Done well:** Output = 1-page resume PDF + 3 tailored bullets. Bar = concise, metrics, matches keywords, zero lies. Accuracy = 0 false claims, spelling 100%. Max time = 45 min/app. Review = compare to job post, read aloud. Evidence = resume PDF + job post URL.

---
## 3. Claude Project Setup

**Project name:** `CS-Study-Internship Copilot`

**Custom instructions (paste in):**
> I am a CS/Engineering student focused on coursework/exams and internship prep. Tools: Google Docs/Sheets/Slides, VS Code, GitHub. Goals 30-90 days: keep GPA up, finish labs on time, solve DSA daily, land interviews. Tone: direct and concise, no fluff, tables and checklists. Behave like Mollick's AI intern: ask for missing context, show reasoning, cite sources, give drafts I review, quiz me for exams. Avoid: never do graded exams for me, never invent functions/facts, never add data I didn't give, never be verbose.

**Files to upload:** syllabus.pdf, rubric, lab template, resume-base.docx, 2 job posts, DSA sheet, current slides.

**Test prompt:** Using only my uploaded syllabus, list my next 2 deadlines as [assignment | due date | weight] and draft a 3-step study plan for the closer one in <100 words. Flag uncertainty.

---

## 4. Evidence Checklist

- [ ] User action required: Claude account at claude.ai, screenshot -> FL-01-claude-account.png
- [ ] User action required: ChatGPT account at chatgpt.com, screenshot -> FL-01-chatgpt-account.png
- [ ] User action required: Academy at academy.anthropic.com, enroll AI Fluency: Framework & Foundations, finish Module 1, screenshot -> FL-01-academy-module1.png
- [ ] User action required: Claude Project CS-Study-Internship Copilot with instructions above + 1 file, screenshot -> FL-01-claude-project.png
- [ ] User action required: Save this file as FL-01-workflow-audit.md with screenshots.

---

## 5. Reflection — 5 Bullets

- Helps most: debugging, condensing slides, drafting boilerplate — saves ~5h/week.
- Wastes time: vague prompts without pasting code/slides — forces rework.
- Never trust: exam answers, grade math, resume facts — hallucination + integrity risk.
- Review manually: submitted code, report numbers, resume claims — run, cite, read aloud.
- Automate first: deadline tracker in Sheets (#7) — 30 min setup, instant win for FL-02.

---

### Success Definitions

1. Debug code: fixed + tested, 100% pass, <=40min, I can explain fix.
2. Exam notes: 1-page/lecture, cited, 95% accurate, <=25min, verified vs slides.
3. Resume tailor: 1-page PDF, 0 lies/errors, <=45min, matched to job post.
