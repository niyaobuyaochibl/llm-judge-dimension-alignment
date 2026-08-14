# Human Annotation Package

This package supports a small human sanity check for the TKDE revision.

## Design

- Annotators: 5 (annotator_01–03 already completed; annotator_04–05 pending)
- Local-only dimensions: diversity
- Local-only workload per annotator: 60 cases per dimension
- Global-stat workload per annotator: 6 cases
- Total annotator-facing cases: 330 (66 per annotator)

The recommended revision use is diversity-focused: local-only annotators judge visible list-level diversity from the displayed top-10 lists, while the global-stat condition checks whether humans can follow system-level Coverage@10 when it is explicitly supplied.

## Expansion to 5 annotators (κ robustness)

annotator_04 and annotator_05 were added so that inter-annotator agreement is
estimated from 5 independent raters rather than 3. Their forms judge exactly the
same six representative conflict pairs and the same sampled users as
annotator_01–03; only the randomized A/B display order differs per annotator, so
the results are directly comparable and mergeable. annotator_01–03 forms and
their completed responses are unchanged.

IMPORTANT: annotator_04/05 must be filled in by two genuine, independent human
raters. Do not auto-fill or copy answers from annotator_01–03. Only real
responses may be added to `completed/`.

## Files

- `forms_html/annotator_XX.html`: browser-based form with JSONL/CSV download buttons.
- `forms_md/annotator_XX.md`: Markdown backup form.
- `answer_sheets/annotator_XX.csv`: empty answer sheet if annotators prefer editing CSV.
- `completed/`: put completed `.jsonl` or `.csv` files here.
- `metadata.csv`: hidden aggregation metadata. Do not show this to annotators.
- `cases_all.jsonl`: annotator-facing cases in machine-readable form.

## Procedure

1. Give each new annotator exactly one HTML file: `forms_html/annotator_04.html`
   and `forms_html/annotator_05.html` (one file per person).
2. Ask them not to open `metadata.csv`.
3. They should choose A, B, or TIE for every case and optionally provide confidence.
4. Save their downloaded `.jsonl` or `.csv` files into `completed/`
   (i.e. `completed/annotator_04.jsonl`, `completed/annotator_05.jsonl`),
   alongside the existing `annotator_01–03.jsonl`.
5. Run (recomputes majorities, per-case Fleiss' κ, and alignment over all 5 raters):

```bash
python code/parse_human_annotation_results.py --package-dir human_annotation_package
```

## Ethics Note

This package collects only recommendation-list preferences and optional notes. It does not need annotator names. Check local institutional requirements before recruiting external participants or collecting any personal information.
