# Data Governance for a Public Repository

## Public code, private data

All credentialed clinical data and patient-level derivatives must remain in institutionally approved private storage outside the Git working tree. A file being deidentified does not automatically make it permissible to publish.

## Prohibited repository content

- source clinical tables and database extracts;
- radiology reports, discharge summaries, progress notes, or note snippets;
- medical images and DICOM metadata;
- patient, admission, stay, study, or note identifiers;
- row-level features, labels, predictions, embeddings, or attention outputs;
- small “example” cohorts copied from a credentialed dataset;
- model checkpoints trained on protected or credentialed data;
- query caches, notebook outputs, debug logs, or stack traces containing records;
- credentials, API tokens, signed URLs, connection strings, or private endpoints.

## Permitted content

- source code and SQL without embedded records;
- configuration templates without secrets or local paths;
- synthetic fixtures generated without copying real records;
- aggregate metrics with disclosure review;
- aggregate plots with sufficiently large groups;
- data dictionaries and schema descriptions permitted by the source license;
- model cards and research documentation.

## Local workflow

1. Store clinical data outside the repository.
2. Pass absolute paths through local environment variables or CLI arguments.
3. Write checkpoints and predictions to an external private output directory.
4. Run `git status --short` before every commit.
5. Run `pytest` and the privacy scanner.
6. Inspect `git diff --cached` before pushing.

## Incident response

If sensitive data or a credential is committed:

1. stop pushing and sharing immediately;
2. revoke exposed credentials;
3. notify the dataset custodian or institutional privacy contact;
4. remove the material from the full Git history, not only the latest commit;
5. rotate affected secrets and document the response;
6. do not assume that deleting a public branch removes prior exposure.

## Model outputs

Serialized models can memorize or reveal properties of training data. This repository ignores model files by default. Publish only after an explicit disclosure and governance review.
