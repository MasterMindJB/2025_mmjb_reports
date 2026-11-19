# Harkness Training Solutions

A private web application for turning a session recording into individual reports and one group report.

An admin signs in, uploads a recording, attaches a name and email to each detected speaker, previews the PDFs, and then presses Send. Each participant receives only their own individual report and the group report. Facilitators and actors do not receive a report. Nothing is emailed until Send is pressed.

The report text still comes from the existing prompt chain (AssemblyAI, then Gemini through LangChain, then fpdf2). Enneagram, Doshas, and AMASE sections are still in the individual report until this application is stable.

## Run it locally

Python 3.12 is the version used in CI and in the Docker image. Python 3.11 and 3.13 should also work.

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng
cp .env.example .env
```

Edit `.env` and set at least:

- `HTS_ADMIN_USERNAME`
- `HTS_ADMIN_PASSWORD` (8 characters or more; use a long password)
- `HTS_SECRET_KEY` (16 characters or more; a random string)

Then:

```shell
python -m uvicorn app.main:create_app --factory --reload
```

Open http://127.0.0.1:8000 and sign in. Transcription needs `ASSEMBLYAI_API_KEY`. Report generation needs `GOOGLE_API_KEY`. Until those are set, the upload and naming screens still work, and the failing step is shown on the session page with a retry button.

Email defaults to dry-run. Send writes one JSON file per recipient under `data/outbox/` and does not contact a mail server.

The original notebook is still in the repo (`report_generation.ipynb`). The application does not run it. Notebook-only packages are listed in `requirements-notebook.txt`.

Turn on the local safety check once per clone:

```shell
git config core.hooksPath .githooks
```

It blocks commits that look like API keys, notebooks with saved outputs, or files from `media/`, `output/`, and `data/`.

## What a session does

1. Sign in. There is one admin account, from the environment. There is no public signup.
2. Upload audio or video with the client, group, session name, date, and a required consent checkbox.
3. The recording is transcribed with speaker labels. Each letter (A, B, C, …) is shown with a few lines and timestamps.
4. Name each speaker and add an email. Merge two letters when they are the same person. Mark a facilitator or an actor so they do not get a report.
5. Generate the PDFs. Personality sections stay in the individual report.
6. Review every PDF in the browser.
7. Send. Each participant gets their own PDF plus `group.pdf`.
8. The recording and the transcript are deleted after the reports exist, within `RETENTION_DAYS` (14 by default, never more than 14). The AssemblyAI transcript is deleted through its delete API. Reports are kept.

If the app is restarted while a step is still running, open the session and use the retry button.

## Reports

Individual PDFs keep the current sections, including Enneagram, Doshas, and AMASE, plus that person's own quotes, action items, and a signature phrase. The signature phrase is that person's most frequent two-word phrase, with a short reading from `prompt_templates/personal_signature_phrase.txt`. It is never included in the group PDF.

The group PDF is the session summary: participation counts, the group narrative, and the charts. It does not include per-person quotes, action items, or personal summaries. Those used to be printed for every speaker in the group file. They now appear only in that speaker's own PDF.

The group report opens with a chord diagram drawn locally in matplotlib and set in Poppins. Each arc is one person, in a navy-to-sky palette. A ribbon leaves the person who spoke first, in that person's color, and narrows toward the person who spoke next. The three strongest pairs stay full strength. Facilitators and actors are tagged in sky caps. Names are not sent to a chart service. A small table lists the top pairs in each direction, and a numeric matrix remains as a secondary view.

## Environment

| Variable | Required | Purpose |
| --- | --- | --- |
| `HTS_ADMIN_USERNAME` | yes | Sign-in name |
| `HTS_ADMIN_PASSWORD` | yes | Sign-in password |
| `HTS_SECRET_KEY` | yes | Signs the session cookie |
| `HTS_DATA_DIR` | no | Storage directory. Default `./data` |
| `RETENTION_DAYS` | no | `0`–`14`. Default `14` |
| `ASSEMBLYAI_API_KEY` | to transcribe | AssemblyAI key |
| `GOOGLE_API_KEY` | to write reports | Gemini key. `GEMINI_API_KEY` is also read |
| `GEMINI_MODEL` | no | Default `gemini-3.1-flash-lite`. Checked when the app starts. `gemini-2.5-flash` returns 404 for new keys. |
| `GEMINI_TIMEOUT_SECONDS` | no | Per Gemini call. Default `120`, maximum `600`. The client retries at most twice. |
| `EMAIL_PROVIDER` | no | `dry-run` (default), `smtp`, or `api` |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_USE_TLS` | for smtp | SMTP delivery |
| `EMAIL_API_URL`, `EMAIL_API_KEY`, `EMAIL_FROM` | for api | JSON API shaped like Resend: `from`, `to`, `subject`, `text`, `attachments[].filename`, `attachments[].content` (base64) |
| `HTS_FEEDBACK_URL` | no | Optional link in the PDF footer |
| `HTS_MAX_UPLOAD_MB` | no | Default `500` |
| `HTS_HTTPS_ONLY` | no | Set `true` behind HTTPS so the cookie is secure |

`EMAIL_PROVIDER=smtp` or `api` sends real mail when Send is pressed. If those credentials are missing, Send stops and reports the configuration error. It does not fall back to sending through a different provider.

## Tests

```shell
python -m pytest
```

The tests do not call AssemblyAI, Gemini, or a mail server. They use a synthetic transcript and a stand-in model. One test checks that a participant's email contains only that person's PDF and the group PDF, and that the group PDF does not contain another person's individual sections.

GitHub Actions runs the same command on Python 3.12.

## Docker

```shell
docker build -t hts-reports .
docker run --env-file .env -p 8000:8000 -v hts-data:/data hts-reports
```

The container listens on port 8000 and stores files in `/data`.

## History

Older commits in this repository contain an API key and real session transcripts. Do not copy those into the app, issues, or new commits. Removing them from history is being handled separately. This application only stores new uploads under `HTS_DATA_DIR`, which git ignores.

## Limits

- One shared admin login. There are no per-client accounts.
- Speaker labels come from a single mixed recording. Groups of about ten people should be checked carefully on the naming screen, and merged when one person was split.
- The statistics are the ones the notebook already computed. Some labels, such as Idea Generation, do not measure what the name suggests. They were left in place on purpose.
- Personality scores are still produced from one transcript. They stay until the report is ready to change.
- The spider chart on an individual report draws an unlabeled average across participants in that session.
- A restarted process does not resume an in-flight transcription by itself. Use the retry button on that session.
- Dry-run is the email default until a provider is configured.
