# Customer Service Chatbot

A Streamlit customer-support chatbot powered by Gemini, LangChain, and a local FAQ knowledge base. It also supports multilingual conversations, sentiment-based escalation, tickets, document uploads, and secure policy search.
## Features

- Answers course and support questions from the FAQ dataset.
- Creates support tickets from chat history.
- Detects sentiment, duplicate payments, account-risk, and legal escalation requests.
- Supports English, Hindi, Tamil, and Kannada.
- Searches versioned policy documents with role-based access and citations.
- Accepts receipts and invoices for OCR and discrepancy checks.
## Run locally

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add GOOGLE_API_KEY to .env

streamlit run main.py
```

Open the local URL shown in the terminal, enter a question, and select **Ask**.

## Example questions

- `Can I take this course if I have never programmed?`
- `What is the refund policy?`
- `I was charged twice for order ORD-1234.`
- `I forgot my login password.`

To create a ticket, first describe the issue in chat, then select **Create Ticket from Conversation**.

## Configuration and testing

- API key: `GOOGLE_API_KEY` in `.env`
- Application settings: `config/defaults.yaml`
- Run tests: `PYTHONPATH=. pytest`

If Gemini is unavailable, the app falls back to local FAQ retrieval and deterministic sentiment handling.

For the full module specification, see [docs/MODULES.md](docs/MODULES.md).

## Project structure

```text
main.py                 # Streamlit interface
support_bot/conversation # Sessions, languages, and entity handling
support_bot/sentiment    # Sentiment analysis and escalation signals
support_bot/tickets      # Ticket creation, routing, and SLA rules
support_bot/knowledge    # Document ingestion and versioned knowledge base
support_bot/rag          # Secure retrieval, citations, and access control
support_bot/files        # Upload validation, OCR, and conflict checks
```

## Demo scenarios

1. **Duplicate payment:** `I was charged twice for order ORD-1234.`
2. **Negative sentiment:** Send several frustrated messages.
3. **High-risk escalation:** `I will take legal action for a data breach.`
4. **Historical policy:** `What was the refund policy on 15 March 2025?`
5. **Access control:** Ask for the internal escalation guide as a customer, then as an internal user.
6. **Receipt mismatch:** Upload a receipt and enter a different charged amount.

## Ticket information

For the best ticket extraction, include:

- Your name
- Order or enrollment ID
- Product or course name
- A clear description of the problem

Example: `My name is Rahul Sharma. Order ORD-1234 for Python was charged twice.`

## Notes

- The FAQ fallback works even when the Gemini service is unavailable.
- Uploaded files are stored locally; do not use real sensitive documents in demos.
- The knowledge-base panel can create and manage policy-document versions.
- Use the sidebar demo panel to switch roles, inspect escalations, and test the simulated clock.

## Troubleshooting

- Confirm `GOOGLE_API_KEY` is set if Gemini responses fail.
- Run `pip install -r requirements.txt` after pulling dependency changes.
- If a local model is missing, reconnect to the internet once and restart the app.
- Use `PYTHONPATH=. pytest` to check the project after making changes.
