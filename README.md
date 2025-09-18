# Poggio MCP Client Example

A minimal standalone client for the Poggio MCP server that demonstrates:

1. Creating an account (with graceful error handling for existing accounts)
2. Polling search results until specific pages are ready
3. Fetching the named pages and logging their content

## Setup

```bash
# Install dependencies
uv sync
```

## Usage

```bash
# Run the client
uv run python main.py
```

## Configuration

The client looks for these pages for domain "example.com":
- "Overview"
- "Business Case"

It will poll every 5 seconds for up to 10 minutes waiting for these pages to be available.

## Environment Variables

- `POGGIO_MCP_SERVER_URL`: URL of the MCP server (default:
  https://mcp.poggio.io/mcp)
- `POGGIO_AUTH_TOKEN`: Authentication token for the Poggio API (required)

## What it does

1. **Creates account**: Calls the `create_account` MCP tool for domain "example.com"
2. **Polls for readiness**: Uses the `search` tool to check page status every 5 seconds
3. **Fetches content**: Once pages show `status: "ready"`, fetches full content via `fetch` tool
4. **Logs results**: Outputs page titles, URLs, and content previews to console
