"""Simple Poggio MCP client example."""

import asyncio
import json
import logging
import os
from typing import List, Set

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

# Configuration
DOMAIN = "example.com"
TARGET_PAGES = {"Overview", "Business Case"}
POLL_INTERVAL = 5  # seconds
MAX_POLL_TIME = 600  # 10 minutes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ItemWithMetadata(BaseModel):
    """Item with metadata from MCP server."""
    id: str
    title: str
    text: str
    url: str
    metadata: Optional[Dict[str, Any]] = None


class CreateAccountResponse(BaseModel):
    """Response from create_account tool."""
    id: str
    org_id: str
    domain: str


async def create_account(session: ClientSession, domain: str) -> CreateAccountResponse:
    """Create an account, handling the case where it already exists."""
    try:
        result = await session.call_tool("create_account", {"account_domain": domain})

        if result.content and len(result.content) > 0:
            # Parse the result content
            if hasattr(result.content[0], 'text'):
                account_data = result.content[0].text
                if isinstance(account_data, str):
                    try:
                        account_dict = json.loads(account_data)
                    except json.JSONDecodeError:
                        # If not JSON, assume it's a simple response
                        account_dict = {
                            "id": "created",
                            "org_id": "default",
                            "domain": domain
                        }
                else:
                    account_dict = account_data
                return CreateAccountResponse.model_validate(account_dict)

    except Exception as e:
        logger.error(f"Error creating account: {e}")
        # Check if error indicates account already exists
        if "already exists" in str(e).lower() or "409" in str(e):
            logger.info(f"Account for domain '{domain}' already exists")
        else:
            logger.info(f"Assuming account for domain '{domain}' already exists")

        # Return a mock response for existing accounts
        return CreateAccountResponse(
            id="existing",
            org_id="existing",
            domain=domain
        )


async def search_items(session: ClientSession, query: str) -> List[ItemWithMetadata]:
    """Search for items."""
    try:
        result = await session.call_tool("search", {"query": query})

        if result.content and len(result.content) > 0:
            # The MCP server returns the list of items directly
            if hasattr(result.content[0], 'text'):
                items_data = result.content[0].text
                if isinstance(items_data, str):
                    try:
                        items_list = json.loads(items_data)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse search response: {items_data}")
                        return []
                else:
                    items_list = items_data

                if isinstance(items_list, list):
                    return [ItemWithMetadata.model_validate(item) for item in items_list]

    except Exception as e:
        logger.error(f"Error searching: {e}")

    return []


async def fetch_item(session: ClientSession, item_id: str) -> Optional[ItemWithMetadata]:
    """Fetch a specific item."""
    try:
        result = await session.call_tool("fetch", {"id": item_id})

        if result.content and len(result.content) > 0:
            # Parse the result content
            if hasattr(result.content[0], 'text'):
                item_data = result.content[0].text
                if isinstance(item_data, str):
                    try:
                        item_dict = json.loads(item_data)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse fetch response: {item_data}")
                        return None
                else:
                    item_dict = item_data
                return ItemWithMetadata.model_validate(item_dict)

        return None

    except Exception as e:
        logger.error(f"Error fetching item {item_id}: {e}")
        return None


async def wait_for_pages(session: ClientSession, domain: str, target_pages: Set[str]) -> List[ItemWithMetadata]:
    """Poll search results until target pages are available."""
    logger.info(f"Polling for pages: {', '.join(target_pages)}")

    start_time = asyncio.get_event_loop().time()

    while True:
        current_time = asyncio.get_event_loop().time()
        elapsed = current_time - start_time

        if elapsed > MAX_POLL_TIME:
            logger.warning(f"Timeout after {MAX_POLL_TIME} seconds.")
            break

        # Search for items
        items = await search_items(session, domain)
        logger.info(f"Found {len(items)} items in search results")

        # Check which target pages are available and ready
        available_pages = []
        for item in items:
            # Skip workspace info items
            if item.id == "workspace_info":
                continue

            # Check if this item matches one of our target pages
            if item.title in target_pages:
                status = item.metadata.get("status", "unknown") if item.metadata else "unknown"
                logger.info(f"Found page '{item.title}' with status: {status}")

                if status == "ready":
                    available_pages.append(item)
                elif status in ["pending", "running"]:
                    logger.info(f"Page '{item.title}' is not ready yet (status: {status})")
                elif status == "error":
                    logger.warning(f"Page '{item.title}' has error status")

        # Check if we have all target pages ready
        ready_page_titles = {page.title for page in available_pages}
        missing_pages = target_pages - ready_page_titles

        if not missing_pages:
            logger.info("All target pages are ready!")
            return available_pages

        logger.info(f"Still waiting for pages: {', '.join(missing_pages)} (elapsed: {elapsed:.1f}s)")
        await asyncio.sleep(POLL_INTERVAL)

    return []


async def main():
    """Main execution function."""
    # Get configuration from environment variables
    server_url = os.getenv("POGGIO_MCP_SERVER_URL", "https://mcp.poggio.io/mcp")
    auth_token = os.getenv("POGGIO_AUTH_TOKEN")

    if not auth_token:
        logger.error("POGGIO_AUTH_TOKEN environment variable is required")
        return

    logger.info(f"Starting Poggio MCP client for domain: {DOMAIN}")
    logger.info(f"Target pages: {', '.join(TARGET_PAGES)}")
    logger.info(f"Server URL: {server_url}")

    # Use MCP streamable HTTP client as intended - as a context manager
    async with streamablehttp_client(
        url=server_url,
        headers={"Authorization": f"Bearer {auth_token}"}
    ) as (read_stream, write_stream, get_session_id):

        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            logger.info("Connected to MCP server")

            # Step 1: Create account (gracefully handle existing accounts)
            logger.info(f"Creating account for domain: {DOMAIN}")
            account = await create_account(session, DOMAIN)
            logger.info(f"Account ready - ID: {account.id}, Org: {account.org_id}")

            # Step 2: Poll for target pages
            logger.info("Starting to poll for target pages...")
            ready_pages = await wait_for_pages(session, DOMAIN, TARGET_PAGES)

            # Step 3: Fetch and log the pages
            if ready_pages:
                logger.info(f"Fetching {len(ready_pages)} ready pages...")

                for page in ready_pages:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"Fetching page: {page.title}")
                    logger.info(f"Page ID: {page.id}")
                    logger.info(f"Status: {page.metadata.get('status', 'unknown') if page.metadata else 'unknown'}")

                    # Fetch full content
                    full_page = await fetch_item(session, page.id)
                    if full_page:
                        logger.info(f"Title: {full_page.title}")
                        logger.info(f"URL: {full_page.url}")
                        logger.info(f"Content length: {len(full_page.text)} characters")
                        logger.info(f"Content preview (first 500 chars):")
                        logger.info(f"{full_page.text[:500]}{'...' if len(full_page.text) > 500 else ''}")
                    else:
                        logger.error(f"Failed to fetch full content for page: {page.title}")

                    logger.info(f"{'='*80}\n")
            else:
                logger.warning("No target pages were found or became ready within the timeout period")


if __name__ == "__main__":
    asyncio.run(main())
