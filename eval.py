"""Evaluation script using LLM-as-a-judge to score page content."""

import argparse
import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Set
import openai

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RubricItem(BaseModel):
    """Individual rubric scoring criteria."""
    name: str
    value: int
    criteria: str


class Dimension(BaseModel):
    """Evaluation dimension with its rubric."""
    name: str
    rubric: List[RubricItem]


class EvalConfig(BaseModel):
    """Configuration for evaluation dimensions and rubrics."""
    dimensions: List[Dimension]


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


class EvaluationResult(BaseModel):
    """Result of dimension evaluation."""
    dimension_name: str
    score: int
    score_name: str
    reasoning: str


class EvaluationResponse(BaseModel):
    """Response model for LLM evaluation without dimension_name."""
    score: int = Field(..., description="The score based on the rubric")
    score_name: str = Field(..., description="The name of the score level")
    reasoning: str = Field(..., description="Detailed explanation of why this score was chosen")


class EvaluationSummary(BaseModel):
    """Summary of all evaluations."""
    domain: str
    total_content_length: int
    dimension_scores: List[EvaluationResult]
    average_score: float
    total_score: int
    max_possible_score: int


async def create_account(session: ClientSession, domain: str) -> CreateAccountResponse:
    """Create an account, handling the case where it already exists."""
    try:
        result = await session.call_tool("create_account", {"account_domain": domain})

        if result.content and len(result.content) > 0:
            if hasattr(result.content[0], 'text'):
                account_data = result.content[0].text
                if isinstance(account_data, str):
                    try:
                        account_dict = json.loads(account_data)
                    except json.JSONDecodeError:
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
        if "already exists" in str(e).lower() or "409" in str(e):
            logger.info(f"Account for domain '{domain}' already exists")
        else:
            logger.info(f"Assuming account for domain '{domain}' already exists")

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


async def fetch_all_domain_content(session: ClientSession, domain: str) -> str:
    """Fetch all page content for a domain and concatenate it."""
    logger.info(f"Fetching all content for domain: {domain}")

    # Search for all items in the domain
    items = await search_items(session, domain)
    logger.info(f"Found {len(items)} items for domain {domain}")

    content_parts = []

    for item in items:
        # Skip workspace info items
        if item.id == "workspace_info":
            continue

        # Check if item is ready
        if item.metadata and item.metadata.get("status") == "ready":
            logger.info(f"Fetching content for: {item.title}")
            full_item = await fetch_item(session, item.id)
            if full_item and full_item.text:
                content_parts.append(f"=== {full_item.title} ===\n{full_item.text}\n")
                logger.info(f"Added {len(full_item.text)} characters from '{full_item.title}'")
        else:
            status = item.metadata.get("status", "unknown") if item.metadata else "unknown"
            logger.warning(f"Skipping '{item.title}' - status: {status}")

    concatenated_content = "\n".join(content_parts)
    logger.info(f"Total concatenated content length: {len(concatenated_content)} characters")

    return concatenated_content


async def evaluate_dimension(
    client: openai.AsyncOpenAI,
    content: str,
    dimension: Dimension,
    model: str
) -> EvaluationResult:
    """Evaluate content on a single dimension using LLM-as-a-judge."""
    logger.info(f"Evaluating dimension: {dimension.name}")

    # Create the rubric description
    rubric_text = "\n".join([
        f"{item.value} - {item.name.upper()}: {item.criteria}"
        for item in dimension.rubric
    ])

    prompt = f"""You are an expert evaluator analyzing content quality on the dimension of "{dimension.name}".

RUBRIC:
{rubric_text}

CONTENT TO EVALUATE:
{content}

Please evaluate the content based on the rubric above. You must:
1. Carefully analyze the content against each level of the rubric
2. Select the most appropriate score (1-5) that best matches the content quality
3. Provide detailed reasoning for your score

Use the submit_evaluation function to provide your assessment."""

    # Generate JSON schema from Pydantic model and ensure additionalProperties is False
    schema = EvaluationResponse.model_json_schema()
    schema["additionalProperties"] = False

    # Ensure all nested objects also have additionalProperties: false
    if "properties" in schema:
        for prop_name, prop_schema in schema["properties"].items():
            if prop_schema.get("type") == "object":
                prop_schema["additionalProperties"] = False

    tools = [
        {
            "type": "function",
            "function": {
                "name": "submit_evaluation",
                "description": "Submit the evaluation result for this dimension",
                "parameters": schema,
                "strict": True
            }
        }
    ]

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "submit_evaluation"}},
            temperature=1
        )

        # Extract the function call arguments and validate with Pydantic
        tool_call = response.choices[0].message.tool_calls[0]
        eval_response = EvaluationResponse.model_validate_json(tool_call.function.arguments)

        return EvaluationResult(
            dimension_name=dimension.name,
            score=eval_response.score,
            score_name=eval_response.score_name,
            reasoning=eval_response.reasoning
        )

    except Exception as e:
        logger.error(f"Error evaluating dimension {dimension.name}: {e}")
        # Return a default result in case of error
        return EvaluationResult(
            dimension_name=dimension.name,
            score=1,
            score_name="error",
            reasoning=f"Error occurred during evaluation: {str(e)}"
        )


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Evaluate domain content using LLM-as-a-judge")
    parser.add_argument("domain", help="Target domain to evaluate")
    parser.add_argument("--model", default="gpt-5-mini", help="OpenAI model to use (default: gpt-5-mini)")
    args = parser.parse_args()

    # Load evaluation configuration
    try:
        with open("config.json", "r") as f:
            config_data = json.load(f)
        config = EvalConfig.model_validate(config_data)
        logger.info(f"Loaded evaluation config with {len(config.dimensions)} dimensions")
    except Exception as e:
        logger.error(f"Error loading config.json: {e}")
        return

    # Setup OpenAI client
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable is required")
        return

    openai_client = openai.AsyncOpenAI(api_key=openai_api_key)

    # Setup MCP client
    server_url = os.getenv("POGGIO_MCP_SERVER_URL", "https://mcp.poggio.io/mcp")
    auth_token = os.getenv("POGGIO_AUTH_TOKEN")

    if not auth_token:
        logger.error("POGGIO_AUTH_TOKEN environment variable is required")
        return

    logger.info(f"Starting evaluation for domain: {args.domain}")
    logger.info(f"Using OpenAI model: {args.model}")
    logger.info(f"Server URL: {server_url}")

    async with streamablehttp_client(
        url=server_url,
        headers={"Authorization": f"Bearer {auth_token}"}
    ) as (read_stream, write_stream, get_session_id):

        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            logger.info("Connected to MCP server")

            # Create account
            logger.info(f"Creating account for domain: {args.domain}")
            account = await create_account(session, args.domain)
            logger.info(f"Account ready - ID: {account.id}, Org: {account.org_id}")

            # Fetch all domain content
            content = await fetch_all_domain_content(session, args.domain)

            if not content.strip():
                logger.error("No content found for domain. Cannot perform evaluation.")
                return

            # Evaluate each dimension
            evaluation_results = []

            for dimension in config.dimensions:
                result = await evaluate_dimension(
                    openai_client,
                    content,
                    dimension,
                    args.model
                )
                evaluation_results.append(result)

            # Calculate summary statistics
            total_score = sum(r.score for r in evaluation_results)
            max_possible = len(config.dimensions) * 5  # Assuming max score is 5
            average_score = total_score / len(evaluation_results) if evaluation_results else 0

            summary = EvaluationSummary(
                domain=args.domain,
                total_content_length=len(content),
                dimension_scores=evaluation_results,
                average_score=average_score,
                total_score=total_score,
                max_possible_score=max_possible
            )

            # Output results
            print("\n" + "="*80)
            print(f"EVALUATION RESULTS for {args.domain}")
            print("="*80)
            print(f"Content Length: {summary.total_content_length:,} characters")
            print(f"Model Used: {args.model}")
            print(f"Total Score: {summary.total_score}/{summary.max_possible_score}")
            print(f"Average Score: {summary.average_score:.2f}/5.00")
            print(f"Percentage: {(summary.average_score/5)*100:.1f}%")
            print()

            for result in evaluation_results:
                print(f"Dimension: {result.dimension_name}")
                print(f"Score: {result.score}/5 ({result.score_name.upper()})")
                print(f"Reasoning: {result.reasoning}")
                print("-" * 40)

            # Save detailed results to JSON file
            output_file = f"eval_results_{args.domain.replace('.', '_')}.json"
            with open(output_file, "w") as f:
                json.dump(summary.model_dump(), f, indent=2)
            print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
