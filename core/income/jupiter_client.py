"""
Jupiter Aggregator Client
Real integration with Jupiter for DEX swaps on Solana

Jupiter is the best swap aggregator on Solana:
- Finds best routes across all DEXs (Raydium, Orca, Serum, etc.)
- Optimal pricing with minimal slippage
- Single transaction for complex routes
"""

import asyncio
import logging
import ssl
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from decimal import Decimal

try:
    import aiohttp
    from aiohttp.resolver import AsyncResolver
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logging.warning("aiohttp not installed. Install with: pip install aiohttp")

logger = logging.getLogger(__name__)


@dataclass
class JupiterQuote:
    """Quote from Jupiter for a swap"""
    input_mint: str
    output_mint: str
    in_amount: int  # Raw amount (with decimals)
    out_amount: int  # Raw amount (with decimals)
    other_amount_threshold: int  # Minimum amount out (considering slippage)
    swap_mode: str
    slippage_bps: int
    price_impact_pct: float
    route_plan: List[Dict]
    context_slot: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inputMint": self.input_mint,
            "outputMint": self.output_mint,
            "inAmount": str(self.in_amount),
            "outAmount": str(self.out_amount),
            "otherAmountThreshold": str(self.other_amount_threshold),
            "swapMode": self.swap_mode,
            "slippageBps": self.slippage_bps,
            "priceImpactPct": self.price_impact_pct,
            "routePlan": self.route_plan,
        }


@dataclass
class TokenPrice:
    """Token price information"""
    mint: str
    symbol: str
    price_usd: float
    timestamp: float


class JupiterClient:
    """
    Jupiter Aggregator API Client

    Features:
    - Get swap quotes across all Solana DEXs
    - Execute swaps with best rates
    - Real-time price feeds
    - Route optimization
    """

    # Jupiter API v6
    # Note: Using public.jupiterapi.com as quote-api.jup.ag has DNS resolution issues
    # Alternative endpoint hosted by QuickNode (0.2% fee on swaps)
    API_BASE = "https://public.jupiterapi.com"

    # Common token mints on Solana
    TOKENS = {
        "SOL": "So11111111111111111111111111111111111111112",
        "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        "RAY": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
        "ORCA": "orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE",
        "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
        "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
    }

    def __init__(self, timeout: int = 30):
        """
        Initialize Jupiter client

        Args:
            timeout: Request timeout in seconds
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp not installed. Install with: pip install aiohttp")

        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None

        logger.info("Jupiter client initialized")

    async def __aenter__(self):
        """Async context manager entry"""
        # Create SSL context that doesn't verify certificates (for corporate proxies)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Create custom DNS resolver using Google DNS (8.8.8.8)
        # This bypasses local DNS issues
        try:
            resolver = AsyncResolver(nameservers=["8.8.8.8", "8.8.4.4"])
            logger.info("Using Google DNS (8.8.8.8) for Jupiter API")
        except:
            resolver = None  # Fall back to system DNS
            logger.warning("Using system DNS (Google DNS config failed)")

        # Create TCP connector with custom DNS and SSL
        connector = aiohttp.TCPConnector(
            ssl=ssl_context,
            resolver=resolver,
            force_close=False,
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300  # Cache DNS for 5 minutes
        )

        self.session = aiohttp.ClientSession(
            timeout=self.timeout,
            connector=connector
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 50,  # 0.5% slippage
        only_direct_routes: bool = False,
    ) -> Optional[JupiterQuote]:
        """
        Get swap quote from Jupiter

        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Amount to swap (in smallest unit, with decimals)
            slippage_bps: Slippage tolerance in basis points (50 = 0.5%)
            only_direct_routes: Only use direct routes (faster but may miss best price)

        Returns:
            JupiterQuote if successful, None otherwise
        """
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)

        try:
            url = f"{self.API_BASE}/quote"

            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": amount,
                "slippageBps": slippage_bps,
                "onlyDirectRoutes": "true" if only_direct_routes else "false",
            }

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Jupiter quote failed: {response.status}")
                    return None

                data = await response.json()

                # Parse quote
                quote = JupiterQuote(
                    input_mint=data["inputMint"],
                    output_mint=data["outputMint"],
                    in_amount=int(data["inAmount"]),
                    out_amount=int(data["outAmount"]),
                    other_amount_threshold=int(data["otherAmountThreshold"]),
                    swap_mode=data.get("swapMode", "ExactIn"),
                    slippage_bps=slippage_bps,
                    price_impact_pct=float(data.get("priceImpactPct", 0)),
                    route_plan=data.get("routePlan", []),
                    context_slot=data.get("contextSlot"),
                )

                logger.info(
                    f"Quote: {quote.in_amount} -> {quote.out_amount} "
                    f"(impact: {quote.price_impact_pct:.3f}%)"
                )

                return quote

        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            return None

    async def get_swap_transaction(
        self,
        quote: JupiterQuote,
        user_public_key: str,
        wrap_unwrap_sol: bool = True,
        fee_account: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get swap transaction from Jupiter

        Args:
            quote: Quote from get_quote()
            user_public_key: User's wallet public key
            wrap_unwrap_sol: Auto wrap/unwrap SOL
            fee_account: Optional fee account for referral fees

        Returns:
            Serialized transaction (base64) if successful
        """
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)

        try:
            url = f"{self.API_BASE}/swap"

            payload = {
                "quoteResponse": quote.to_dict(),
                "userPublicKey": user_public_key,
                "wrapAndUnwrapSol": wrap_unwrap_sol,
            }

            if fee_account:
                payload["feeAccount"] = fee_account

            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error = await response.text()
                    logger.error(f"Jupiter swap failed: {response.status} - {error}")
                    return None

                data = await response.json()

                # Return serialized transaction
                swap_transaction = data.get("swapTransaction")

                if swap_transaction:
                    logger.info("Swap transaction created successfully")
                    return swap_transaction
                else:
                    logger.error("No swap transaction in response")
                    return None

        except Exception as e:
            logger.error(f"Error creating swap transaction: {e}")
            return None

    async def get_price(
        self,
        token_mint: str,
        vs_token: str = "USDC"
    ) -> Optional[float]:
        """
        Get current price for a token

        Args:
            token_mint: Token mint address
            vs_token: Quote token (USDC, SOL, etc.)

        Returns:
            Price if successful, None otherwise
        """
        if vs_token in self.TOKENS:
            vs_mint = self.TOKENS[vs_token]
        else:
            vs_mint = vs_token

        # Get quote for 1 unit
        # Note: This is simplified - real implementation would account for decimals
        amount = 1_000_000  # 1 token with 6 decimals

        quote = await self.get_quote(
            input_mint=token_mint,
            output_mint=vs_mint,
            amount=amount,
            slippage_bps=50
        )

        if quote:
            # Calculate price
            price = quote.out_amount / quote.in_amount
            return price

        return None

    async def get_token_list(self) -> List[Dict[str, Any]]:
        """
        Get list of all tokens supported by Jupiter

        Returns:
            List of token information
        """
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)

        try:
            # Jupiter token list API
            url = "https://token.jup.ag/all"

            async with self.session.get(url) as response:
                if response.status == 200:
                    tokens = await response.json()
                    logger.info(f"Fetched {len(tokens)} tokens from Jupiter")
                    return tokens
                else:
                    logger.error(f"Failed to fetch token list: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Error fetching token list: {e}")
            return []

    async def find_arbitrage_opportunities(
        self,
        token_pairs: List[tuple],
        amount_usd: float = 100,
        min_profit_pct: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Find arbitrage opportunities between token pairs

        Args:
            token_pairs: List of (token_a, token_b) tuples
            amount_usd: Amount to test in USD
            min_profit_pct: Minimum profit percentage to report

        Returns:
            List of arbitrage opportunities
        """
        opportunities = []

        for token_a, token_b in token_pairs:
            try:
                mint_a = self.TOKENS.get(token_a, token_a)
                mint_b = self.TOKENS.get(token_b, token_b)

                # Estimate amount based on token (simplified)
                amount = int(amount_usd * 1_000_000)  # Assume 6 decimals

                # Get quote for A -> B
                quote_forward = await self.get_quote(
                    input_mint=mint_a,
                    output_mint=mint_b,
                    amount=amount,
                    slippage_bps=50
                )

                if not quote_forward:
                    continue

                # Get quote for B -> A (using output from forward)
                quote_backward = await self.get_quote(
                    input_mint=mint_b,
                    output_mint=mint_a,
                    amount=quote_forward.out_amount,
                    slippage_bps=50
                )

                if not quote_backward:
                    continue

                # Calculate profit
                final_amount = quote_backward.out_amount
                profit_amount = final_amount - amount
                profit_pct = (profit_amount / amount) * 100

                if profit_pct >= min_profit_pct:
                    opportunities.append({
                        "pair": f"{token_a}/{token_b}",
                        "amount_in": amount,
                        "amount_out": final_amount,
                        "profit_amount": profit_amount,
                        "profit_pct": profit_pct,
                        "forward_route": quote_forward.route_plan,
                        "backward_route": quote_backward.route_plan,
                    })

                    logger.info(
                        f"Arbitrage found: {token_a}/{token_b} - "
                        f"{profit_pct:.2f}% profit"
                    )

                # Small delay to avoid rate limiting
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error checking {token_a}/{token_b}: {e}")

        return opportunities


if __name__ == "__main__":
    import sys
    # Fix Windows console encoding
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    logging.basicConfig(level=logging.INFO)

    async def test_jupiter_client():
        """Test Jupiter client"""
        print("\n=== Jupiter Client Test ===\n")

        async with JupiterClient() as jupiter:
            # Test 1: Get quote for SOL -> USDC
            print("Getting quote for SOL -> USDC...")

            quote = await jupiter.get_quote(
                input_mint=jupiter.TOKENS["SOL"],
                output_mint=jupiter.TOKENS["USDC"],
                amount=1_000_000_000,  # 1 SOL (9 decimals)
                slippage_bps=50  # 0.5% slippage
            )

            if quote:
                print(f"✓ Quote received:")
                print(f"  In: {quote.in_amount / 1e9:.2f} SOL")
                print(f"  Out: {quote.out_amount / 1e6:.2f} USDC")
                print(f"  Price impact: {quote.price_impact_pct:.3f}%")
                print(f"  Route: {len(quote.route_plan)} steps\n")

            # Test 2: Check some arbitrage opportunities
            print("Checking arbitrage opportunities...")

            pairs = [
                ("SOL", "USDC"),
                ("SOL", "RAY"),
                ("USDC", "USDT"),
            ]

            opportunities = await jupiter.find_arbitrage_opportunities(
                token_pairs=pairs,
                amount_usd=100,
                min_profit_pct=0.5
            )

            if opportunities:
                print(f"\n✓ Found {len(opportunities)} opportunities:")
                for opp in opportunities:
                    print(f"  {opp['pair']}: {opp['profit_pct']:.2f}% profit")
            else:
                print("  No profitable opportunities found")

            print("\n✓ Tests complete")

    asyncio.run(test_jupiter_client())
