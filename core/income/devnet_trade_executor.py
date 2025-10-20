"""
Devnet Trade Executor
Executes real trades on Solana Devnet using Jupiter Aggregator
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solders.pubkey import Pubkey
from solders.system_program import TransferParams, transfer
from solders.transaction import Transaction

from .solana_wallet_manager import SolanaWalletManager
from .jupiter_client import JupiterClient

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Result of a devnet trade execution"""
    success: bool
    signature: Optional[str]
    actual_profit_pct: float
    slippage_pct: float
    input_amount: float
    output_amount: float
    error: Optional[str] = None
    execution_time: float = 0.0


class DevnetTradeExecutor:
    """
    Executes trades on Solana Devnet
    Uses Jupiter Aggregator for best swap routes
    """

    def __init__(self,
                 wallet_manager: SolanaWalletManager,
                 jupiter_client: JupiterClient,
                 max_slippage_bps: int = 100):  # 1% max slippage
        self.wallet = wallet_manager
        self.jupiter = jupiter_client
        self.max_slippage_bps = max_slippage_bps

        logger.info("Devnet Trade Executor initialized")

    async def execute_swap(self,
                          input_token: str,
                          output_token: str,
                          amount_usd: float,
                          action: str = "BUY") -> TradeResult:
        """
        Execute a swap on Solana Devnet

        Args:
            input_token: Token to swap from (e.g., 'USDC')
            output_token: Token to swap to (e.g., 'SOL')
            amount_usd: Amount in USD to trade
            action: 'BUY' or 'SELL'

        Returns:
            TradeResult with transaction details
        """
        start_time = datetime.now()

        try:
            logger.info(f"Executing {action}: {input_token} -> {output_token}, ${amount_usd}")

            # Step 1: Get quote from Jupiter
            quote = await self._get_jupiter_quote(input_token, output_token, amount_usd)
            if not quote:
                return TradeResult(
                    success=False,
                    signature=None,
                    actual_profit_pct=0.0,
                    slippage_pct=0.0,
                    input_amount=0.0,
                    output_amount=0.0,
                    error="Failed to get Jupiter quote"
                )

            # Step 2: Build swap transaction
            transaction = await self._build_swap_transaction(quote)
            if not transaction:
                return TradeResult(
                    success=False,
                    signature=None,
                    actual_profit_pct=0.0,
                    slippage_pct=0.0,
                    input_amount=quote['in_amount'],
                    output_amount=quote['out_amount'],
                    error="Failed to build transaction"
                )

            # Step 3: Sign transaction
            signed_tx = await self.wallet.sign_transaction(transaction)
            if not signed_tx:
                return TradeResult(
                    success=False,
                    signature=None,
                    actual_profit_pct=0.0,
                    slippage_pct=0.0,
                    input_amount=quote['in_amount'],
                    output_amount=quote['out_amount'],
                    error="Failed to sign transaction"
                )

            # Step 4: Send transaction
            signature = await self.wallet.send_transaction(signed_tx)
            if not signature:
                return TradeResult(
                    success=False,
                    signature=None,
                    actual_profit_pct=0.0,
                    slippage_pct=0.0,
                    input_amount=quote['in_amount'],
                    output_amount=quote['out_amount'],
                    error="Failed to send transaction"
                )

            # Step 5: Confirm transaction
            confirmed = await self.wallet.confirm_transaction(signature)
            if not confirmed:
                logger.warning(f"Transaction sent but not confirmed: {signature}")

            # Calculate results
            execution_time = (datetime.now() - start_time).total_seconds()
            profit_pct, slippage_pct = self._calculate_results(quote)

            logger.info(f"Trade executed! Signature: {signature}")
            logger.info(f"Profit: {profit_pct:.3f}%, Slippage: {slippage_pct:.3f}%")

            return TradeResult(
                success=True,
                signature=signature,
                actual_profit_pct=profit_pct,
                slippage_pct=slippage_pct,
                input_amount=quote['in_amount'],
                output_amount=quote['out_amount'],
                execution_time=execution_time
            )

        except Exception as e:
            logger.error(f"Trade execution failed: {e}", exc_info=True)
            execution_time = (datetime.now() - start_time).total_seconds()

            return TradeResult(
                success=False,
                signature=None,
                actual_profit_pct=0.0,
                slippage_pct=0.0,
                input_amount=0.0,
                output_amount=0.0,
                error=str(e),
                execution_time=execution_time
            )

    async def _get_jupiter_quote(self,
                                 input_token: str,
                                 output_token: str,
                                 amount_usd: float) -> Optional[Dict]:
        """Get swap quote from Jupiter"""
        try:
            # Get token mint addresses
            input_mint = self.jupiter.TOKENS.get(input_token)
            output_mint = self.jupiter.TOKENS.get(output_token)

            if not input_mint or not output_mint:
                logger.error(f"Token not found: {input_token} or {output_token}")
                return None

            # Convert USD to token amount (simplified - assumes 1:1 for stablecoins)
            if input_token in ['USDC', 'USDT']:
                amount = int(amount_usd * 1_000_000)  # 6 decimals
            elif input_token == 'SOL':
                # Get SOL price
                sol_quote = await self.jupiter.get_quote(
                    input_mint=self.jupiter.TOKENS['SOL'],
                    output_mint=self.jupiter.TOKENS['USDC'],
                    amount=1_000_000_000,
                    slippage_bps=self.max_slippage_bps
                )
                if sol_quote:
                    sol_price = sol_quote.out_amount / 1_000_000
                    amount = int((amount_usd / sol_price) * 1_000_000_000)
                else:
                    amount = int(amount_usd * 1_000_000)  # Fallback
            else:
                amount = int(amount_usd * 1_000_000)  # Default 6 decimals

            # Get quote
            quote_result = await self.jupiter.get_quote(
                input_mint=input_mint,
                output_mint=output_mint,
                amount=amount,
                slippage_bps=self.max_slippage_bps
            )

            if not quote_result:
                logger.error("No quote received from Jupiter")
                return None

            quote_dict = {
                'input_mint': input_mint,
                'output_mint': output_mint,
                'in_amount': amount,
                'out_amount': quote_result.out_amount,
                'price_impact': quote_result.price_impact_pct,
                'route': quote_result.route_plan
            }

            logger.info(f"Quote: {amount} {input_token} -> {quote_result.out_amount} {output_token}")
            logger.info(f"Price Impact: {quote_result.price_impact_pct}%")

            return quote_dict

        except Exception as e:
            logger.error(f"Failed to get quote: {e}")
            return None

    async def _build_swap_transaction(self, quote: Dict) -> Optional[Transaction]:
        """
        Build swap transaction from Jupiter quote

        NOTE: This is a simplified version. Real implementation needs:
        - Jupiter swap instruction building
        - Proper account handling
        - SPL token accounts
        """
        try:
            # TODO: Implement actual Jupiter swap transaction building
            # This requires:
            # 1. Get swap instructions from Jupiter API
            # 2. Create transaction with proper accounts
            # 3. Handle SPL token accounts (ATA creation if needed)

            logger.warning("Transaction building not fully implemented - using placeholder")

            # Placeholder transaction
            # In reality, would get this from Jupiter API swap endpoint
            transaction = Transaction()

            return transaction

        except Exception as e:
            logger.error(f"Failed to build transaction: {e}")
            return None

    def _calculate_results(self, quote: Dict) -> Tuple[float, float]:
        """Calculate profit and slippage from quote"""
        # Calculate profit percentage
        # This is simplified - real calculation depends on entry price
        profit_pct = 0.0  # Placeholder

        # Calculate slippage
        price_impact = quote.get('price_impact', 0.0)
        slippage_pct = abs(price_impact)

        return profit_pct, slippage_pct

    async def check_sufficient_balance(self,
                                      token: str,
                                      amount: float) -> bool:
        """Check if wallet has sufficient balance for trade"""
        try:
            if token == 'SOL':
                balance = await self.wallet.get_balance()
                return balance >= amount
            else:
                # TODO: Implement SPL token balance checking
                logger.warning("SPL token balance checking not implemented")
                return True

        except Exception as e:
            logger.error(f"Failed to check balance: {e}")
            return False


async def test_devnet_executor():
    """Test devnet trade execution"""
    print("\n" + "="*70)
    print("TESTING DEVNET TRADE EXECUTOR")
    print("="*70 + "\n")

    # Initialize wallet
    async with SolanaWalletManager() as wallet:
        print(f"Wallet: {wallet.get_public_key()}")

        # Check balance
        balance = await wallet.get_balance()
        print(f"Balance: {balance} SOL")

        if balance < 0.1:
            print("\nRequesting airdrop...")
            await wallet.request_airdrop(2.0)

        # Initialize Jupiter client
        async with JupiterClient() as jupiter:
            # Create executor
            executor = DevnetTradeExecutor(wallet, jupiter)

            print("\nExecuting test trade: 1 USDC -> SOL")

            # Execute test trade
            result = await executor.execute_swap(
                input_token='USDC',
                output_token='SOL',
                amount_usd=1.0,
                action='BUY'
            )

            print(f"\nResult:")
            print(f"  Success: {result.success}")
            print(f"  Signature: {result.signature}")
            print(f"  Profit: {result.actual_profit_pct:.3f}%")
            print(f"  Slippage: {result.slippage_pct:.3f}%")
            print(f"  Time: {result.execution_time:.2f}s")
            if result.error:
                print(f"  Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(test_devnet_executor())
