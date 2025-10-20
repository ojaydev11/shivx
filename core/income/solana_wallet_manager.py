"""
Solana Wallet Manager for Devnet Trading
Handles wallet creation, loading, and transaction signing
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from base64 import b64decode, b64encode
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.transaction import Transaction

logger = logging.getLogger(__name__)


class SolanaWalletManager:
    """
    Manages Solana wallets for devnet/mainnet trading
    """

    def __init__(self,
                 wallet_path: str = "data/wallets/devnet_trading_wallet.json",
                 rpc_url: str = "https://api.devnet.solana.com"):
        self.wallet_path = Path(wallet_path)
        self.rpc_url = rpc_url
        self.keypair: Optional[Keypair] = None
        self.client: Optional[AsyncClient] = None

    async def initialize(self) -> bool:
        """Initialize wallet and connection"""
        try:
            # Create wallet directory if not exists
            self.wallet_path.parent.mkdir(parents=True, exist_ok=True)

            # Load or generate keypair
            if self.wallet_path.exists():
                logger.info(f"Loading existing wallet from {self.wallet_path}")
                self.keypair = self._load_keypair()
            else:
                logger.info(f"Generating new wallet at {self.wallet_path}")
                self.keypair = self._generate_keypair()

            if not self.keypair:
                logger.error("Failed to initialize keypair")
                return False

            # Initialize RPC client
            self.client = AsyncClient(self.rpc_url, commitment=Confirmed)

            logger.info(f"Wallet initialized: {self.get_public_key()}")
            logger.info(f"Connected to: {self.rpc_url}")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize wallet: {e}", exc_info=True)
            return False

    def _generate_keypair(self) -> Keypair:
        """Generate new keypair and save to file"""
        keypair = Keypair()

        # Save to file (array of bytes)
        wallet_data = {
            'public_key': str(keypair.pubkey()),
            'secret_key': b64encode(bytes(keypair)).decode('utf-8')
        }

        with open(self.wallet_path, 'w') as f:
            json.dump(wallet_data, f, indent=2)

        logger.info(f"New wallet generated and saved to {self.wallet_path}")
        return keypair

    def _load_keypair(self) -> Optional[Keypair]:
        """Load keypair from file"""
        try:
            with open(self.wallet_path, 'r') as f:
                wallet_data = json.load(f)

            secret_key_bytes = b64decode(wallet_data['secret_key'])
            keypair = Keypair.from_bytes(secret_key_bytes)

            logger.info(f"Wallet loaded: {keypair.pubkey()}")
            return keypair

        except Exception as e:
            logger.error(f"Failed to load keypair: {e}")
            return None

    def get_public_key(self) -> str:
        """Get wallet public key as string"""
        if not self.keypair:
            return ""
        return str(self.keypair.pubkey())

    async def get_balance(self, token_mint: Optional[str] = None) -> float:
        """
        Get wallet balance
        If token_mint is None, returns SOL balance
        Otherwise returns SPL token balance
        """
        try:
            if not self.client or not self.keypair:
                logger.error("Wallet not initialized")
                return 0.0

            if token_mint is None:
                # Get SOL balance
                response = await self.client.get_balance(self.keypair.pubkey())
                if response.value is not None:
                    # Convert lamports to SOL
                    return response.value / 1_000_000_000
                return 0.0
            else:
                # Get SPL token balance
                # TODO: Implement SPL token balance checking
                logger.warning("SPL token balance checking not yet implemented")
                return 0.0

        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0

    async def request_airdrop(self, amount_sol: float = 2.0) -> bool:
        """
        Request SOL airdrop from devnet faucet
        Only works on devnet
        """
        try:
            if not self.client or not self.keypair:
                logger.error("Wallet not initialized")
                return False

            logger.info(f"Requesting {amount_sol} SOL airdrop...")

            # Convert SOL to lamports
            lamports = int(amount_sol * 1_000_000_000)

            # Request airdrop
            response = await self.client.request_airdrop(
                self.keypair.pubkey(),
                lamports
            )

            if response.value:
                logger.info(f"Airdrop requested. Signature: {response.value}")

                # Wait for confirmation
                await self.client.confirm_transaction(response.value)

                # Check new balance
                new_balance = await self.get_balance()
                logger.info(f"Airdrop confirmed. New balance: {new_balance} SOL")
                return True
            else:
                logger.error("Airdrop request failed")
                return False

        except Exception as e:
            logger.error(f"Airdrop failed: {e}", exc_info=True)
            return False

    async def sign_transaction(self, transaction: Transaction) -> Optional[Transaction]:
        """Sign transaction with wallet keypair"""
        try:
            if not self.keypair:
                logger.error("Wallet not initialized")
                return None

            # Sign transaction
            transaction.sign(self.keypair)
            return transaction

        except Exception as e:
            logger.error(f"Failed to sign transaction: {e}")
            return None

    async def send_transaction(self, transaction: Transaction) -> Optional[str]:
        """
        Send signed transaction to network
        Returns transaction signature if successful
        """
        try:
            if not self.client:
                logger.error("Client not initialized")
                return None

            # Send transaction
            response = await self.client.send_transaction(transaction)

            if response.value:
                signature = str(response.value)
                logger.info(f"Transaction sent. Signature: {signature}")
                return signature
            else:
                logger.error("Transaction failed to send")
                return None

        except Exception as e:
            logger.error(f"Failed to send transaction: {e}", exc_info=True)
            return None

    async def confirm_transaction(self,
                                  signature: str,
                                  timeout: int = 60) -> bool:
        """
        Confirm transaction with timeout
        """
        try:
            if not self.client:
                logger.error("Client not initialized")
                return False

            logger.info(f"Confirming transaction: {signature}")

            # Confirm transaction
            response = await self.client.confirm_transaction(signature)

            if response.value:
                logger.info(f"Transaction confirmed: {signature}")
                return True
            else:
                logger.warning(f"Transaction not confirmed: {signature}")
                return False

        except Exception as e:
            logger.error(f"Failed to confirm transaction: {e}")
            return False

    async def close(self):
        """Close connections"""
        if self.client:
            await self.client.close()
            logger.info("Wallet connections closed")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


async def test_wallet():
    """Test wallet functionality"""
    print("\n" + "="*70)
    print("TESTING SOLANA WALLET MANAGER")
    print("="*70 + "\n")

    async with SolanaWalletManager() as wallet:
        print(f"Wallet Address: {wallet.get_public_key()}")

        # Check balance
        balance = await wallet.get_balance()
        print(f"Current SOL Balance: {balance}")

        # Request airdrop if balance is low
        if balance < 1.0:
            print("\nRequesting 2 SOL airdrop...")
            success = await wallet.request_airdrop(2.0)
            if success:
                print("Airdrop successful!")
                balance = await wallet.get_balance()
                print(f"New Balance: {balance} SOL")
            else:
                print("Airdrop failed (might be rate limited)")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_wallet())
