@echo off
REM Start Trading System in Devnet Mode
REM Executes REAL blockchain transactions on Solana Devnet (test tokens only)

echo.
echo ======================================================================
echo STARTING SHIVX TRADING SYSTEM - DEVNET MODE
echo ======================================================================
echo.
echo This will execute REAL blockchain transactions on Solana Devnet
echo using test tokens (worthless, but real transactions).
echo.
echo Configuration:
echo   - Mode: DEVNET
echo   - Network: Solana Devnet
echo   - Max Position: $10 per trade
echo   - Tokens: Test SOL, USDC, USDT
echo.
echo ======================================================================
echo.

REM Set environment for devnet mode
set TRADING_MODE=devnet
set SOLANA_RPC_URL=https://api.devnet.solana.com
set DEVNET_WALLET_PATH=data/wallets/devnet_trading_wallet.json
set MAX_POSITION_SIZE_USD=10
set MAX_SLIPPAGE_BPS=100

echo Loading devnet configuration...
if exist .env.devnet (
    echo Using .env.devnet configuration
    copy /Y .env.devnet .env
) else (
    echo Warning: .env.devnet not found, using environment variables
)

echo.
echo Starting trading system...
echo.

REM Start the trading system
python start_advanced_trading.py

echo.
echo ======================================================================
echo DEVNET TRADING STOPPED
echo ======================================================================
echo.

pause
