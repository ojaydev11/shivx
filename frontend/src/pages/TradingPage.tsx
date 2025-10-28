// ============================================================================
// ShivX Frontend - Trading Page
// ============================================================================

import { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  LinearProgress,
} from '@mui/material';
import { Add, TrendingUp, TrendingDown } from '@mui/icons-material';
import { api, handleAPIError } from '@services/api';
import { useStore } from '@store/index';
import { toast } from 'react-hot-toast';

export default function TradingPage() {
  const { positions, setPositions } = useStore();
  const [loading, setLoading] = useState(true);
  const [trades, setTrades] = useState([]);
  const [tradeDialogOpen, setTradeDialogOpen] = useState(false);
  const [tradeForm, setTradeForm] = useState({
    symbol: '',
    side: 'buy' as 'buy' | 'sell',
    quantity: '',
    price: '',
  });

  const loadData = async () => {
    try {
      setLoading(true);
      const [positionsData, tradesData] = await Promise.all([
        api.getPositions(),
        api.getTrades(),
      ]);
      setPositions(positionsData);
      setTrades(tradesData);
    } catch (error) {
      handleAPIError(error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleExecuteTrade = async () => {
    try {
      await api.executeTrade({
        symbol: tradeForm.symbol,
        side: tradeForm.side,
        quantity: parseFloat(tradeForm.quantity),
        price: tradeForm.price ? parseFloat(tradeForm.price) : undefined,
      });
      toast.success('Trade executed successfully');
      setTradeDialogOpen(false);
      setTradeForm({ symbol: '', side: 'buy', quantity: '', price: '' });
      loadData();
    } catch (error) {
      handleAPIError(error);
    }
  };

  const totalPnL = positions.reduce((sum, pos) => sum + pos.unrealizedPnL, 0);

  return (
    <Box>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1">
          Trading
        </Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setTradeDialogOpen(true)}
        >
          New Trade
        </Button>
      </Box>

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Total P&L
              </Typography>
              <Typography
                variant="h4"
                color={totalPnL >= 0 ? 'success.main' : 'error.main'}
                sx={{ display: 'flex', alignItems: 'center' }}
              >
                {totalPnL >= 0 ? <TrendingUp /> : <TrendingDown />}
                ${totalPnL.toFixed(2)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Open Positions
              </Typography>
              <Typography variant="h4">{positions.length}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Total Trades
              </Typography>
              <Typography variant="h4">{trades.length}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Win Rate
              </Typography>
              <Typography variant="h4">65.4%</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Open Positions */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Open Positions
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Symbol</TableCell>
                  <TableCell>Side</TableCell>
                  <TableCell align="right">Quantity</TableCell>
                  <TableCell align="right">Entry Price</TableCell>
                  <TableCell align="right">Current Price</TableCell>
                  <TableCell align="right">P&L</TableCell>
                  <TableCell>Opened</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {positions.map((position) => (
                  <TableRow key={position.id}>
                    <TableCell>{position.symbol}</TableCell>
                    <TableCell>
                      <Chip
                        label={position.side}
                        size="small"
                        color={position.side === 'long' ? 'success' : 'error'}
                      />
                    </TableCell>
                    <TableCell align="right">{position.quantity}</TableCell>
                    <TableCell align="right">${position.entryPrice.toFixed(2)}</TableCell>
                    <TableCell align="right">${position.currentPrice.toFixed(2)}</TableCell>
                    <TableCell
                      align="right"
                      sx={{ color: position.unrealizedPnL >= 0 ? 'success.main' : 'error.main' }}
                    >
                      ${position.unrealizedPnL.toFixed(2)}
                    </TableCell>
                    <TableCell>{new Date(position.openedAt).toLocaleString()}</TableCell>
                  </TableRow>
                ))}
                {positions.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={7} align="center">
                      No open positions
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* New Trade Dialog */}
      <Dialog open={tradeDialogOpen} onClose={() => setTradeDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Execute Trade</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Symbol"
            value={tradeForm.symbol}
            onChange={(e) => setTradeForm({ ...tradeForm, symbol: e.target.value.toUpperCase() })}
            margin="normal"
            placeholder="BTC, ETH, SOL..."
            required
          />
          <FormControl fullWidth margin="normal">
            <InputLabel>Side</InputLabel>
            <Select
              value={tradeForm.side}
              label="Side"
              onChange={(e) => setTradeForm({ ...tradeForm, side: e.target.value as 'buy' | 'sell' })}
            >
              <MenuItem value="buy">Buy</MenuItem>
              <MenuItem value="sell">Sell</MenuItem>
            </Select>
          </FormControl>
          <TextField
            fullWidth
            label="Quantity"
            type="number"
            value={tradeForm.quantity}
            onChange={(e) => setTradeForm({ ...tradeForm, quantity: e.target.value })}
            margin="normal"
            required
          />
          <TextField
            fullWidth
            label="Price (optional, leave empty for market price)"
            type="number"
            value={tradeForm.price}
            onChange={(e) => setTradeForm({ ...tradeForm, price: e.target.value })}
            margin="normal"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTradeDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleExecuteTrade}
            variant="contained"
            disabled={!tradeForm.symbol || !tradeForm.quantity}
          >
            Execute
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
