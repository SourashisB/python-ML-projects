import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import time
from collections import deque

# Import modules from main.py
from main import MarketSimulator, MarketMaker, Backtester, StrategyOptimizer

class TestMarketSimulator(unittest.TestCase):
    def setUp(self):
        self.simulator = MarketSimulator(initial_price=100, volatility=0.5)
    
    def test_initialization(self):
        self.assertEqual(self.simulator.price, 100)
        self.assertEqual(self.simulator.volatility, 0.5)
        self.assertIsInstance(self.simulator.order_book["bids"], deque)
        self.assertIsInstance(self.simulator.order_book["asks"], deque)
        
    def test_update_price(self):
        original_price = self.simulator.price
        
        # Mock random.normal to return a fixed value
        with patch('numpy.random.normal', return_value=1.0):
            self.simulator.update_price()
            self.assertEqual(self.simulator.price, original_price + 1.0)
    
    def test_update_price_stays_positive(self):
        # Force the price to go negative, should be capped at 1.0
        self.simulator.price = 0.5
        with patch('numpy.random.normal', return_value=-5.0):
            self.simulator.update_price()
            self.assertEqual(self.simulator.price, 1.0)
    
    def test_get_bid_ask(self):
        self.simulator.price = 100.0
        
        # Mock random.uniform to return a fixed spread
        with patch('numpy.random.uniform', return_value=0.05):
            bid, ask = self.simulator.get_bid_ask()
            self.assertEqual(bid, 99.975)
            self.assertEqual(ask, 100.025)
    
    @patch('time.sleep')  # Mock sleep to avoid actual delay
    def test_generate_market_data(self, mock_sleep):
        self.simulator.order_book = {"bids": deque(maxlen=100), "asks": deque(maxlen=100)}
        self.simulator.generate_market_data(duration=1, interval=0.1)
        
        # Should have generated 10 data points
        self.assertEqual(len(self.simulator.order_book["bids"]), 10)
        self.assertEqual(len(self.simulator.order_book["asks"]), 10)
        mock_sleep.assert_called_with(0.1)


class TestMarketMaker(unittest.TestCase):
    def setUp(self):
        self.simulator = MagicMock()
        self.simulator.get_bid_ask.return_value = (99.0, 101.0)
        self.simulator.price = 100.0
        self.market_maker = MarketMaker(self.simulator, spread=0.05, position_limit=10)
    
    def test_initialization(self):
        self.assertEqual(self.market_maker.spread, 0.05)
        self.assertEqual(self.market_maker.position, 0)
        self.assertEqual(self.market_maker.cash, 10000)
        self.assertEqual(self.market_maker.trades, [])
    
    def test_place_orders(self):
        with patch('random.random', side_effect=[0.4, 0.6]):  # First below 0.5, second above 0.5
            with patch.object(self.market_maker, 'execute_trade') as mock_execute:
                self.market_maker.place_orders()
                
                # Should execute buy but not sell
                mock_execute.assert_called_once_with(98.95, "buy")  # 99.0 - 0.05
    
    def test_execute_trade_buy(self):
        # Test buying
        self.market_maker.position = 0
        self.market_maker.cash = 10000
        self.market_maker.execute_trade(100.0, "buy")
        
        self.assertEqual(self.market_maker.position, 1)
        self.assertEqual(self.market_maker.cash, 9900.0)
        self.assertEqual(self.market_maker.trades, [("buy", 100.0)])
    
    def test_execute_trade_sell(self):
        # Test selling
        self.market_maker.position = 0
        self.market_maker.cash = 10000
        self.market_maker.execute_trade(100.0, "sell")
        
        self.assertEqual(self.market_maker.position, -1)
        self.assertEqual(self.market_maker.cash, 10100.0)
        self.assertEqual(self.market_maker.trades, [("sell", 100.0)])
    
    def test_position_limit_buy(self):
        # Test position limit for buying
        self.market_maker.position = 10  # Already at limit
        self.market_maker.cash = 10000
        self.market_maker.execute_trade(100.0, "buy")
        
        # Position should not change
        self.assertEqual(self.market_maker.position, 10)
        self.assertEqual(self.market_maker.cash, 10000)
        self.assertEqual(self.market_maker.trades, [])
    
    def test_position_limit_sell(self):
        # Test position limit for selling
        self.market_maker.position = -10  # Already at limit
        self.market_maker.cash = 10000
        self.market_maker.execute_trade(100.0, "sell")
        
        # Position should not change
        self.assertEqual(self.market_maker.position, -10)
        self.assertEqual(self.market_maker.cash, 10000)
        self.assertEqual(self.market_maker.trades, [])
    
    @patch('time.sleep')  # Mock sleep to avoid actual delay
    def test_run_strategy(self, mock_sleep):
        with patch.object(self.market_maker, 'place_orders') as mock_place_orders:
            self.market_maker.run_strategy(duration=1, interval=0.1)
            
            # Should have called place_orders 10 times
            self.assertEqual(mock_place_orders.call_count, 10)
    
    def test_performance(self):
        # Setup a test scenario
        self.market_maker.cash = 9500
        self.market_maker.position = 5
        self.simulator.price = 110  # Current market price
        
        # Calculate expected PnL
        expected_pnl = 9500 + (5 * 110) - 10000  # Cash + Position Value - Initial Cash
        
        performance = self.market_maker.performance()
        self.assertEqual(performance["PnL"], expected_pnl)
        self.assertEqual(performance["Final Position"], 5)


class TestBacktester(unittest.TestCase):
    def setUp(self):
        self.simulator = MagicMock()
        self.strategy = MagicMock()
        self.strategy.performance.return_value = {"PnL": 500, "Final Position": 5}
        self.backtester = Backtester(self.simulator, self.strategy)
    
    @patch('threading.Thread')
    def test_run(self, mock_thread):
        # Mock the thread objects
        mock_market_thread = MagicMock()
        mock_strategy_thread = MagicMock()
        mock_thread.side_effect = [mock_market_thread, mock_strategy_thread]
        
        result = self.backtester.run(duration=10)
        
        # Verify threads were started and joined
        self.simulator.generate_market_data.assert_called_once_with(10)
        self.strategy.run_strategy.assert_called_once_with(10)
        self.assertEqual(mock_market_thread.start.call_count, 1)
        self.assertEqual(mock_strategy_thread.start.call_count, 1)
        self.assertEqual(mock_market_thread.join.call_count, 1)
        self.assertEqual(mock_strategy_thread.join.call_count, 1)
        
        # Verify the result
        self.assertEqual(result, {"PnL": 500, "Final Position": 5})


class TestStrategyOptimizer(unittest.TestCase):
    def setUp(self):
        self.simulator_class = MagicMock()
        self.strategy_class = MagicMock()
        self.param_grid = {
            "spread": [0.01, 0.05],
            "position_limit": [5, 10]
        }
        self.optimizer = StrategyOptimizer(
            self.simulator_class, 
            self.strategy_class, 
            self.param_grid
        )
    
    def test_initialization(self):
        # Should have 4 parameter combinations
        self.assertEqual(len(self.optimizer.param_grid), 4)
    
    @patch('main.Backtester')
    def test_optimize(self, mock_backtester):
        # Mock the backtester to return different PnLs for different parameters
        mock_backtester_instances = []
        mock_results = [
            {"PnL": 100, "Final Position": 2},  # For params: {"spread": 0.01, "position_limit": 5}
            {"PnL": 200, "Final Position": 3},  # For params: {"spread": 0.01, "position_limit": 10}
            {"PnL": 50, "Final Position": 1},   # For params: {"spread": 0.05, "position_limit": 5}
            {"PnL": 150, "Final Position": 2}   # For params: {"spread": 0.05, "position_limit": 10}
        ]
        
        for result in mock_results:
            mock_instance = MagicMock()
            mock_instance.run.return_value = result
            mock_backtester_instances.append(mock_instance)
        
        mock_backtester.side_effect = mock_backtester_instances
        
        # Run the optimizer
        best_params, best_pnl, all_results = self.optimizer.optimize(duration=5)
        
        # Verify the best parameters (should be the ones with PnL of 200)
        self.assertEqual(best_params, {"spread": 0.01, "position_limit": 10})
        self.assertEqual(best_pnl, 200)
        
        # Verify all results were collected
        self.assertEqual(len(all_results), 4)
        
        # Verify backtester was called with the right arguments
        for i, mock_instance in enumerate(mock_backtester_instances):
            mock_instance.run.assert_called_once_with(5)


if __name__ == '__main__':
    unittest.main()