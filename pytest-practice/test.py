import pytest
from project import PriorityQueue, PriorityNode

class TestPriorityNode:
    def test_init(self):
        node = PriorityNode("task", 5)
        assert node.value == "task"
        assert node.priority == 5
        assert node.insertion_order == 0
    
    def test_comparison_lt(self):
        node1 = PriorityNode("task1", 5)
        node2 = PriorityNode("task2", 10)
        assert node1 < node2  # Lower priority value means higher priority
        
        # Test insertion order as tiebreaker
        node3 = PriorityNode("task3", 5)
        node1.insertion_order = 1
        node3.insertion_order = 2
        assert node1 < node3
    
    def test_equality(self):
        node1 = PriorityNode("task", 5)
        node2 = PriorityNode("task", 5)
        node3 = PriorityNode("other", 5)
        node4 = PriorityNode("task", 10)
        
        assert node1 == node2
        assert node1 != node3
        assert node1 != node4
        assert node1 != "not a node"
    
    def test_repr(self):
        node = PriorityNode("task", 5)
        assert repr(node) == "PriorityNode(value=task, priority=5)"


class TestPriorityQueue:
    def test_init(self):
        pq = PriorityQueue()
        assert len(pq) == 0
        assert pq.is_empty()
    
    def test_enqueue_single(self):
        pq = PriorityQueue()
        pq.enqueue("task", 5)
        
        assert len(pq) == 1
        assert not pq.is_empty()
        assert pq.peek() == "task"
    
    def test_enqueue_multiple(self):
        pq = PriorityQueue()
        pq.enqueue("task3", 3)
        pq.enqueue("task1", 1)
        pq.enqueue("task2", 2)
        
        assert len(pq) == 3
        assert pq.peek() == "task1"  # Lowest priority value should be at the top
    
    def test_enqueue_duplicate_value(self):
        pq = PriorityQueue()
        pq.enqueue("task", 5)
        
        with pytest.raises(ValueError) as excinfo:
            pq.enqueue("task", 3)
        assert "already exists" in str(excinfo.value)
    
    def test_dequeue_single(self):
        pq = PriorityQueue()
        pq.enqueue("task", 5)
        
        assert pq.dequeue() == "task"
        assert pq.is_empty()
    
    def test_dequeue_multiple_priority_order(self):
        pq = PriorityQueue()
        pq.enqueue("task3", 3)
        pq.enqueue("task1", 1)
        pq.enqueue("task2", 2)
        
        assert pq.dequeue() == "task1"
        assert pq.dequeue() == "task2"
        assert pq.dequeue() == "task3"
        assert pq.is_empty()
    
    def test_dequeue_empty(self):
        pq = PriorityQueue()
        
        with pytest.raises(IndexError) as excinfo:
            pq.dequeue()
        assert "empty" in str(excinfo.value)
    
    def test_peek_empty(self):
        pq = PriorityQueue()
        
        with pytest.raises(IndexError) as excinfo:
            pq.peek()
        assert "empty" in str(excinfo.value)
    
    def test_contains(self):
        pq = PriorityQueue()
        pq.enqueue("task1", 1)
        pq.enqueue("task2", 2)
        
        assert "task1" in pq
        assert "task2" in pq
        assert "task3" not in pq
    
    def test_equal_priorities(self):
        pq = PriorityQueue()
        pq.enqueue("task1", 5)
        pq.enqueue("task2", 5)
        pq.enqueue("task3", 5)
        
        # Should maintain insertion order for equal priorities
        assert pq.dequeue() == "task1"
        assert pq.dequeue() == "task2"
        assert pq.dequeue() == "task3"
    
    def test_update_priority_increase(self):
        pq = PriorityQueue()
        pq.enqueue("task1", 3)
        pq.enqueue("task2", 2)
        pq.enqueue("task3", 1)
        
        # Decrease priority value (increase importance)
        pq.update_priority("task1", 0)
        
        assert pq.dequeue() == "task1"
    
    def test_update_priority_decrease(self):
        pq = PriorityQueue()
        pq.enqueue("task1", 1)
        pq.enqueue("task2", 2)
        pq.enqueue("task3", 3)
        
        # Increase priority value (decrease importance)
        pq.update_priority("task1", 4)
        
        assert pq.dequeue() == "task2"
        assert pq.dequeue() == "task3"
        assert pq.dequeue() == "task1"
    
    def test_update_priority_nonexistent(self):
        pq = PriorityQueue()
        pq.enqueue("task1", 1)
        
        with pytest.raises(KeyError) as excinfo:
            pq.update_priority("nonexistent", 5)
        assert "not found" in str(excinfo.value)
    
    def test_remove(self):
        pq = PriorityQueue()
        pq.enqueue("task1", 1)
        pq.enqueue("task2", 2)
        pq.enqueue("task3", 3)
        
        pq.remove("task2")
        
        assert len(pq) == 2
        assert "task2" not in pq
        assert pq.dequeue() == "task1"
        assert pq.dequeue() == "task3"
    
    def test_remove_highest_priority(self):
        pq = PriorityQueue()
        pq.enqueue("task1", 1)
        pq.enqueue("task2", 2)
        
        pq.remove("task1")
        
        assert pq.peek() == "task2"
    
    def test_remove_nonexistent(self):
        pq = PriorityQueue()
        pq.enqueue("task", 1)
        
        with pytest.raises(KeyError) as excinfo:
            pq.remove("nonexistent")
        assert "not found" in str(excinfo.value)
    
    def test_clear(self):
        pq = PriorityQueue()
        pq.enqueue("task1", 1)
        pq.enqueue("task2", 2)
        
        pq.clear()
        
        assert pq.is_empty()
        assert len(pq) == 0
    
    def test_get_priority(self):
        pq = PriorityQueue()
        pq.enqueue("task1", 5)
        pq.enqueue("task2", 10)
        
        assert pq.get_priority("task1") == 5
        assert pq.get_priority("task2") == 10
    
    def test_get_priority_nonexistent(self):
        pq = PriorityQueue()
        
        with pytest.raises(KeyError) as excinfo:
            pq.get_priority("nonexistent")
        assert "not found" in str(excinfo.value)
    
    def test_complex_scenario(self):
        """Test a more complex workflow with multiple operations."""
        pq = PriorityQueue()
        
        # Add tasks with priorities
        pq.enqueue("low_priority", 30)
        pq.enqueue("medium_priority", 20)
        pq.enqueue("high_priority", 10)
        
        # Update priorities
        pq.update_priority("low_priority", 5)  # Now highest priority
        
        # Remove a task
        pq.remove("medium_priority")
        
        # Check resulting order
        assert pq.dequeue() == "low_priority"
        assert pq.dequeue() == "high_priority"
        assert pq.is_empty()
    
    def test_stress_many_items(self):
        """Test with a larger number of items."""
        pq = PriorityQueue()
        
        # Insert 1000 items in reverse priority order
        for i in range(1000, 0, -1):
            pq.enqueue(f"task{i}", i)
        
        # Verify they come out in correct order
        for i in range(1, 1001):
            assert pq.dequeue() == f"task{i}"
    
    def test_mixed_priorities(self):
        """Test with a mix of operations and priorities."""
        pq = PriorityQueue()
        
        # Add some initial items
        pq.enqueue("A", 5)
        pq.enqueue("B", 3)
        pq.enqueue("C", 7)
        
        # Check current highest priority
        assert pq.peek() == "B"
        
        # Update priorities
        pq.update_priority("A", 1)
        assert pq.peek() == "A"
        
        # Add more items
        pq.enqueue("D", 2)
        pq.enqueue("E", 6)
        
        # Remove an item
        pq.remove("C")
        
        # Verify final queue state through dequeuing
        assert pq.dequeue() == "A"
        assert pq.dequeue() == "D"
        assert pq.dequeue() == "B"
        assert pq.dequeue() == "E"
        assert pq.is_empty()