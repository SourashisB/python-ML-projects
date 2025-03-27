class PriorityNode:
    """Node class for priority queue elements."""
    
    def __init__(self, value, priority):
        """Initialize a node with value and priority."""
        self.value = value
        self.priority = priority
        self.insertion_order = 0  # Used as tie-breaker for equal priorities
    
    def __lt__(self, other):
        """Compare nodes based on priority, then insertion order."""
        if self.priority == other.priority:
            return self.insertion_order < other.insertion_order
        return self.priority < other.priority
    
    def __eq__(self, other):
        """Check if nodes are equal based on priority and value."""
        if not isinstance(other, PriorityNode):
            return False
        return (self.priority == other.priority and 
                self.value == other.value)
    
    def __repr__(self):
        """String representation of the node."""
        return f"PriorityNode(value={self.value}, priority={self.priority})"


class PriorityQueue:
    """
    A priority queue implementation using a binary heap.
    Lower priority values indicate higher priority.
    """
    
    def __init__(self):
        """Initialize an empty priority queue."""
        self._heap = []
        self._insertion_count = 0
        self._value_index_map = {}  # For O(1) lookups and updates
    
    def __len__(self):
        """Return the number of elements in the queue."""
        return len(self._heap)
    
    def __contains__(self, value):
        """Check if value exists in the queue."""
        return value in self._value_index_map
    
    def is_empty(self):
        """Check if the queue is empty."""
        return len(self._heap) == 0
    
    def peek(self):
        """Return the highest priority element without removing it."""
        if self.is_empty():
            raise IndexError("Cannot peek: priority queue is empty")
        return self._heap[0].value
    
    def _swap(self, i, j):
        """Swap two elements in the heap and update their indices."""
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
        self._value_index_map[self._heap[i].value] = i
        self._value_index_map[self._heap[j].value] = j
    
    def _sift_up(self, index):
        """Move an element up the heap to its correct position."""
        parent = (index - 1) // 2
        
        if index > 0 and self._heap[index] < self._heap[parent]:
            self._swap(index, parent)
            self._sift_up(parent)
    
    def _sift_down(self, index):
        """Move an element down the heap to its correct position."""
        smallest = index
        left = 2 * index + 1
        right = 2 * index + 2
        
        if (left < len(self._heap) and 
            self._heap[left] < self._heap[smallest]):
            smallest = left
            
        if (right < len(self._heap) and 
            self._heap[right] < self._heap[smallest]):
            smallest = right
            
        if smallest != index:
            self._swap(index, smallest)
            self._sift_down(smallest)
    
    def enqueue(self, value, priority):
        """
        Add an element to the priority queue.
        
        Args:
            value: The value to store
            priority: The priority level (lower is higher priority)
            
        Raises:
            ValueError: If the value already exists in the queue
        """
        if value in self._value_index_map:
            raise ValueError(f"Value '{value}' already exists in the queue")
            
        node = PriorityNode(value, priority)
        node.insertion_order = self._insertion_count
        self._insertion_count += 1
        
        self._heap.append(node)
        index = len(self._heap) - 1
        self._value_index_map[value] = index
        
        self._sift_up(index)
    
    def dequeue(self):
        """
        Remove and return the highest priority element.
        
        Returns:
            The value with the highest priority
            
        Raises:
            IndexError: If the queue is empty
        """
        if self.is_empty():
            raise IndexError("Cannot dequeue: priority queue is empty")
            
        result = self._heap[0].value
        
        # Get the last element
        last_item = self._heap.pop()
        
        if self._heap:  # If the queue isn't empty after popping
            # Move last item to the front
            self._heap[0] = last_item
            self._value_index_map[last_item.value] = 0
            
            # And sift it down to correct position
            self._sift_down(0)
        
        # Remove the dequeued item from our map
        del self._value_index_map[result]
        
        return result
    
    def update_priority(self, value, new_priority):
        """
        Update the priority of an existing element.
        
        Args:
            value: The value whose priority to update
            new_priority: The new priority value
            
        Raises:
            KeyError: If the value doesn't exist in the queue
        """
        if value not in self._value_index_map:
            raise KeyError(f"Value '{value}' not found in the queue")
            
        index = self._value_index_map[value]
        old_priority = self._heap[index].priority
        self._heap[index].priority = new_priority
        
        # If the priority decreased (higher importance), sift up
        if new_priority < old_priority:
            self._sift_up(index)
        # If the priority increased (lower importance), sift down
        elif new_priority > old_priority:
            self._sift_down(index)
    
    def remove(self, value):
        """
        Remove a specific element from the queue.
        
        Args:
            value: The value to remove
            
        Raises:
            KeyError: If the value doesn't exist in the queue
        """
        if value not in self._value_index_map:
            raise KeyError(f"Value '{value}' not found in the queue")
            
        index = self._value_index_map[value]
        
        # If it's the last element, just remove it
        if index == len(self._heap) - 1:
            self._heap.pop()
            del self._value_index_map[value]
            return
        
        # Otherwise swap with the last element and remove
        self._swap(index, len(self._heap) - 1)
        self._heap.pop()
        del self._value_index_map[value]
        
        # If the queue isn't empty after removal, fix the heap
        if index < len(self._heap):
            old_priority = self._heap[index].priority
            
            # Try sift up and sift down to ensure heap property
            self._sift_up(index)
            
            # If the element hasn't moved up, try sifting down
            if self._value_index_map[self._heap[index].value] == index:
                self._sift_down(index)
    
    def clear(self):
        """Remove all elements from the queue."""
        self._heap = []
        self._value_index_map = {}
        self._insertion_count = 0

    def get_priority(self, value):
        """
        Get the priority of a specific value.
        
        Args:
            value: The value to lookup
            
        Returns:
            The priority of the value
            
        Raises:
            KeyError: If the value doesn't exist in the queue
        """
        if value not in self._value_index_map:
            raise KeyError(f"Value '{value}' not found in the queue")
            
        index = self._value_index_map[value]
        return self._heap[index].priority