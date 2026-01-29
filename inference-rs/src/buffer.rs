//! Rolling data buffers for maintaining feature state.
//!
//! Provides efficient circular buffers for storing time-series data
//! with minimal allocations during inference.

use std::collections::VecDeque;

/// A generic rolling buffer with fixed capacity.
///
/// Uses a circular buffer implementation to efficiently maintain
/// a sliding window of data without reallocations.
#[derive(Debug, Clone)]
pub struct RollingBuffer<T> {
    buffer: VecDeque<T>,
    capacity: usize,
}

impl<T> RollingBuffer<T> {
    /// Create a new rolling buffer with the specified capacity.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of elements to store
    ///
    /// # Example
    /// ```
    /// use ash_inference::RollingBuffer;
    /// let buffer: RollingBuffer<f64> = RollingBuffer::new(100);
    /// ```
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Add a new element to the buffer.
    ///
    /// If the buffer is at capacity, removes the oldest element.
    ///
    /// # Arguments
    /// * `value` - The value to add
    pub fn push(&mut self, value: T) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(value);
    }

    /// Get the current number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Check if the buffer is at full capacity.
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }

    /// Get the buffer capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear all elements from the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Get a slice of the most recent `n` elements.
    ///
    /// Returns `None` if there are fewer than `n` elements.
    pub fn last_n(&self, n: usize) -> Option<Vec<&T>> {
        if self.buffer.len() < n {
            return None;
        }
        Some(self.buffer.iter().rev().take(n).rev().collect())
    }

    /// Get an iterator over all elements (oldest to newest).
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.buffer.iter()
    }
}

impl<T: Clone> RollingBuffer<T> {
    /// Get a vector of all elements (oldest to newest).
    pub fn to_vec(&self) -> Vec<T> {
        self.buffer.iter().cloned().collect()
    }
}

/// Feature buffer for maintaining model input state.
///
/// Stores the necessary historical data to compute features
/// for model inference. Pre-allocates buffers to avoid
/// allocations during the hot path.
#[derive(Debug)]
pub struct FeatureBuffer {
    /// Rolling buffer of prices
    pub prices: RollingBuffer<f64>,

    /// Rolling buffer of volumes
    pub volumes: RollingBuffer<f64>,

    /// Rolling buffer of timestamps (seconds since epoch)
    pub timestamps: RollingBuffer<f64>,

    /// Maximum lookback window in seconds
    pub lookback_seconds: usize,
}

impl FeatureBuffer {
    /// Create a new feature buffer.
    ///
    /// # Arguments
    /// * `lookback_seconds` - Maximum lookback window in seconds
    /// * `frequency_hz` - Expected data frequency in Hz (e.g., 1 Hz = 1 sample/sec)
    ///
    /// # Example
    /// ```
    /// use ash_inference::FeatureBuffer;
    /// // 300 second lookback at 1 Hz sampling
    /// let buffer = FeatureBuffer::new(300, 1);
    /// ```
    pub fn new(lookback_seconds: usize, frequency_hz: usize) -> Self {
        let capacity = lookback_seconds * frequency_hz;

        Self {
            prices: RollingBuffer::new(capacity),
            volumes: RollingBuffer::new(capacity),
            timestamps: RollingBuffer::new(capacity),
            lookback_seconds,
        }
    }

    /// Update the buffer with new tick data.
    ///
    /// # Arguments
    /// * `timestamp` - Unix timestamp in seconds
    /// * `price` - Current price
    /// * `volume` - Current volume
    pub fn update(&mut self, timestamp: f64, price: f64, volume: f64) {
        self.timestamps.push(timestamp);
        self.prices.push(price);
        self.volumes.push(volume);
    }

    /// Check if the buffer has sufficient data for inference.
    ///
    /// Returns true if buffer contains at least `min_samples` data points.
    pub fn has_sufficient_data(&self, min_samples: usize) -> bool {
        self.prices.len() >= min_samples
    }

    /// Clear all buffered data.
    pub fn clear(&mut self) {
        self.prices.clear();
        self.volumes.clear();
        self.timestamps.clear();
    }

    /// Get the number of data points currently buffered.
    pub fn len(&self) -> usize {
        self.prices.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.prices.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_buffer_basic() {
        let mut buffer = RollingBuffer::new(3);
        assert!(buffer.is_empty());
        assert_eq!(buffer.capacity(), 3);

        buffer.push(1);
        buffer.push(2);
        buffer.push(3);

        assert!(buffer.is_full());
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.to_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn test_rolling_buffer_overflow() {
        let mut buffer = RollingBuffer::new(3);

        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        buffer.push(4); // Should evict 1

        assert_eq!(buffer.to_vec(), vec![2, 3, 4]);
        assert!(buffer.is_full());
    }

    #[test]
    fn test_rolling_buffer_last_n() {
        let mut buffer = RollingBuffer::new(5);

        buffer.push(1);
        buffer.push(2);
        buffer.push(3);

        let last_2 = buffer.last_n(2).unwrap();
        assert_eq!(last_2, vec![&2, &3]);

        assert!(buffer.last_n(5).is_none()); // Not enough elements
    }

    #[test]
    fn test_rolling_buffer_clear() {
        let mut buffer = RollingBuffer::new(3);
        buffer.push(1);
        buffer.push(2);

        buffer.clear();
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_feature_buffer_basic() {
        let mut buffer = FeatureBuffer::new(300, 1);
        assert!(buffer.is_empty());

        buffer.update(1000.0, 100.5, 1000.0);
        buffer.update(1001.0, 100.6, 1500.0);

        assert_eq!(buffer.len(), 2);
        assert!(buffer.has_sufficient_data(2));
        assert!(!buffer.has_sufficient_data(3));
    }

    #[test]
    fn test_feature_buffer_clear() {
        let mut buffer = FeatureBuffer::new(300, 1);

        buffer.update(1000.0, 100.5, 1000.0);
        buffer.update(1001.0, 100.6, 1500.0);

        buffer.clear();
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_feature_buffer_capacity() {
        // 10 second lookback at 2 Hz = 20 capacity
        let buffer = FeatureBuffer::new(10, 2);
        assert_eq!(buffer.prices.capacity(), 20);
    }
}
