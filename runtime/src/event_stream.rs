use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use crossbeam_channel::{bounded, Receiver, Sender, TryRecvError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    pub timestamp: f64,
    pub sensor_id: u32,
    pub value: f32,
    pub quality: u8,
    pub flags: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventBatch {
    pub events: Vec<TelemetryEvent>,
    pub batch_id: u64,
    pub received_at: f64,
}

pub struct EventBuffer {
    inner: VecDeque<TelemetryEvent>,
    capacity: usize,
    dropped: u64,
}

impl EventBuffer {
    pub fn new(capacity: usize) -> Self {
        EventBuffer {
            inner: VecDeque::with_capacity(capacity),
            capacity,
            dropped: 0,
        }
    }

    pub fn push(&mut self, event: TelemetryEvent) -> bool {
        if self.inner.len() >= self.capacity {
            self.dropped += 1;
            return false;
        }
        self.inner.push_back(event);
        true
    }

    pub fn drain_window(&mut self, window_size: usize) -> Vec<TelemetryEvent> {
        let n = window_size.min(self.inner.len());
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            if let Some(e) = self.inner.pop_front() {
                result.push(e);
            }
        }
        result
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn dropped_count(&self) -> u64 {
        self.dropped
    }

    /// Resample events into a flat (n_sensors * window_size) float32 array.
    /// Missing sensor/time slots are filled via linear interpolation.
    pub fn as_matrix(&self, n_sensors: u32, window_size: usize) -> Vec<f32> {
        let n_sensors = n_sensors as usize;
        let mut matrix = vec![0.0f32; n_sensors * window_size];

        if self.inner.is_empty() || window_size == 0 {
            return matrix;
        }

        // Determine time range from available events
        let events: Vec<&TelemetryEvent> = self.inner.iter().collect();
        let t_start = events.iter().map(|e| e.timestamp).fold(f64::INFINITY, f64::min);
        let t_end = events.iter().map(|e| e.timestamp).fold(f64::NEG_INFINITY, f64::max);
        let t_range = if (t_end - t_start).abs() < 1e-12 { 1.0 } else { t_end - t_start };

        // For each sensor, collect (time_index, value) pairs and interpolate
        let mut sensor_data: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_sensors];
        for ev in &events {
            let sid = ev.sensor_id as usize;
            if sid >= n_sensors {
                continue;
            }
            if ev.quality < 50 {
                continue; // skip bad quality
            }
            let frac = (ev.timestamp - t_start) / t_range;
            let t_idx = ((frac * (window_size - 1) as f64).round() as usize).min(window_size - 1);
            sensor_data[sid].push((t_idx, ev.value));
        }

        for s in 0..n_sensors {
            let row_offset = s * window_size;
            let pts = &sensor_data[s];
            if pts.is_empty() {
                continue;
            }
            // Sort by time index
            let mut sorted_pts = pts.clone();
            sorted_pts.sort_by_key(|p| p.0);

            // Fill directly known points
            for &(ti, val) in &sorted_pts {
                matrix[row_offset + ti] = val;
            }

            // Linear interpolation between known points
            if sorted_pts.len() >= 2 {
                for i in 0..sorted_pts.len() - 1 {
                    let (t0, v0) = sorted_pts[i];
                    let (t1, v1) = sorted_pts[i + 1];
                    if t1 > t0 + 1 {
                        let span = (t1 - t0) as f32;
                        for t in (t0 + 1)..t1 {
                            let alpha = (t - t0) as f32 / span;
                            matrix[row_offset + t] = v0 * (1.0 - alpha) + v1 * alpha;
                        }
                    }
                }
                // Forward-fill before first point and after last point
                let (first_t, first_v) = sorted_pts[0];
                for t in 0..first_t {
                    matrix[row_offset + t] = first_v;
                }
                let (last_t, last_v) = *sorted_pts.last().unwrap();
                for t in (last_t + 1)..window_size {
                    matrix[row_offset + t] = last_v;
                }
            }
        }

        matrix
    }
}

#[derive(Debug, Default, Clone)]
pub struct ProcessorStats {
    pub events_received: u64,
    pub events_dropped: u64,
    pub batches_emitted: u64,
    pub avg_latency_us: f64,
    pub last_event_time: f64,
}

pub struct EventProcessor {
    buffer: Arc<Mutex<EventBuffer>>,
    tx: Sender<EventBatch>,
    rx: Receiver<EventBatch>,
    n_sensors: u32,
    window_size: usize,
    window_step: usize,
    batch_counter: u64,
    pub stats: ProcessorStats,
    last_emit_time: Option<Instant>,
    // Track how many events have been accumulated since last emit
    events_since_emit: usize,
}

impl EventProcessor {
    pub fn new(n_sensors: u32, window_size: usize, window_step: usize,
               buffer_capacity: usize) -> Self {
        let (tx, rx) = bounded(256);
        let step = if window_step == 0 { window_size } else { window_step };
        EventProcessor {
            buffer: Arc::new(Mutex::new(EventBuffer::new(buffer_capacity))),
            tx,
            rx,
            n_sensors,
            window_size,
            window_step: step,
            batch_counter: 0,
            stats: ProcessorStats::default(),
            last_emit_time: None,
            events_since_emit: 0,
        }
    }

    pub fn push_event(&mut self, event: TelemetryEvent) -> bool {
        let ts = event.timestamp;
        let accepted = {
            let mut buf = self.buffer.lock().unwrap();
            buf.push(event)
        };
        if accepted {
            self.stats.events_received += 1;
            self.stats.last_event_time = ts;
            self.events_since_emit += 1;
        } else {
            self.stats.events_dropped += 1;
        }

        // Attempt to emit a batch when enough events have accumulated
        if self.events_since_emit >= self.window_step {
            let buf_len = self.buffer.lock().unwrap().len();
            if buf_len >= self.window_size {
                self._try_emit();
            }
        }
        accepted
    }

    fn _try_emit(&mut self) {
        let now = Instant::now();
        let received_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        let events = {
            let mut buf = self.buffer.lock().unwrap();
            // For sliding window: peek without removing (we step by window_step)
            let all: Vec<TelemetryEvent> = buf.inner.iter().take(self.window_size).cloned().collect();
            // Advance by window_step
            for _ in 0..self.window_step.min(buf.inner.len()) {
                buf.inner.pop_front();
            }
            all
        };

        if events.len() < self.window_size {
            return;
        }

        let batch = EventBatch {
            events,
            batch_id: self.batch_counter,
            received_at,
        };
        self.batch_counter += 1;
        self.events_since_emit = 0;

        if let Some(last) = self.last_emit_time {
            let latency_us = now.duration_since(last).as_micros() as f64;
            let n = self.stats.batches_emitted as f64;
            self.stats.avg_latency_us =
                (self.stats.avg_latency_us * n + latency_us) / (n + 1.0);
        }
        self.last_emit_time = Some(now);
        self.stats.batches_emitted += 1;

        // Non-blocking send; drop batch if channel full
        let _ = self.tx.try_send(batch);
    }

    pub fn try_get_batch(&mut self) -> Option<EventBatch> {
        // Also trigger emit if buffer is large enough
        let buf_len = self.buffer.lock().unwrap().len();
        if buf_len >= self.window_size && self.events_since_emit >= self.window_step {
            self._try_emit();
        }
        match self.rx.try_recv() {
            Ok(batch) => Some(batch),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => None,
        }
    }

    pub fn get_stats(&self) -> ProcessorStats {
        let dropped = {
            let buf = self.buffer.lock().unwrap();
            buf.dropped_count()
        };
        let mut s = self.stats.clone();
        s.events_dropped += dropped;
        s
    }

    pub fn reset(&mut self) {
        let mut buf = self.buffer.lock().unwrap();
        buf.inner.clear();
        buf.dropped = 0;
        drop(buf);
        self.stats = ProcessorStats::default();
        self.batch_counter = 0;
        self.events_since_emit = 0;
        self.last_emit_time = None;
        // Drain channel
        while self.rx.try_recv().is_ok() {}
    }
}

pub fn from_json_bytes(bytes: &[u8]) -> Result<TelemetryEvent, String> {
    serde_json::from_slice(bytes).map_err(|e| e.to_string())
}

pub fn to_matrix(events: &[TelemetryEvent], n_sensors: u32, window_size: usize) -> Vec<f32> {
    let n_sensors_usize = n_sensors as usize;
    let mut matrix = vec![0.0f32; n_sensors_usize * window_size];

    if events.is_empty() || window_size == 0 {
        return matrix;
    }

    let t_start = events.iter().map(|e| e.timestamp).fold(f64::INFINITY, f64::min);
    let t_end = events.iter().map(|e| e.timestamp).fold(f64::NEG_INFINITY, f64::max);
    let t_range = if (t_end - t_start).abs() < 1e-12 { 1.0 } else { t_end - t_start };

    let mut sensor_data: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_sensors_usize];
    for ev in events {
        let sid = ev.sensor_id as usize;
        if sid >= n_sensors_usize || ev.quality < 50 {
            continue;
        }
        let frac = (ev.timestamp - t_start) / t_range;
        let t_idx = ((frac * (window_size - 1) as f64).round() as usize).min(window_size - 1);
        sensor_data[sid].push((t_idx, ev.value));
    }

    for s in 0..n_sensors_usize {
        let row_offset = s * window_size;
        let pts = &sensor_data[s];
        if pts.is_empty() {
            continue;
        }
        let mut sorted_pts = pts.clone();
        sorted_pts.sort_by_key(|p| p.0);

        for &(ti, val) in &sorted_pts {
            matrix[row_offset + ti] = val;
        }

        if sorted_pts.len() >= 2 {
            for i in 0..sorted_pts.len() - 1 {
                let (t0, v0) = sorted_pts[i];
                let (t1, v1) = sorted_pts[i + 1];
                if t1 > t0 + 1 {
                    let span = (t1 - t0) as f32;
                    for t in (t0 + 1)..t1 {
                        let alpha = (t - t0) as f32 / span;
                        matrix[row_offset + t] = v0 * (1.0 - alpha) + v1 * alpha;
                    }
                }
            }
            let (first_t, first_v) = sorted_pts[0];
            for t in 0..first_t {
                matrix[row_offset + t] = first_v;
            }
            let (last_t, last_v) = *sorted_pts.last().unwrap();
            for t in (last_t + 1)..window_size {
                matrix[row_offset + t] = last_v;
            }
        }
    }

    matrix
}
