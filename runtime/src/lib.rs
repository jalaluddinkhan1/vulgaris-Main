use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use numpy::{PyArray1, PyArray2, IntoPyArray};
use pyo3::exceptions::PyValueError;

mod event_stream;
use event_stream::{EventProcessor, TelemetryEvent, to_matrix, from_json_bytes};

#[pyclass(name = "PyEventProcessor")]
struct PyEventProcessorWrapper {
    inner: EventProcessor,
    n_sensors: u32,
    window_size: usize,
}

#[pymethods]
impl PyEventProcessorWrapper {
    #[new]
    #[pyo3(signature = (n_sensors, window_size, window_step=None, buffer_capacity=65536))]
    fn new(n_sensors: u32, window_size: usize, window_step: Option<usize>,
           buffer_capacity: usize) -> Self {
        let step = window_step.unwrap_or(window_size);
        PyEventProcessorWrapper {
            inner: EventProcessor::new(n_sensors, window_size, step, buffer_capacity),
            n_sensors,
            window_size,
        }
    }

    #[pyo3(signature = (timestamp, sensor_id, value, quality=100))]
    fn push(&mut self, timestamp: f64, sensor_id: u32, value: f32, quality: u8) -> bool {
        let event = TelemetryEvent {
            timestamp,
            sensor_id,
            value,
            quality,
            flags: 0,
        };
        self.inner.push_event(event)
    }

    fn push_json(&mut self, json_str: &str) -> PyResult<bool> {
        let ev = from_json_bytes(json_str.as_bytes())
            .map_err(|e| PyValueError::new_err(format!("JSON parse error: {}", e)))?;
        Ok(self.inner.push_event(ev))
    }

    fn try_get_batch<'py>(&mut self, py: Python<'py>) -> Option<&'py PyArray2<f32>> {
        let batch = self.inner.try_get_batch()?;
        let matrix = to_matrix(&batch.events, self.n_sensors, self.window_size);
        let arr = ndarray::Array2::from_shape_vec(
            (self.n_sensors as usize, self.window_size),
            matrix,
        ).unwrap_or_else(|_| ndarray::Array2::zeros((self.n_sensors as usize, self.window_size)));
        Some(arr.into_pyarray(py))
    }

    fn get_stats<'py>(&self, py: Python<'py>) -> &'py PyDict {
        let stats = self.inner.get_stats();
        let d = PyDict::new(py);
        d.set_item("events_received", stats.events_received).unwrap();
        d.set_item("events_dropped", stats.events_dropped).unwrap();
        d.set_item("batches_emitted", stats.batches_emitted).unwrap();
        d.set_item("avg_latency_us", stats.avg_latency_us).unwrap();
        d.set_item("last_event_time", stats.last_event_time).unwrap();
        d
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[pyclass(name = "PyEventBatch")]
struct PyEventBatchWrapper {
    #[pyo3(get)]
    batch_id: u64,
    #[pyo3(get)]
    timestamp: f64,
    data_flat: Vec<f32>,
    n_sensors: usize,
    window_size: usize,
}

#[pymethods]
impl PyEventBatchWrapper {
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> &'py PyArray2<f32> {
        let arr = ndarray::Array2::from_shape_vec(
            (self.n_sensors, self.window_size),
            self.data_flat.clone(),
        ).unwrap_or_else(|_| ndarray::Array2::zeros((self.n_sensors, self.window_size)));
        arr.into_pyarray(py)
    }
}

/// Parse an array of TelemetryEvent objects from JSON bytes.
#[pyfunction]
fn parse_events_json<'py>(py: Python<'py>, json_bytes: &PyBytes) -> PyResult<&'py PyList> {
    let bytes = json_bytes.as_bytes();
    let events: Vec<TelemetryEvent> = serde_json::from_slice(bytes)
        .map_err(|e| PyValueError::new_err(format!("JSON parse error: {}", e)))?;

    let list = PyList::empty(py);
    for ev in events {
        let d = PyDict::new(py);
        d.set_item("timestamp", ev.timestamp)?;
        d.set_item("sensor_id", ev.sensor_id)?;
        d.set_item("value", ev.value)?;
        d.set_item("quality", ev.quality)?;
        d.set_item("flags", ev.flags)?;
        list.append(d)?;
    }
    Ok(list)
}

#[pymodule]
fn vulgaris_runtime(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyEventProcessorWrapper>()?;
    m.add_class::<PyEventBatchWrapper>()?;
    m.add_function(wrap_pyfunction!(parse_events_json, m)?)?;
    Ok(())
}
