// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]

#[path = "common.rs"]
mod common;

use libfuzzer_sys::fuzz_target;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyList, PyModule};

fn eval_numpy<'py>(
    py: Python<'py>,
    expr: &str,
    locals: &Bound<'py, PyDict>,
) -> Option<Bound<'py, PyAny>> {
    py.eval_bound(expr, None, Some(locals)).ok()
}

fuzz_target!(|data: &[u8]| {
    let mut cursor = common::ByteCursor::new(data);

    let n_seed = cursor.next_u8();
    let d_seed = cursor.next_u8();
    let x_mode = cursor.next_u8();
    let time_mode = cursor.next_u8();
    let detector_seed = cursor.next_u8();
    let cost_seed = cursor.next_u8();
    let pen_seed = cursor.next_u8();
    let step_seed = cursor.next_u8();

    let payload_len = common::bounded(cursor.next_u8(), 0, 256).saturating_mul(8);
    let payload = cursor.take_padded(payload_len);

    Python::with_gil(|py| {
        let Ok(np) = PyModule::import_bound(py, "numpy") else {
            return;
        };
        let locals = PyDict::new_bound(py);
        if locals.set_item("np", &np).is_err() {
            return;
        }

        let n = usize::from(n_seed % 8);
        let d = usize::from(d_seed % 5);
        let flat_len = n.saturating_mul(d).min(64);
        let vals2_len = flat_len.saturating_add(8).max(2).min(common::MAX_VALUE_LEN);
        let step = common::bounded(step_seed, 1, 4);
        let strided_n = (vals2_len + step - 1) / step;

        let mut vals = common::decode_f64_chunks(&payload, flat_len);
        common::ensure_len(&mut vals, flat_len);
        vals.truncate(flat_len);

        let mut vals2 = common::decode_f64_chunks(cursor.remaining(), vals2_len);
        common::ensure_len(&mut vals2, vals2_len);
        vals2.truncate(vals2_len);

        if locals
            .set_item("vals", PyList::new_bound(py, vals))
            .is_err()
        {
            return;
        }
        if locals
            .set_item("vals2", PyList::new_bound(py, vals2))
            .is_err()
        {
            return;
        }
        if locals.set_item("n", n).is_err()
            || locals.set_item("d", d).is_err()
            || locals.set_item("step", step).is_err()
        {
            return;
        }

        let x_expr = match x_mode % 8 {
            0 => "np.array(vals, dtype=np.float64)",
            1 => "np.array(vals, dtype=np.float32)",
            2 => "np.array(vals, dtype=np.int64)",
            3 => "np.array(vals, dtype=np.float64).reshape((n, d), order='C')",
            4 => "np.asfortranarray(np.array(vals, dtype=np.float64).reshape((n, d), order='C'))",
            5 => "np.array(vals2, dtype=np.float64)[::step]",
            6 => "np.array(vals, dtype=np.float64).reshape((1, n, d))",
            _ => "'not-an-array'",
        };

        let Some(x_obj) = eval_numpy(py, x_expr, &locals) else {
            return;
        };

        let x_n = match x_mode % 8 {
            3 | 4 => n,
            5 => strided_n,
            6 => 1,
            _ => flat_len,
        };

        if locals.set_item("time_n", x_n).is_err() {
            return;
        }

        let time_expr = match time_mode % 7 {
            0 => None,
            1 => Some("np.arange(time_n, dtype=np.int64)"),
            2 => Some("np.arange(time_n, dtype='datetime64[ns]')"),
            3 => Some("np.arange(time_n + 1, dtype=np.int64)"),
            4 => Some("np.zeros((time_n, 1), dtype=np.int64)"),
            5 => Some("np.arange(time_n, dtype=np.float64)"),
            _ => Some("'bad-time'"),
        };

        let time_obj = time_expr.and_then(|expr| eval_numpy(py, expr, &locals));

        let detector = match detector_seed % 3 {
            0 => "pelt",
            1 => "binseg",
            _ => "unknown-detector",
        };

        let cost = match cost_seed % 3 {
            0 => "l2",
            1 => "normal",
            _ => "unknown-cost",
        };

        let pen = match pen_seed % 5 {
            0 => None,
            1 => Some(1.0 + f64::from(pen_seed) / 8.0),
            2 => Some(-1.0),
            3 => Some(f64::NAN),
            _ => Some(f64::INFINITY),
        };

        let _ = cpd_python::fuzz_detect_offline_numpy_case(
            py,
            &x_obj,
            time_obj.as_ref(),
            detector,
            cost,
            pen,
        );
    });
});
