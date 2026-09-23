//! Adaptive sampling for plotted curves.
//!
//! The browser calls this through WebAssembly so a high plot-point count and
//! a deep recursion stay off the JavaScript thread. `cargo test -p plotter`
//! covers the sampler. Rebuild the page copy with:
//! `cargo build -p plotter --target wasm32-unknown-unknown --release`
//! and copy `target/wasm32-unknown-unknown/release/plotter.wasm` to `web/public/`.

use serde::Deserialize;
use std::cell::RefCell;

#[derive(Deserialize)]
struct Request {
    min: f64,
    max: f64,
    points: u32,
    recursion: u32,
    exclusions: bool,
    yspan: f64,
    degrees: bool,
    expr: Expr,
}

#[derive(Deserialize, Clone)]
#[serde(tag = "t")]
enum Expr {
    #[serde(rename = "rat")]
    Rat { n: f64, d: f64 },
    #[serde(rename = "dec")]
    Dec { v: Option<f64> },
    #[serde(rename = "sym")]
    Sym { name: String },
    #[serde(rename = "add")]
    Add { args: Vec<Expr> },
    #[serde(rename = "mul")]
    Mul { args: Vec<Expr> },
    #[serde(rename = "div")]
    Div { num: Box<Expr>, den: Box<Expr> },
    #[serde(rename = "pow")]
    Pow { base: Box<Expr>, exp: Box<Expr> },
    #[serde(rename = "call")]
    Call { name: String, args: Vec<Expr> },
    #[serde(rename = "group")]
    Group { body: Box<Expr> },
}

#[derive(Clone)]
struct Node {
    t: f64,
    y: Option<f64>,
    cut: bool,
}

const MAX_NODES: usize = 16000;

fn eval_expr(expr: &Expr, t: f64, degrees: bool) -> Option<f64> {
    let value = match expr {
        Expr::Rat { n, d } => {
            if *d == 0.0 {
                return None;
            }
            n / d
        }
        Expr::Dec { v } => (*v)?,
        Expr::Sym { name } => match name.as_str() {
            "x" | "t" | "theta" | "y" => t,
            "pi" => std::f64::consts::PI,
            "e" => std::f64::consts::E,
            _ => t,
        },
        Expr::Add { args } => args
            .iter()
            .try_fold(0.0, |sum, arg| Some(sum + eval_expr(arg, t, degrees)?))?,
        Expr::Mul { args } => args.iter().try_fold(1.0, |product, arg| {
            Some(product * eval_expr(arg, t, degrees)?)
        })?,
        Expr::Div { num, den } => {
            let den = eval_expr(den, t, degrees)?;
            if den == 0.0 || !den.is_finite() {
                return None;
            }
            eval_expr(num, t, degrees)? / den
        }
        Expr::Pow { base, exp } => {
            let base = eval_expr(base, t, degrees)?;
            let exp = eval_expr(exp, t, degrees)?;
            if base < 0.0 && exp.fract() != 0.0 {
                return None;
            }
            base.powf(exp)
        }
        Expr::Call { name, args } => {
            let mut values = Vec::with_capacity(args.len());
            for arg in args {
                values.push(eval_expr(arg, t, degrees)?);
            }
            call_number(name, &values, degrees)?
        }
        Expr::Group { body } => eval_expr(body, t, degrees)?,
    };
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

fn call_number(name: &str, args: &[f64], degrees: bool) -> Option<f64> {
    let x = *args.first()?;
    let to_rad = |n: f64| {
        if degrees {
            n * std::f64::consts::PI / 180.0
        } else {
            n
        }
    };
    let from_rad = |n: f64| {
        if degrees {
            n * 180.0 / std::f64::consts::PI
        } else {
            n
        }
    };
    let value = match name {
        "sqrt" if args.len() == 1 => x.sqrt(),
        "ln" if args.len() == 1 => x.ln(),
        "log" if args.len() == 1 => x.log10(),
        "log" if args.len() == 2 && args[1] != 0.0 => x.ln() / args[1].ln(),
        "sin" if args.len() == 1 => to_rad(x).sin(),
        "cos" if args.len() == 1 => to_rad(x).cos(),
        "tan" if args.len() == 1 => to_rad(x).tan(),
        "abs" if args.len() == 1 => x.abs(),
        "exp" if args.len() == 1 => x.exp(),
        "asin" if args.len() == 1 => from_rad(x.asin()),
        "acos" if args.len() == 1 => from_rad(x.acos()),
        "atan" if args.len() == 1 => from_rad(x.atan()),
        _ => return None,
    };
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

fn sample_at(index: u32, count: u32, min: f64, max: f64) -> f64 {
    if count <= 1 {
        return (min + max) / 2.0;
    }
    min + (max - min) * f64::from(index) / f64::from(count - 1)
}

fn should_split(left: Option<f64>, mid: Option<f64>, right: Option<f64>, y_span: f64) -> bool {
    let (Some(left), Some(mid), Some(right)) = (left, mid, right) else {
        return true;
    };
    let bend = (mid - (left + right) / 2.0).abs();
    let spike = mid.abs() > left.abs().max(right.abs()) * 3.0 + y_span * 0.35;
    bend > (y_span * 0.012).max(1e-4) || spike
}

fn is_discontinuity(left: Option<f64>, mid: Option<f64>, right: Option<f64>, y_span: f64) -> bool {
    let (Some(left), Some(mid), Some(right)) = (left, mid, right) else {
        return true;
    };
    // A value between the two ends is a steep but continuous climb. A pole sits
    // beyond both of them, or the sample in the middle is not a real number.
    let between = (mid - left) * (mid - right) <= 0.0;
    if between {
        return false;
    }
    let peak = left.abs().max(right.abs());
    mid.abs() > peak * 1.5 + y_span * 0.25
}

/// Sample one explicit curve. The parameter inside the expression is whatever symbol it uses;
/// every symbol other than `pi` and `e` is the sample variable.
fn sample_curve(request: &Request) -> Vec<Node> {
    let count = request.points.clamp(12, 800);
    let recursion = request.recursion.min(8);
    let span = (request.max - request.min).abs().max(1e-12);
    let y_span = request.yspan.max(1e-6);
    let at = |t: f64| eval_expr(&request.expr, t, request.degrees);
    let mut times = Vec::with_capacity(count as usize + 1);
    for index in 0..count {
        times.push(sample_at(index, count, request.min, request.max));
    }
    if request.min < 0.0 && request.max > 0.0 {
        if let Some((index, _)) = times.iter().enumerate().min_by(|left, right| {
            left.1
                .abs()
                .partial_cmp(&right.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            if times[index].abs() > 1e-12 {
                times[index] = 0.0;
                times.sort_by(|left, right| {
                    left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
    }
    let mut nodes = Vec::with_capacity(times.len());
    for t in times {
        nodes.push(Node {
            t,
            y: at(t),
            cut: false,
        });
    }
    for level in 0..=recursion {
        let last = level == recursion;
        let mut next = Vec::with_capacity(nodes.len() * 2);
        let mut grew = false;
        for pair in nodes.windows(2) {
            let left = &pair[0];
            let right = &pair[1];
            next.push(Node {
                t: left.t,
                y: left.y,
                cut: false,
            });
            let mid_t = (left.t + right.t) / 2.0;
            let too_fine = right.t - left.t < span / 10000.0;
            let mid_y = at(mid_t);
            if !last
                && !too_fine
                && nodes.len() < MAX_NODES
                && should_split(left.y, mid_y, right.y, y_span)
            {
                next.push(Node {
                    t: mid_t,
                    y: mid_y,
                    cut: false,
                });
                grew = true;
            } else if request.exclusions && is_discontinuity(left.y, mid_y, right.y, y_span) {
                if let Some(node) = next.last_mut() {
                    node.cut = true;
                }
            }
        }
        if let Some(tail) = nodes.last() {
            next.push(Node {
                t: tail.t,
                y: tail.y,
                cut: false,
            });
        }
        nodes = next;
        if !grew {
            break;
        }
    }
    if request.exclusions {
        mark_spikes(&mut nodes, y_span);
    }
    nodes
}

/// A pole that lands on a sample looks like one enormous point between two smaller ones.
fn mark_spikes(nodes: &mut [Node], y_span: f64) {
    if nodes.len() < 3 {
        return;
    }
    for index in 1..nodes.len() - 1 {
        let (Some(left), Some(mid), Some(right)) =
            (nodes[index - 1].y, nodes[index].y, nodes[index + 1].y)
        else {
            continue;
        };
        let peak = left.abs().max(right.abs());
        if mid.abs() > peak * 3.0 + y_span && mid.abs() > y_span * 2.0 {
            nodes[index - 1].cut = true;
            nodes[index].cut = true;
        }
    }
}

fn sample_json(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let request: Request = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    let nodes = sample_curve(&request);
    let payload: Vec<(f64, Option<f64>, bool)> = nodes
        .into_iter()
        .map(|node| (node.t, node.y, node.cut))
        .collect();
    serde_json::to_vec(&payload).map_err(|error| error.to_string())
}

thread_local! {
    static INPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static OUTPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Resize the input buffer and return its pointer. The next `plot_sample` reads `len` bytes from it.
#[no_mangle]
pub extern "C" fn plot_input_ptr(len: i32) -> *mut u8 {
    if len <= 0 {
        return std::ptr::null_mut();
    }
    INPUT.with(|cell| {
        let mut buffer = cell.borrow_mut();
        buffer.resize(len as usize, 0);
        buffer.as_mut_ptr()
    })
}

/// Sample the curve described by the JSON at `ptr`. Returns the output length, or -1.
///
/// # Safety
///
/// `ptr` must point to `len` readable bytes, such as the buffer returned by `plot_input_ptr(len)`.
#[no_mangle]
pub unsafe extern "C" fn plot_sample(ptr: *const u8, len: i32) -> i32 {
    if ptr.is_null() || len < 0 {
        return -1;
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    let output = sample_json(bytes).unwrap_or_else(|error| {
        serde_json::to_vec(&serde_json::json!({ "error": error }))
            .unwrap_or_else(|_| b"{\"error\":\"plot\"}".to_vec())
    });
    let size = output.len() as i32;
    OUTPUT.with(|cell| *cell.borrow_mut() = output);
    size
}

/// Pointer to the latest sample JSON. Valid until the next `plot_sample`.
#[no_mangle]
pub extern "C" fn plot_output_ptr() -> *const u8 {
    OUTPUT.with(|cell| cell.borrow().as_ptr())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secant() -> Expr {
        Expr::Div {
            num: Box::new(Expr::Rat { n: 1.0, d: 1.0 }),
            den: Box::new(Expr::Pow {
                base: Box::new(Expr::Call {
                    name: "cos".into(),
                    args: vec![Expr::Sym { name: "x".into() }],
                }),
                exp: Box::new(Expr::Rat { n: 2.0, d: 1.0 }),
            }),
        }
    }

    fn request(expr: Expr, points: u32, recursion: u32, exclusions: bool) -> Request {
        Request {
            min: -10.0,
            max: 10.0,
            points,
            recursion,
            exclusions,
            yspan: 20.0,
            degrees: false,
            expr,
        }
    }

    #[test]
    fn a_smooth_sine_stays_connected() {
        let nodes = sample_curve(&request(
            Expr::Call {
                name: "sin".into(),
                args: vec![Expr::Sym { name: "x".into() }],
            },
            64,
            3,
            true,
        ));
        assert!(nodes.iter().all(|node| !node.cut && node.y.is_some()));
    }

    #[test]
    fn secant_squared_breaks_around_poles() {
        let nodes = sample_curve(&request(secant(), 96, 5, true));
        let cuts = nodes.iter().filter(|node| node.cut).count();
        assert!(cuts >= 4, "expected breaks at the poles, got {cuts}");
        let mut run_has_left = false;
        for node in &nodes {
            if node.cut || node.y.is_none() {
                run_has_left = false;
                continue;
            }
            if node.t < std::f64::consts::FRAC_PI_2 - 0.05 {
                run_has_left = true;
            } else if node.t > std::f64::consts::FRAC_PI_2 + 0.05 && run_has_left {
                panic!("the curve stays connected across pi/2 at x={}", node.t);
            }
        }
    }

    #[test]
    fn plot_points_set_the_initial_grid_when_recursion_is_off() {
        let nodes = sample_curve(&request(
            Expr::Call {
                name: "sin".into(),
                args: vec![Expr::Sym { name: "x".into() }],
            },
            20,
            0,
            false,
        ));
        assert_eq!(nodes.len(), 20);
    }

    #[test]
    fn json_round_trip_returns_nodes() {
        let raw = br#"{"min":-1,"max":1,"points":12,"recursion":1,"exclusions":true,"yspan":2,"degrees":false,"expr":{"t":"call","name":"sin","args":[{"t":"sym","name":"x"}]}}"#;
        let output = sample_json(raw).expect("json");
        let nodes: Vec<(f64, Option<f64>, bool)> = serde_json::from_slice(&output).expect("nodes");
        assert!(nodes.len() >= 12);
        assert!(nodes.iter().all(|node| node.1.is_some()));
    }
}
