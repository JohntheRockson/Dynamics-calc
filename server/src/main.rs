//! HTTP API + static file server for the attitude-sim web app.
//!
//! Two JSON endpoints (`GET /api/scenarios`, `POST /api/simulate`) wrap
//! `attitude_engine::run_sim` directly -- one request in, one full
//! [`attitude_engine::SimLog`] out. There is no session/streaming state: a
//! run of a few thousand RK4 steps completes in well under a millisecond,
//! so a plain request/response is simpler and just as "live" as a
//! websocket would feel. In production mode (`web/dist` present) this
//! binary also serves the built frontend, so the whole app is one process
//! on one port.

use attitude_engine::sim::{run_sim, scenario_catalog, SimError, SimRequest};
use axum::{
    body::Bytes,
    extract::Json,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use serde_json::json;
use std::net::SocketAddr;
use std::path::PathBuf;
use tower_http::compression::CompressionLayer;
use tower_http::cors::CorsLayer;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;

struct ApiError(SimError);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match &self.0 {
            SimError::TooManySamples(_) => StatusCode::PAYLOAD_TOO_LARGE,
            _ => StatusCode::BAD_REQUEST,
        };
        (status, Json(json!({ "error": self.0.to_string() }))).into_response()
    }
}

async fn get_scenarios() -> Json<serde_json::Value> {
    Json(json!({ "scenarios": scenario_catalog() }))
}

async fn post_simulate(Json(req): Json<SimRequest>) -> Result<Json<serde_json::Value>, ApiError> {
    let log = run_sim(&req).map_err(ApiError)?;
    Ok(Json(json!(log)))
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok", "service": "attitude-server" }))
}

fn agent_unavailable() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({ "error": "The math agent is not running. From web/, run npm run agent with XAI_API_KEY set." })),
    )
        .into_response()
}

/// The built app is served by this process, so /api/agent has to be forwarded
/// to the Node agent (`npm run agent` in web/). Dev Vite proxies that path
/// itself and never hits this handler.
async fn forward_agent(method: &str, path: &str, body: Bytes) -> Response {
    let port = std::env::var("MATH_AGENT_PORT").unwrap_or_else(|_| "8788".to_string());
    let head = if method == "POST" {
        format!("POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len())
    } else {
        format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n")
    };
    let connect = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        tokio::net::TcpStream::connect(format!("127.0.0.1:{port}")),
    )
    .await;
    let mut stream = match connect {
        Ok(Ok(stream)) => stream,
        _ => return agent_unavailable(),
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    if stream.write_all(head.as_bytes()).await.is_err() {
        return agent_unavailable();
    }
    if method == "POST" && stream.write_all(&body).await.is_err() {
        return agent_unavailable();
    }
    let mut raw = Vec::new();
    if stream.read_to_end(&mut raw).await.is_err() {
        return agent_unavailable();
    }
    let Some(split) = raw.windows(4).position(|mark| mark == b"\r\n\r\n") else {
        return (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": "The math agent sent an incomplete response." })),
        )
            .into_response();
    };
    let preface = String::from_utf8_lossy(&raw[..split]);
    let status = preface
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse::<u16>().ok())
        .and_then(|code| StatusCode::from_u16(code).ok())
        .unwrap_or(StatusCode::BAD_GATEWAY);
    let payload = Bytes::copy_from_slice(&raw[split + 4..]);
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        payload,
    )
        .into_response()
}

async fn get_agent_health() -> Response {
    forward_agent("GET", "/api/agent/health", Bytes::new()).await
}

async fn post_agent(body: Bytes) -> Response {
    forward_agent("POST", "/api/agent", body).await
}

fn frontend_dir() -> Option<PathBuf> {
    // Look for a built frontend next to the workspace (web/dist), trying a
    // couple of plausible working directories so `cargo run` from either
    // the repo root or `server/` finds it.
    for candidate in ["web/dist", "../web/dist"] {
        let p = PathBuf::from(candidate);
        if p.join("index.html").is_file() {
            return Some(p);
        }
    }
    None
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("info".parse().unwrap()),
        )
        .init();

    let api = Router::new()
        .route("/api/health", get(health))
        .route("/api/scenarios", get(get_scenarios))
        .route("/api/simulate", post(post_simulate))
        .route("/api/agent/health", get(get_agent_health))
        .route("/api/agent", post(post_agent))
        .layer(CorsLayer::permissive());

    let app = match frontend_dir() {
        Some(dir) => {
            tracing::info!("serving built frontend from {}", dir.display());
            let index = dir.join("index.html");
            let serve_dir = ServeDir::new(&dir).not_found_service(ServeFile::new(index));
            api.fallback_service(serve_dir)
        }
        None => {
            tracing::warn!("web/dist not found; run `npm run build` in web/ to serve the frontend from this binary (API-only for now)");
            api
        }
    };

    // SimLog payloads are a few thousand samples of mostly-small floats
    // (very compressible) and the frontend bundle is a few hundred KB of
    // JS/CSS; gzip shrinks both substantially for a snappier first paint
    // and simulate-request round trip.
    let app = app
        .layer(CompressionLayer::new())
        .layer(TraceLayer::new_for_http());

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("attitude-server listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind port");
    axum::serve(listener, app).await.expect("server error");
}

#[cfg(test)]
mod tests {
    use super::*;
    use attitude_engine::sim::SimRequest;

    #[test]
    fn default_request_serializes_and_round_trips_via_engine() {
        let req = SimRequest::default();
        let log = run_sim(&req).expect("default request should run");
        let value = serde_json::to_value(&log).expect("SimLog serializes");
        assert!(value["t"].is_array());
        assert!(value["summary"]["final_att_error_deg"].is_number());
    }
}
