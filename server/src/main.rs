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
    extract::Json,
    http::StatusCode,
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
    let app = app.layer(CompressionLayer::new()).layer(TraceLayer::new_for_http());

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
