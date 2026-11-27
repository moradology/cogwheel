//! HTTP/1.1 + HTTP/2 TCP listener lifecycle and graceful connection draining.

use std::net::SocketAddr;
use std::sync::Arc;

use hyper::service::service_fn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as ServerBuilder;
use tokio::net::TcpListener;

use super::handle_request;
use crate::engine::TileEngine;

/// Bind and serve HTTP/1.1 + HTTP/2 requests until shutdown.
pub async fn serve(engine: Arc<TileEngine>, addr: SocketAddr) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    serve_listener(engine, listener).await
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sig) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            sig.recv().await;
        } else {
            std::future::pending::<()>().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}

/// Serve HTTP/1.1 + HTTP/2 requests on an already-bound listener until `SIGINT`/`SIGTERM`.
pub async fn serve_listener(engine: Arc<TileEngine>, listener: TcpListener) -> std::io::Result<()> {
    serve_listener_with_shutdown(engine, listener, shutdown_signal()).await
}

/// Serve HTTP/1.1 + HTTP/2 requests on `listener` until `shutdown` resolves, then immediately
/// close the listening socket and drain in-flight connections up to
/// `COGZILLA_SHUTDOWN_TIMEOUT_SECS` (default 15 s).
pub async fn serve_listener_with_shutdown<F>(
    engine: Arc<TileEngine>,
    listener: TcpListener,
    shutdown: F,
) -> std::io::Result<()>
where
    F: std::future::Future<Output = ()>,
{
    let graceful = hyper_util::server::graceful::GracefulShutdown::new();
    tokio::pin!(shutdown);
    loop {
        let (stream, _) = tokio::select! {
            _ = &mut shutdown => {
                tracing::info!("shutdown signal received; draining in-flight connections");
                break;
            }
            accept_res = listener.accept() => match accept_res {
                Ok(pair) => pair,
                Err(e) => {
                    // Transient socket errors (EMFILE/ENFILE/ECONNABORTED) must never crash the server.
                    tracing::warn!(error = %e, "TCP accept error; backing off briefly");
                    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
                    continue;
                }
            }
        };
        let _ = stream.set_nodelay(true);
        let io = TokioIo::new(stream);
        let eng = engine.clone();
        let svc = service_fn(move |req| handle_request(eng.clone(), req));
        let mut builder = ServerBuilder::new(TokioExecutor::new());
        builder
            .http1()
            .timer(hyper_util::rt::TokioTimer::new())
            .header_read_timeout(std::time::Duration::from_secs(15));
        let conn = builder.serve_connection(io, svc).into_owned();
        let fut = graceful.watch(conn);
        tokio::spawn(async move {
            let _ = fut.await;
        });
    }
    // Close the listening socket immediately so load balancers get ECONNREFUSED instead of
    // queueing new handshakes in the kernel backlog while in-flight connections drain.
    drop(listener);
    let drain_secs = crate::config::Config::global().shutdown_timeout_secs;
    match tokio::time::timeout(
        std::time::Duration::from_secs(drain_secs),
        graceful.shutdown(),
    )
    .await
    {
        Ok(()) => tracing::info!("all in-flight connections drained"),
        Err(_) => {
            tracing::warn!(timeout_secs = drain_secs, "graceful shutdown timed out; forcing exit");
        }
    }
    Ok(())
}
