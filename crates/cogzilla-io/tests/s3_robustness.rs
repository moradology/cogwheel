use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use bytes::Bytes;
use cogzilla_io::s3::{
    DynamicCredsCache, S3Config, encode_s3_canonical_uri, fetch_ecs_credentials,
    parse_aws_credentials_ini, parse_aws_json_credentials, parse_s3_uri,
};
use cogzilla_io::{CogSource, IoMetrics};
use http_body_util::Full;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as ServerBuilder;
use tokio::net::TcpListener;

async fn spawn_mock_s3<F>(handler: F) -> String
where
    F: Fn(Request<hyper::body::Incoming>) -> Response<Full<Bytes>> + Send + Sync + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handler = Arc::new(handler);
    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            let h = handler.clone();
            tokio::spawn(async move {
                let io = TokioIo::new(stream);
                let svc = service_fn(move |req| {
                    let h = h.clone();
                    async move { Ok::<_, Infallible>(h(req)) }
                });
                let _ = ServerBuilder::new(TokioExecutor::new())
                    .serve_connection(io, svc)
                    .await;
            });
        }
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn s3_sigv4_signs_session_token_requester_pays_and_special_keys() {
    let seen_uri = Arc::new(parking_lot::Mutex::new(String::new()));
    let seen_auth = Arc::new(parking_lot::Mutex::new(String::new()));
    let seen_payer = Arc::new(parking_lot::Mutex::new(String::new()));
    let seen_tok = Arc::new(parking_lot::Mutex::new(String::new()));

    let u = seen_uri.clone();
    let a = seen_auth.clone();
    let p = seen_payer.clone();
    let t = seen_tok.clone();

    let endpoint = spawn_mock_s3(move |req| {
        *u.lock() = req.uri().path().to_string();
        *a.lock() = req
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        *p.lock() = req
            .headers()
            .get("x-amz-request-payer")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        *t.lock() = req
            .headers()
            .get("x-amz-security-token")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        Response::builder()
            .status(StatusCode::PARTIAL_CONTENT)
            .body(Full::new(Bytes::from_static(b"HELLO_S3_COG")))
            .unwrap()
    })
    .await;

    let cfg = S3Config {
        access_key: "AKIAIOSFODNN7EXAMPLE".to_string(),
        secret_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".to_string(),
        session_token: Some("IQoJb3JpZ2luX2VjECSessionToken123".to_string()),
        region: "us-west-2".to_string(),
        endpoint_url: Some(endpoint),
        no_sign_request: false,
        requester_pays: true,
        force_path_style: true,
        max_retries: 2,
    };

    let client = reqwest::Client::new();
    let metrics = Arc::new(IoMetrics::default());
    // Test /vsis3/ alias AND special chars (spaces, +, :, =) in key!
    let src = CogSource::open_remote(
        "/vsis3/usgs-landsat/2025-06-15T18:45:00Z/tile name+1=ok.tif",
        client,
        Some(cfg),
        metrics,
    )
    .unwrap();

    let data = src.fetch_range(0, 12).await.unwrap();
    assert_eq!(&data[..], b"HELLO_S3_COG");

    // Verify RFC 3986 path encoding
    assert_eq!(
        *seen_uri.lock(),
        "/usgs-landsat/2025-06-15T18%3A45%3A00Z/tile%20name%2B1%3Dok.tif"
    );
    // Verify SigV4 includes BOTH x-amz-request-payer AND x-amz-security-token in sorted SignedHeaders
    let auth = seen_auth.lock().clone();
    assert!(
        auth.contains("SignedHeaders=host;range;x-amz-content-sha256;x-amz-date;x-amz-request-payer;x-amz-security-token"),
        "unexpected Authorization header: {auth}"
    );
    assert!(auth.contains("/us-west-2/s3/aws4_request"));
    assert_eq!(*seen_payer.lock(), "requester");
    assert_eq!(*seen_tok.lock(), "IQoJb3JpZ2luX2VjECSessionToken123");
}

#[tokio::test]
async fn s3_auto_discovers_and_caches_bucket_region_on_301() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_cl = calls.clone();

    let endpoint = spawn_mock_s3(move |req| {
        calls_cl.fetch_add(1, Ordering::SeqCst);
        let auth = req
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if auth.contains("/us-east-1/s3/aws4_request") {
            return Response::builder()
                .status(StatusCode::MOVED_PERMANENTLY)
                .header("x-amz-bucket-region", "eu-central-1")
                .body(Full::new(Bytes::new()))
                .unwrap();
        }
        assert!(
            auth.contains("/eu-central-1/s3/aws4_request"),
            "expected re-signed eu-central-1, got {auth}"
        );
        Response::builder()
            .status(StatusCode::PARTIAL_CONTENT)
            .body(Full::new(Bytes::from_static(b"EU_DATA")))
            .unwrap()
    })
    .await;

    let cfg = S3Config {
        access_key: "AKIA_TEST".to_string(),
        secret_key: "SECRET_TEST".to_string(),
        region: "us-east-1".to_string(),
        endpoint_url: Some(endpoint),
        force_path_style: true,
        ..Default::default()
    };

    let src = CogSource::open_remote(
        "s3://region-discovery-test-bucket/scene.tif",
        reqwest::Client::new(),
        Some(cfg),
        Arc::new(IoMetrics::default()),
    )
    .unwrap();

    // First fetch: 1st request gets 301 + x-amz-bucket-region: eu-central-1, 2nd succeeds (2 calls total)
    let d1 = src.fetch_range(0, 7).await.unwrap();
    assert_eq!(&d1[..], b"EU_DATA");
    assert_eq!(calls.load(Ordering::SeqCst), 2);

    // Second fetch: bucket region is cached -> succeeds in 1 call with 0 redirects!
    let d2 = src.fetch_range(0, 7).await.unwrap();
    assert_eq!(&d2[..], b"EU_DATA");
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn s3_retries_transient_503_slowdown_and_supports_no_sign_request() {
    let calls = Arc::new(AtomicUsize::new(0));
    let saw_auth = Arc::new(AtomicUsize::new(0));
    let calls_cl = calls.clone();
    let auth_cl = saw_auth.clone();

    let endpoint = spawn_mock_s3(move |req| {
        if req.headers().contains_key("authorization") {
            auth_cl.fetch_add(1, Ordering::SeqCst);
        }
        let n = calls_cl.fetch_add(1, Ordering::SeqCst);
        if n < 2 {
            return Response::builder()
                .status(StatusCode::SERVICE_UNAVAILABLE)
                .body(Full::new(Bytes::from_static(b"SlowDown")))
                .unwrap();
        }
        Response::builder()
            .status(StatusCode::PARTIAL_CONTENT)
            .body(Full::new(Bytes::from_static(b"RECOVERED")))
            .unwrap()
    })
    .await;

    let cfg = S3Config {
        access_key: "SHOULD_NOT_SIGN".to_string(),
        secret_key: "SHOULD_NOT_SIGN".to_string(),
        endpoint_url: Some(endpoint),
        no_sign_request: true,
        force_path_style: true,
        max_retries: 3,
        ..Default::default()
    };

    let src = CogSource::open_remote(
        "s3://public-bucket/tile.tif",
        reqwest::Client::new(),
        Some(cfg),
        Arc::new(IoMetrics::default()),
    )
    .unwrap();

    let data = src.fetch_range(0, 9).await.unwrap();
    assert_eq!(&data[..], b"RECOVERED");
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(saw_auth.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn s3_parses_credentials_ini_ecs_json_and_uris() {
    let ini = r#"
[default]
aws_access_key_id = AKIA_DEFAULT
aws_secret_access_key = SECRET_DEFAULT

[profile prod-eo]
aws_access_key_id = AKIA_PROD
aws_secret_access_key = SECRET_PROD
aws_session_token = TOK_PROD
"#;
    let c_def = parse_aws_credentials_ini(ini, "default").unwrap();
    assert_eq!(c_def.access_key, "AKIA_DEFAULT");
    assert!(c_def.session_token.is_none());

    let c_prod = parse_aws_credentials_ini(ini, "prod-eo").unwrap();
    assert_eq!(c_prod.access_key, "AKIA_PROD");
    assert_eq!(c_prod.session_token.as_deref(), Some("TOK_PROD"));

    // Mock ECS task metadata endpoint
    let ecs_endpoint = spawn_mock_s3(|_req| {
        let json = r#"{"AccessKeyId":"ASIA_ECS","SecretAccessKey":"SEC_ECS","Token":"TOK_ECS","Expiration":"2099-01-01T00:00:00Z"}"#;
        Response::builder()
            .status(StatusCode::OK)
            .body(Full::new(Bytes::from(json)))
            .unwrap()
    })
    .await;
    let ecs_creds = fetch_ecs_credentials(&reqwest::Client::new(), &ecs_endpoint)
        .await
        .unwrap();
    assert_eq!(ecs_creds.access_key, "ASIA_ECS");
    assert_eq!(ecs_creds.session_token.as_deref(), Some("TOK_ECS"));

    let _ = DynamicCredsCache::default();
    assert_eq!(
        parse_s3_uri("s3://bucket/a/b.tif"),
        Some(("bucket", "a/b.tif"))
    );
    assert_eq!(
        parse_s3_uri("/vsis3/bucket/a/b.tif"),
        Some(("bucket", "a/b.tif"))
    );
    assert_eq!(
        encode_s3_canonical_uri("a/b c+d:e.tif"),
        "/a/b%20c%2Bd%3Ae.tif"
    );
    assert!(parse_aws_json_credentials("{}").is_none());
}
