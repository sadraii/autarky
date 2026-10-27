//! Bounded community HTTP transport. Authentication never follows redirects.

use anyhow::{bail, Context, Result};
use reqwest::{header::HeaderValue, redirect::Policy, Client, Response, Url};
use serde_json::Value;
use std::time::Duration;

pub const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;

pub fn client() -> Result<Client> {
    client_with_timeout(Duration::from_secs(15))
}

fn client_with_timeout(timeout: Duration) -> Result<Client> {
    Client::builder()
        .redirect(Policy::none())
        .connect_timeout(timeout.min(Duration::from_secs(5)))
        .timeout(timeout)
        .no_proxy()
        .build()
        .context("Cannot create the community HTTP client")
}

/// Preserve a configured path prefix without allowing a route to change origin.
pub fn endpoint(base: &str, route: &str) -> Result<Url> {
    let mut url = Url::parse(base).context("Invalid community endpoint URL")?;
    if !url.username().is_empty() || url.password().is_some() {
        bail!("Community endpoint URLs must not contain credentials");
    }
    if url.query().is_some() || url.fragment().is_some() {
        bail!("Community endpoint URLs must not contain a query or fragment");
    }
    let loopback = url.host_str().is_some_and(|host| {
        host.eq_ignore_ascii_case("localhost")
            || host
                .trim_matches(['[', ']'])
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback())
    });
    if url.host().is_none() || !(url.scheme() == "https" || url.scheme() == "http" && loopback) {
        bail!("Community sharing requires HTTPS; HTTP is permitted only for loopback development");
    }
    let path = format!(
        "{}/{}",
        url.path().trim_end_matches('/'),
        route.trim_start_matches('/')
    );
    url.set_path(&path);
    Ok(url)
}

pub fn key_header(key: &str) -> Result<HeaderValue> {
    if key.is_empty() || key.trim() != key {
        bail!("The community API key must be nonempty and must not have surrounding whitespace");
    }
    let mut value = HeaderValue::from_str(key).context("Invalid community API key header")?;
    value.set_sensitive(true);
    Ok(value)
}

/// Validate status and a bounded JSON body without printing an untrusted body.
pub async fn response_json(mut response: Response) -> Result<Value> {
    if !response.status().is_success() {
        bail!("Community server returned HTTP {}", response.status());
    }
    if response
        .content_length()
        .is_some_and(|len| len > MAX_RESPONSE_BYTES as u64)
    {
        bail!("Community response exceeds the 2 MiB limit");
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .context("Cannot read the community response")?
    {
        if chunk.len() > MAX_RESPONSE_BYTES.saturating_sub(body.len()) {
            bail!("Community response exceeds the 2 MiB limit");
        }
        body.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&body).context("Community server did not return valid JSON")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn fixture(
        status: &str,
        headers: &str,
        body: &str,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let reply = format!(
            "HTTP/1.1 {status}\r\nConnection: close\r\nContent-Length: {}\r\n{headers}\r\n{body}",
            body.len()
        );
        let task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0; 8192];
            let count = stream.read(&mut request).await.unwrap();
            assert!(
                count > 0,
                "Expected an HTTP request before the fixture response"
            );
            let _ = stream.write_all(reply.as_bytes()).await;
        });
        (format!("http://{address}"), task)
    }

    #[test]
    fn endpoint_policy_and_base_paths_are_explicit() {
        for base in [
            "http://127.0.0.1:1234",
            "http://[::1]:1234/",
            "http://localhost:1234",
            "https://community.example/base/",
        ] {
            let url = endpoint(base, "/api/v1/benchmarks").unwrap();
            assert!(!url.path().contains("//"));
            assert!(url.path().ends_with("/api/v1/benchmarks"));
        }
        assert_eq!(
            endpoint("https://community.example/base", "api/v1/compare")
                .unwrap()
                .path(),
            "/base/api/v1/compare"
        );
        for base in [
            "http://example.com",
            "http://127.0.0.1.example.com",
            "ftp://127.0.0.1",
            "https://user:private@example.com",
            "https://example.com/?private=value",
            "https://example.com/#private",
        ] {
            assert!(endpoint(base, "api/v1/benchmark").is_err());
        }
        assert_eq!(
            endpoint("https://community.example", "//other.example/path")
                .unwrap()
                .host_str(),
            Some("community.example")
        );
    }

    #[test]
    fn authentication_debug_output_is_redacted() {
        let header = key_header("fixture-private-key").unwrap();
        assert!(header.is_sensitive());
        assert!(!format!("{header:?}").contains("fixture-private-key"));
        for invalid in ["", " key", "key\r\nInjected: value"] {
            assert!(key_header(invalid).is_err());
        }
    }

    #[tokio::test]
    async fn redirects_never_forward_authentication_to_a_second_origin() {
        let destination = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let headers = format!(
            "Location: http://{}/leak\r\n",
            destination.local_addr().unwrap()
        );
        let (base, server) = fixture("302 Found", &headers, "").await;
        let response = client()
            .unwrap()
            .get(endpoint(&base, "api/v1/benchmarks").unwrap())
            .header("X-API-Key", key_header("fixture-private-key").unwrap())
            .send()
            .await
            .unwrap();
        assert!(response_json(response)
            .await
            .unwrap_err()
            .to_string()
            .contains("302"));
        assert!(
            tokio::time::timeout(Duration::from_millis(100), destination.accept())
                .await
                .is_err()
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn status_and_json_errors_are_not_empty_results() {
        for (status, body) in [
            ("500 Internal Server Error", "private server error"),
            ("200 OK", "<html>not JSON</html>"),
        ] {
            let (base, server) = fixture(status, "", body).await;
            let response = client()
                .unwrap()
                .get(endpoint(&base, "api/v1/benchmarks").unwrap())
                .send()
                .await
                .unwrap();
            let error = response_json(response).await.unwrap_err();
            assert!(!format!("{error:#}").contains(body));
            server.await.unwrap();
        }
        let (base, server) = fixture("200 OK", "", r#"{"count":0,"results":[]}"#).await;
        let response = client()
            .unwrap()
            .get(endpoint(&base, "api/v1/benchmarks").unwrap())
            .send()
            .await
            .unwrap();
        assert_eq!(
            response_json(response).await.unwrap()["results"],
            serde_json::json!([])
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn stalled_connections_obey_a_request_deadline() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (_connection, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_millis(500)).await;
        });
        let result = client_with_timeout(Duration::from_millis(50))
            .unwrap()
            .get(endpoint(&format!("http://{address}"), "api/v1/benchmarks").unwrap())
            .send()
            .await;
        assert!(result.unwrap_err().is_timeout());
        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn a_streamed_oversized_body_is_rejected() {
        let body = " ".repeat(MAX_RESPONSE_BYTES + 1);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0; 8192];
            let count = stream.read(&mut request).await.unwrap();
            assert!(
                count > 0,
                "Expected an HTTP request before the fixture response"
            );
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n")
                .await
                .unwrap();
            let _ = stream.write_all(body.as_bytes()).await;
        });
        let response = client()
            .unwrap()
            .get(endpoint(&format!("http://{address}"), "api/v1/benchmarks").unwrap())
            .send()
            .await
            .unwrap();
        assert!(response_json(response)
            .await
            .unwrap_err()
            .to_string()
            .contains("2 MiB"));
        server.await.unwrap();
    }
}
