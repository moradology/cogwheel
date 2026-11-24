//! TLS for the Postgres connection, with libpq's `sslmode` meanings.
//!
//! `prefer` and `require` encrypt without checking who the server is, exactly as libpq
//! does, so managed databases whose certificates chain to a private CA work without
//! extra setup. `verify-ca` and `verify-full` check the chain against `PGSSLROOTCERT`
//! when it is set and the public web roots otherwise.

use std::sync::Arc;

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::CryptoProvider;
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{ClientConfig, DigitallySignedStruct, RootCertStore, SignatureScheme};
use tokio_postgres_rustls::MakeRustlsConnect;

use crate::PgStacError;

/// How far the server's certificate is checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlsVerify {
    /// Encrypt only (`sslmode=prefer` and `sslmode=require`).
    None,
    /// Check the chain and the host name (`sslmode=verify-ca` and `sslmode=verify-full`).
    Full,
}

#[derive(Debug)]
struct AcceptAnyServerCert(Arc<CryptoProvider>);

impl ServerCertVerifier for AcceptAnyServerCert {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}

fn load_root_file(path: &str) -> Result<RootCertStore, PgStacError> {
    let pem = std::fs::read(path)
        .map_err(|e| PgStacError::Config(format!("cannot read PGSSLROOTCERT '{path}': {e}")))?;
    let mut store = RootCertStore::empty();
    let mut found = 0usize;
    for der in pem_certificates(&pem) {
        if store.add(CertificateDer::from(der)).is_ok() {
            found += 1;
        }
    }
    if found == 0 {
        return Err(PgStacError::Config(format!(
            "PGSSLROOTCERT '{path}' contains no usable certificates"
        )));
    }
    Ok(store)
}

/// Decode every `CERTIFICATE` block of a PEM file to DER.
fn pem_certificates(pem: &[u8]) -> Vec<Vec<u8>> {
    const BEGIN: &str = "-----BEGIN CERTIFICATE-----";
    const END: &str = "-----END CERTIFICATE-----";
    let text = String::from_utf8_lossy(pem);
    let mut out = Vec::new();
    let mut rest: &str = &text;
    while let Some(start) = rest.find(BEGIN) {
        let after = &rest[start + BEGIN.len()..];
        let Some(end) = after.find(END) else { break };
        if let Some(der) = base64_decode(&after[..end]) {
            out.push(der);
        }
        rest = &after[end + END.len()..];
    }
    out
}

fn base64_decode(input: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut acc = 0u32;
    let mut bits = 0u32;
    for b in input.bytes() {
        let v = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' | b'\r' | b'\n' | b' ' | b'\t' => continue,
            _ => return None,
        };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

pub(crate) fn make_connector(verify: TlsVerify) -> Result<MakeRustlsConnect, PgStacError> {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let builder = ClientConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .map_err(|e| PgStacError::Config(format!("TLS setup failed: {e}")))?;
    let config = match verify {
        TlsVerify::None => builder
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(AcceptAnyServerCert(provider)))
            .with_no_client_auth(),
        TlsVerify::Full => {
            let roots = match std::env::var("PGSSLROOTCERT") {
                Ok(path) if !path.trim().is_empty() => load_root_file(path.trim())?,
                _ => RootCertStore {
                    roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
                },
            };
            builder.with_root_certificates(roots).with_no_client_auth()
        }
    };
    Ok(MakeRustlsConnect::new(config))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trips_standard_alphabet() {
        assert_eq!(base64_decode("aGVsbG8gd29ybGQ=").unwrap(), b"hello world");
        assert_eq!(base64_decode("aGVs\nbG8=").unwrap(), b"hello");
        assert!(base64_decode("not*base64").is_none());
    }

    #[test]
    fn pem_blocks_are_extracted_in_order() {
        let pem = b"junk\n-----BEGIN CERTIFICATE-----\nAQID\n-----END CERTIFICATE-----\n\
                    -----BEGIN CERTIFICATE-----\nBAUG\n-----END CERTIFICATE-----\n";
        assert_eq!(pem_certificates(pem), vec![vec![1, 2, 3], vec![4, 5, 6]]);
    }

    #[test]
    fn both_verification_modes_build() {
        assert!(make_connector(TlsVerify::None).is_ok());
        assert!(make_connector(TlsVerify::Full).is_ok());
    }
}
