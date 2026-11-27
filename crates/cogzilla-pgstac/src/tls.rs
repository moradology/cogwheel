//! TLS for the Postgres connection, with libpq's `sslmode` meanings.
//!
//! `prefer` and `require` encrypt without checking who the server is, exactly as libpq
//! does, so managed databases whose certificates chain to a private CA work without
//! extra setup. `verify-ca` and `verify-full` check the chain against the configured
//! root certificate file when there is one and the public web roots otherwise, and
//! `require` checks the chain too once a root certificate file is configured. Only
//! `verify-full` also checks the host name.

use std::sync::Arc;

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::client::verify_server_cert_signed_by_trust_anchor;
use rustls::crypto::CryptoProvider;
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::server::ParsedCertificate;
use rustls::{ClientConfig, DigitallySignedStruct, RootCertStore, SignatureScheme};
use tokio_postgres_rustls::MakeRustlsConnect;

use crate::PgStacError;

/// How far the server's certificate is checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlsVerify {
    /// Encrypt only (`sslmode=prefer`, and `sslmode=require` with no root certificate).
    None,
    /// Check the chain but not the host name (`sslmode=verify-ca`, and `sslmode=require`
    /// with a root certificate).
    Ca,
    /// Check the chain and the host name (`sslmode=verify-full`).
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

/// Checks that the certificate chains to a trusted root and is in date, and leaves the
/// host name alone, so a server reached through an alias or an address still passes.
#[derive(Debug)]
struct ChainOnly {
    roots: RootCertStore,
    provider: Arc<CryptoProvider>,
}

impl ServerCertVerifier for ChainOnly {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        let cert = ParsedCertificate::try_from(end_entity)?;
        verify_server_cert_signed_by_trust_anchor(
            &cert,
            &self.roots,
            intermediates,
            now,
            self.provider.signature_verification_algorithms.all,
        )?;
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
            &self.provider.signature_verification_algorithms,
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
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider.signature_verification_algorithms.supported_schemes()
    }
}

fn load_root_file(path: &str) -> Result<RootCertStore, PgStacError> {
    let pem = std::fs::read(path)
        .map_err(|e| PgStacError::Config(format!("cannot read root certificate file '{path}': {e}")))?;
    let mut store = RootCertStore::empty();
    let mut found = 0usize;
    for der in pem_certificates(&pem) {
        if store.add(CertificateDer::from(der)).is_ok() {
            found += 1;
        }
    }
    if found == 0 {
        return Err(PgStacError::Config(format!(
            "root certificate file '{path}' contains no usable certificates"
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

/// The roots a verifying mode trusts: the given PEM file, or the public web roots.
fn trusted_roots(root_cert: Option<&str>) -> Result<RootCertStore, PgStacError> {
    match root_cert {
        Some(path) => load_root_file(path),
        None => Ok(RootCertStore {
            roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
        }),
    }
}

pub(crate) fn make_connector(
    verify: TlsVerify,
    root_cert: Option<&str>,
) -> Result<MakeRustlsConnect, PgStacError> {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let builder = ClientConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .map_err(|e| PgStacError::Config(format!("TLS setup failed: {e}")))?;
    let config = match verify {
        TlsVerify::None => builder
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(AcceptAnyServerCert(provider)))
            .with_no_client_auth(),
        TlsVerify::Ca => builder
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(ChainOnly {
                roots: trusted_roots(root_cert)?,
                provider,
            }))
            .with_no_client_auth(),
        TlsVerify::Full => builder
            .with_root_certificates(trusted_roots(root_cert)?)
            .with_no_client_auth(),
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

    const TESTDATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata");

    fn test_cert(name: &str) -> CertificateDer<'static> {
        let pem = std::fs::read(format!("{TESTDATA}/{name}")).unwrap();
        CertificateDer::from(pem_certificates(&pem).remove(0))
    }

    // The test certificates were issued in November 2026 and run for a hundred years.
    fn a_time_they_are_valid() -> UnixTime {
        UnixTime::since_unix_epoch(std::time::Duration::from_secs(1_900_000_000))
    }

    fn chain_only(root_file: &str) -> ChainOnly {
        ChainOnly {
            roots: load_root_file(&format!("{TESTDATA}/{root_file}")).unwrap(),
            provider: Arc::new(rustls::crypto::ring::default_provider()),
        }
    }

    fn full(root_file: &str) -> Arc<rustls::client::WebPkiServerVerifier> {
        let roots = load_root_file(&format!("{TESTDATA}/{root_file}")).unwrap();
        rustls::client::WebPkiServerVerifier::builder_with_provider(
            Arc::new(roots),
            Arc::new(rustls::crypto::ring::default_provider()),
        )
        .build()
        .unwrap()
    }

    #[test]
    fn chain_only_accepts_a_trusted_certificate_under_any_name() {
        let verifier = chain_only("ca.pem");
        let leaf = test_cert("leaf.pem");
        for name in ["db.example.test", "an-alias.internal", "10.1.2.3"] {
            let name = ServerName::try_from(name).unwrap();
            assert!(
                verifier
                    .verify_server_cert(&leaf, &[], &name, &[], a_time_they_are_valid())
                    .is_ok()
            );
        }
    }

    #[test]
    fn chain_only_refuses_an_untrusted_or_expired_certificate() {
        let leaf = test_cert("leaf.pem");
        let name = ServerName::try_from("db.example.test").unwrap();
        // Same issuer name, different key: the signature does not check out.
        assert!(
            chain_only("other-ca.pem")
                .verify_server_cert(&leaf, &[], &name, &[], a_time_they_are_valid())
                .is_err()
        );
        let long_after = UnixTime::since_unix_epoch(std::time::Duration::from_secs(9_000_000_000));
        assert!(
            chain_only("ca.pem")
                .verify_server_cert(&leaf, &[], &name, &[], long_after)
                .is_err()
        );
        let garbage = CertificateDer::from(vec![1u8, 2, 3]);
        assert!(
            chain_only("ca.pem")
                .verify_server_cert(&garbage, &[], &name, &[], a_time_they_are_valid())
                .is_err()
        );
    }

    #[test]
    fn full_verification_still_checks_the_name() {
        let verifier = full("ca.pem");
        let leaf = test_cert("leaf.pem");
        let right = ServerName::try_from("db.example.test").unwrap();
        let alias = ServerName::try_from("an-alias.internal").unwrap();
        assert!(
            verifier
                .verify_server_cert(&leaf, &[], &right, &[], a_time_they_are_valid())
                .is_ok()
        );
        assert!(
            verifier
                .verify_server_cert(&leaf, &[], &alias, &[], a_time_they_are_valid())
                .is_err()
        );
    }

    #[test]
    fn every_verification_mode_builds() {
        let root = format!("{TESTDATA}/ca.pem");
        for verify in [TlsVerify::None, TlsVerify::Ca, TlsVerify::Full] {
            assert!(make_connector(verify, None).is_ok());
            assert!(make_connector(verify, Some(&root)).is_ok());
        }
        assert!(make_connector(TlsVerify::None, Some("/no/such/file.pem")).is_ok());
        assert!(make_connector(TlsVerify::Ca, Some("/no/such/file.pem")).is_err());
        assert!(make_connector(TlsVerify::Full, Some("/no/such/file.pem")).is_err());
    }
}
