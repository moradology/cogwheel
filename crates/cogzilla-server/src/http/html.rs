//! `f=html` on the routes that describe things (tilesets, tile matrix sets, colour maps,
//! algorithms, conformance): the JSON document those routes answer, written out as a page,
//! as TiTiler writes its templates. The page carries a trail back up the path, a link to the
//! same document as JSON, the document's links, and the rest of it as tables.

use serde_json::Value;

/// Whether a route answers `f=html`: a list of tilesets or one tileset (`.../tiles`,
/// `.../tiles/{tileMatrixSetId}`), and the tile matrix set, colour map, algorithm and
/// conformance documents.
pub(crate) fn offers_html(path: &str) -> bool {
    let segs: Vec<&str> = path.trim_matches('/').split('/').filter(|s| !s.is_empty()).collect();
    match segs.as_slice() {
        ["tileMatrixSets"]
        | ["tileMatrixSets", _]
        | ["colorMaps"]
        | ["colorMaps", _]
        | ["algorithms"]
        | ["algorithms", _]
        | ["conformance"] => true,
        [.., "tiles"] => segs.len() > 1,
        [.., "tiles", tms] => segs.len() > 2 && !tms.contains('.'),
        _ => false,
    }
}

/// The last explicit `f=` or `format=` query parameter value (percent-decoded), matching `parse_query`.
pub(crate) fn explicit_format(query: &str) -> Option<String> {
    query
        .split('&')
        .filter_map(|kv| kv.split_once('='))
        .rfind(|(k, _)| *k == "f" || *k == "format")
        .map(|(_, v)| super::percent_decode(v))
}

/// Whether an `Accept` header prefers `text/html` over `application/json` (RFC 9110 §12.5.1).
/// Wildcards (`*/*`, `text/*`) do not trigger HTML on their own, so API clients (`curl`, etc.)
/// keep JSON, while explicit higher-q `application/*` or `*/*` beat a lower-q `text/html`.
pub(crate) fn accepts_html(accept: &str) -> bool {
    let mut best_html: Option<(f32, usize)> = None;
    let mut best_json: Option<(f32, usize)> = None;
    let mut best_app_wild: Option<(f32, usize)> = None;
    let mut best_any_wild: Option<(f32, usize)> = None;
    for (idx, item) in accept.split(',').enumerate() {
        let mut parts = item.split(';');
        let Some(media_type) = parts.next().map(str::trim) else {
            continue;
        };
        let is_html = media_type.eq_ignore_ascii_case("text/html");
        let is_json = media_type.eq_ignore_ascii_case("application/json");
        let is_app_wild = media_type.eq_ignore_ascii_case("application/*");
        let is_any_wild = media_type == "*/*";
        if !is_html && !is_json && !is_app_wild && !is_any_wild {
            continue;
        }
        let mut q = 1.0f32;
        for param in parts {
            if let Some((k, v)) = param.split_once('=')
                && k.trim().eq_ignore_ascii_case("q")
            {
                if let Ok(parsed) = v.trim().parse::<f32>()
                    && parsed.is_finite()
                {
                    q = parsed.clamp(0.0, 1.0);
                } else {
                    q = 0.0;
                }
            }
        }
        let slot = if is_html {
            &mut best_html
        } else if is_json {
            &mut best_json
        } else if is_app_wild {
            &mut best_app_wild
        } else {
            &mut best_any_wild
        };
        if slot.is_none_or(|(prev_q, _)| q > prev_q) {
            *slot = Some((q, idx));
        }
    }
    let Some((q_html, pos_html)) = best_html else {
        return false;
    };
    if q_html <= 0.0 {
        return false;
    }
    // Effective q for application/json is the most-specific matching range (RFC 9110 §12.5.1).
    let eff_json = best_json.or(best_app_wild).or(best_any_wild);
    match eff_json {
        Some((q_json, pos_json)) => {
            if best_json.is_some() {
                q_html > q_json || (q_html == q_json && pos_html < pos_json)
            } else {
                // Explicit `text/html` is more specific than `application/*` or `*/*` when q is equal.
                q_html >= q_json
            }
        }
        None => true,
    }
}

/// Whether a request asks for HTML: explicit `?f=` / `?format=` always wins over `Accept`.
pub(crate) fn wants_html(query: &str, accept: Option<&str>) -> bool {
    match explicit_format(query) {
        Some(f) => f.eq_ignore_ascii_case("html"),
        None => accept.is_some_and(accepts_html),
    }
}

/// Ensure `Vary` includes `Accept` without duplicating it or clobbering existing tokens.
pub(crate) fn append_vary_accept(headers: &mut hyper::HeaderMap) {
    if let Some(existing) = headers
        .get(hyper::header::VARY)
        .and_then(|v| v.to_str().ok())
    {
        let has_accept = existing
            .split(',')
            .any(|tok| tok.trim().eq_ignore_ascii_case("accept"));
        if !has_accept
            && let Ok(hv) = hyper::header::HeaderValue::from_str(&format!("{existing}, Accept"))
        {
            headers.insert(hyper::header::VARY, hv);
        }
    } else {
        headers.insert(
            hyper::header::VARY,
            hyper::header::HeaderValue::from_static("Accept"),
        );
    }
}

/// A query with `f` set to `format` in place of whatever it was.
fn query_with_format(query: &str, format: &str) -> String {
    let mut out: Vec<String> = query
        .split('&')
        .filter(|kv| !kv.is_empty() && kv.split_once('=').map_or(*kv, |(k, _)| k) != "f")
        .map(str::to_string)
        .collect();
    out.push(format!("f={format}"));
    out.join("&")
}

/// A link to another document, kept as HTML when that document offers it.
fn html_href(href: &str) -> String {
    let (base, query) = href.split_once('?').unwrap_or((href, ""));
    let path = base.split_once("://").map_or(base, |(_, rest)| rest.find('/').map_or("", |i| &rest[i..]));
    if offers_html(path) && !href.contains('{') {
        format!("{base}?{}", query_with_format(query, "html"))
    } else {
        href.to_string()
    }
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// `tileMatrixSets` as `TileMatrixSets`, `ff265…` as `Ff265…`, as TiTiler titles its trail.
fn titled(seg: &str) -> String {
    let mut c = seg.chars();
    c.next().map_or_else(String::new, |f| f.to_uppercase().chain(c).collect())
}

fn value_html(v: &Value, out: &mut String) {
    match v {
        Value::String(s) if s.starts_with("http://") || s.starts_with("https://") => {
            out.push_str(&format!("<a href=\"{0}\">{0}</a>", escape(s)));
        }
        Value::String(s) => out.push_str(&escape(s)),
        Value::Null => out.push_str("<em>null</em>"),
        Value::Bool(_) | Value::Number(_) => out.push_str(&v.to_string()),
        Value::Array(items) if items.iter().all(|i| !i.is_object() && !i.is_array()) => {
            let parts: Vec<String> = items
                .iter()
                .map(|i| {
                    let mut s = String::new();
                    value_html(i, &mut s);
                    s
                })
                .collect();
            out.push_str(&parts.join(", "));
        }
        Value::Array(items) => {
            out.push_str("<ol>");
            for item in items {
                out.push_str("<li>");
                value_html(item, out);
                out.push_str("</li>");
            }
            out.push_str("</ol>");
        }
        Value::Object(map) => {
            out.push_str("<table class=\"table table-sm\">");
            for (k, v) in map {
                out.push_str(&format!("<tr><th>{}</th><td>", escape(k)));
                if k == "links" {
                    links_html(v, out);
                } else {
                    value_html(v, out);
                }
                out.push_str("</td></tr>");
            }
            out.push_str("</table>");
        }
    }
}

fn links_html(links: &Value, out: &mut String) {
    let Some(links) = links.as_array() else {
        return value_html(links, out);
    };
    out.push_str("<ul>");
    for link in links {
        let href = link["href"].as_str().unwrap_or_default();
        let text = link["title"].as_str().or_else(|| link["rel"].as_str()).unwrap_or(href);
        out.push_str(&format!("<li><a href=\"{}\">{}</a></li>", escape(&html_href(href)), escape(text)));
    }
    out.push_str("</ul>");
}

/// The page for `doc`, the JSON a route answered at `path` with `query`.
pub(crate) fn page(path: &str, query: &str, doc: &Value) -> String {
    let segs: Vec<&str> = path.trim_matches('/').split('/').filter(|s| !s.is_empty()).collect();
    let title = doc["title"]
        .as_str()
        .map(str::to_string)
        .or_else(|| doc.get("tilesets").map(|_| "Tilesets".to_string()))
        .unwrap_or_else(|| segs.last().map_or_else(|| "cogzilla".to_string(), |s| titled(s)));

    // The trail, relative to this page so that it holds under any prefix the server is behind.
    // A page at `/a/b/c` sits in `/a/b/`: `/` is `../../` from there and `/a` is `../../a`.
    let depth = segs.len();
    let home = if depth > 1 { "../".repeat(depth - 1) } else { "./".to_string() };
    let mut trail = format!("<li class=\"breadcrumb-item\"><a href=\"{home}\">Home</a></li>");
    for (i, seg) in segs.iter().enumerate() {
        if i + 1 == depth {
            trail.push_str(&format!("<li class=\"breadcrumb-item active\" aria-current=\"page\">{}</li>", escape(&titled(seg))));
        } else {
            trail.push_str(&format!(
                "<li class=\"breadcrumb-item\"><a href=\"{}{}\">{}</a></li>",
                "../".repeat(depth - 1 - i),
                escape(seg),
                escape(&titled(seg))
            ));
        }
    }
    let json_href = format!("{}?{}", segs.last().copied().unwrap_or(""), query_with_format(query, "json"));
    trail.push_str(&format!("<li class=\"ml-auto json-link\"><a target=\"_blank\" href=\"{}\">JSON</a></li>", escape(&json_href)));

    let mut body = format!("<h1>{}</h1>", escape(&title));
    if let Some(links) = doc.get("links") {
        body.push_str("<h2>Links</h2>");
        links_html(links, &mut body);
    }
    if let Some(sets) = doc.get("tilesets").and_then(Value::as_array) {
        body.push_str("<h2>Tilesets</h2><ul>");
        for set in sets {
            let title = set["title"].as_str().unwrap_or_default();
            let href = set["links"]
                .as_array()
                .and_then(|l| l.iter().find(|l| l["rel"] == "self"))
                .and_then(|l| l["href"].as_str())
                .unwrap_or_default();
            body.push_str(&format!(
                "<li><a href=\"{}\">tileset tiled using {} TileMatrixSet</a></li>",
                escape(&html_href(href)),
                escape(title)
            ));
        }
        body.push_str("</ul>");
    }
    if let Value::Object(map) = doc {
        let rest: serde_json::Map<String, Value> =
            map.iter().filter(|(k, _)| !matches!(k.as_str(), "links" | "tilesets" | "title")).map(|(k, v)| (k.clone(), v.clone())).collect();
        if !rest.is_empty() {
            value_html(&Value::Object(rest), &mut body);
        }
    } else {
        value_html(doc, &mut body);
    }

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<title>{title}</title>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no"/>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.5.3/css/bootstrap.min.css"/>
<style>body {{ padding: 1.5rem; }} .json-link {{ font-weight: normal; }} th {{ white-space: nowrap; }}</style>
</head>
<body>
<nav aria-label="breadcrumb"><ol class="breadcrumb bg-light">{trail}</ol></nav>
{body}
</body>
</html>
"#,
        title = escape(&title)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn routes_that_offer_html() {
        for path in [
            "/cog/tiles",
            "/cog/tiles/WebMercatorQuad",
            "/searches/abc/tiles",
            "/collections/c/items/i/tiles/WGS1984Quad",
            "/tileMatrixSets",
            "/tileMatrixSets/WebMercatorQuad",
            "/colorMaps",
            "/colorMaps/viridis",
            "/algorithms",
            "/algorithms/hillshade",
            "/conformance",
        ] {
            assert!(offers_html(path), "{path}");
        }
        for path in ["/cog/tiles/WebMercatorQuad/1/2/3", "/cog/tiles/1/2/3.png", "/tiles", "/cog/info", "/cog/tiles/x.png"] {
            assert!(!offers_html(path), "{path}");
        }
        assert!(wants_html("url=a&f=html", None));
        assert!(wants_html("format=html", None));
        assert!(!wants_html("url=a&f=json", Some("text/html,application/xhtml+xml,*/*;q=0.8")));
        assert!(!wants_html("format=json", Some("text/html,application/xhtml+xml,*/*;q=0.8")));
        assert!(!wants_html("f=html&f=json", None));
        assert!(wants_html("f=json&f=html", None));
        assert!(wants_html("f=%68tml", None));
        assert!(wants_html("", Some("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")));
        assert!(!wants_html("", Some("application/json")));
        assert!(!wants_html("", Some("*/*")));
        assert!(!wants_html("", Some("application/json, text/html;q=0.9")));
        assert!(!wants_html("", Some("application/*, text/html;q=0.5")));
        assert!(!wants_html("", Some("text/html;q=0.8, */*;q=0.9")));
        assert!(!wants_html("", Some("text/html;q=0, */*;q=0.8")));
        assert!(wants_html("", Some("text/html;q=0.9, application/json;q=0.5")));
    }

    #[test]
    fn a_page_of_a_document() {
        let doc = json!({
            "tilesets": [{
                "title": "WebMercatorQuad",
                "links": [{"href": "http://h/cog/tiles/WebMercatorQuad?url=a%26b", "rel": "self", "type": "application/json"}],
            }],
            "note": "<b>",
        });
        let page = page("/cog/tiles", "url=a%26b&f=html", &doc);
        assert!(page.contains("<title>Tilesets</title>"));
        assert!(page.contains("href=\"http://h/cog/tiles/WebMercatorQuad?url=a%26b&amp;f=html\""));
        assert!(page.contains("tileset tiled using WebMercatorQuad TileMatrixSet"));
        assert!(page.contains("href=\"tiles?url=a%26b&amp;f=json\""));
        assert!(page.contains("&lt;b&gt;"));
        assert!(!page.contains("<b>"));
        // The trail climbs from this page.
        assert!(page.contains("<a href=\"../\">Home</a>"));
        assert!(page.contains("<a href=\"../cog\">Cog</a>"));
        // A templated link stays as it is.
        assert_eq!(html_href("http://h/cog/tiles/WebMercatorQuad/{z}/{x}/{y}"), "http://h/cog/tiles/WebMercatorQuad/{z}/{x}/{y}");
        assert_eq!(html_href("http://h/colorMaps/viridis"), "http://h/colorMaps/viridis?f=html");
    }
}
