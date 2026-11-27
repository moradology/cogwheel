//! CRS resolution for Zarr arrays (#368).
//!
//! Kept in one self-contained module so that as GeoZarr conventions settle (`geo-proj`, `spatial`)
//! they can move in one piece.
//!
//! Resolution priority:
//! 1. CF `grid_mapping` structured attributes (`grid_mapping_name` + projection parameters) mapped
//!    directly onto `cogzilla_proj::CrsModel`.
//! 2. Explicit EPSG code from GeoZarr `proj:code` / `proj:epsg`, an `epsg` / `epsg_code`
//!    attribute, or a trailing `AUTHORITY["EPSG",...]` / `ID["EPSG",...]` scraped from
//!    `spatial_ref` / `crs_wkt`.
//! 3. Default to `EPSG:4326` (`CrsModel::Geographic`) ONLY when the store declares no CRS at all.
//!    A CRS that is named/present but cannot be modelled returns `ZarrError::UnsupportedCrs`
//!    rather than silently falling back to `EPSG:4326`.

use std::collections::BTreeMap;

use cogzilla_proj::{
    AlbersEqualArea, CrsModel, CylindricalEqualArea, Ellipsoid, EqualEarth,
    LambertAzimuthalEqualArea, LambertConformalConic, PolarStereographic, Sinusoidal,
    TransverseMercator,
};
use serde_json::Value;

use crate::{ArrayMeta, GroupView, ZarrError};

/// Resolved CRS for a Zarr array.
#[derive(Debug, Clone)]
pub struct ZarrCrs {
    pub model: CrsModel,
    /// EPSG code when known (`Some(4326)` for default geographic or explicit EPSG).
    pub epsg: Option<u16>,
    /// Human-readable description (e.g. `"EPSG:3857"` or `"CF:transverse_mercator"`).
    pub description: String,
}

/// Resolve the CRS for `meta` inside `group`.
pub fn resolve_crs(group: &GroupView<'_>, meta: &ArrayMeta) -> Result<ZarrCrs, ZarrError> {
    // Locate the CF grid_mapping variable if declared on the array (or fallback `spatial_ref` / `crs`).
    let gm_name = meta
        .attributes
        .get("grid_mapping")
        .and_then(Value::as_str)
        .map(|s| {
            // CF extended grid_mapping syntax: "crs_name: x y" -> take first token before ':'
            s.split(':').next().unwrap_or(s).trim().to_string()
        });

    let gm_attrs: Option<&BTreeMap<String, Value>> = if let Some(ref name) = gm_name {
        group.array(name).map(|a| &a.attributes)
    } else {
        group
            .array("spatial_ref")
            .or_else(|| group.array("crs"))
            .map(|a| &a.attributes)
    };

    // 1. CF `grid_mapping_name` (structured JSON parameters -> CrsModel)
    if let Some(attrs) = gm_attrs
        && let Some(gm_proj) = attrs.get("grid_mapping_name").and_then(Value::as_str)
    {
        return crs_from_cf_grid_mapping(gm_proj, attrs);
    }

    // 2. Explicit EPSG / GeoZarr `proj:code` / `proj:epsg` / WKT `AUTHORITY` on:
    //    (a) grid_mapping variable attributes, (b) array attributes, (c) group attributes.
    let attr_sources: [Option<&BTreeMap<String, Value>>; 3] =
        [gm_attrs, Some(&meta.attributes), Some(group.attributes())];

    for attrs in attr_sources.into_iter().flatten() {
        if let Some(crs) = try_crs_from_epsg_or_geozarr(attrs)? {
            return Ok(crs);
        }
    }

    for attrs in attr_sources.into_iter().flatten() {
        if let Some(wkt) = attrs
            .get("spatial_ref")
            .or_else(|| attrs.get("crs_wkt"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            if let Some(code) = extract_epsg_from_wkt(wkt) {
                let model = CrsModel::from_epsg(code).map_err(|e| ZarrError::UnsupportedCrs {
                    crs: format!("EPSG:{code}"),
                    detail: e.to_string(),
                })?;
                return Ok(ZarrCrs {
                    model,
                    epsg: Some(code),
                    description: format!("EPSG:{code}"),
                });
            }
            let snippet: String = wkt.chars().take(80).collect();
            return Err(ZarrError::UnsupportedCrs {
                crs: snippet,
                detail: "WKT CRS has no recognized trailing EPSG authority code".into(),
            });
        }
    }

    // If `grid_mapping` explicitly named a variable that does not exist in the group, refuse
    // rather than silently falling back to 4326.
    if let Some(name) = gm_name
        && gm_attrs.is_none()
    {
        return Err(ZarrError::UnsupportedCrs {
            crs: name,
            detail: "referenced CF grid_mapping variable was not found in group".into(),
        });
    }

    // 3. No CRS metadata declared anywhere -> default to EPSG:4326 (matching TiTiler).
    Ok(ZarrCrs {
        model: CrsModel::Geographic,
        epsg: Some(4326),
        description: "EPSG:4326".into(),
    })
}

fn crs_from_cf_grid_mapping(
    name: &str,
    attrs: &BTreeMap<String, Value>,
) -> Result<ZarrCrs, ZarrError> {
    // Check if an explicit EPSG code is also attached (for `epsg` reporting)
    let explicit_epsg = extract_explicit_epsg(attrs).or_else(|| {
        attrs
            .get("spatial_ref")
            .or_else(|| attrs.get("crs_wkt"))
            .and_then(Value::as_str)
            .and_then(extract_epsg_from_wkt)
    });

    let ellip = parse_cf_ellipsoid(attrs);
    let fe = attr_f64(attrs, "false_easting").unwrap_or(0.0);
    let fn_ = attr_f64(attrs, "false_northing").unwrap_or(0.0);

    let model = match name {
        "latitude_longitude" => {
            return Ok(ZarrCrs {
                model: CrsModel::Geographic,
                epsg: Some(explicit_epsg.unwrap_or(4326)),
                description: format!("EPSG:{}", explicit_epsg.unwrap_or(4326)),
            });
        }
        "transverse_mercator" => {
            let lon0 = require_attr_f64(attrs, "longitude_of_central_meridian", name)?;
            let lat0 = attr_f64(attrs, "latitude_of_projection_origin").unwrap_or(0.0);
            let k0 = attr_f64(attrs, "scale_factor_at_central_meridian").unwrap_or(1.0);
            CrsModel::TransverseMercator(TransverseMercator::new(ellip, lon0, lat0, k0, fe, fn_))
        }
        "albers_conical_equal_area" => {
            let (lat1, lat2) = require_two_parallels(attrs, name)?;
            let lat0 = attr_f64(attrs, "latitude_of_projection_origin").unwrap_or(0.0);
            let lon0 = require_attr_f64(attrs, "longitude_of_central_meridian", name)?;
            CrsModel::Albers(AlbersEqualArea::new(ellip, lat1, lat2, lat0, lon0, fe, fn_))
        }
        "lambert_conformal_conic" => {
            let lat0 = attr_f64(attrs, "latitude_of_projection_origin").unwrap_or(0.0);
            let lon0 = require_attr_f64(attrs, "longitude_of_central_meridian", name)?;
            let parallels = attr_f64_list(attrs, "standard_parallel");
            if parallels.len() >= 2 {
                CrsModel::LambertConformalConic(LambertConformalConic::new_2sp(
                    ellip,
                    parallels[0],
                    parallels[1],
                    lat0,
                    lon0,
                    fe,
                    fn_,
                ))
            } else {
                let lat_sp = parallels.first().copied().unwrap_or(lat0);
                let k0 = attr_f64(attrs, "scale_factor_at_projection_origin").unwrap_or(1.0);
                CrsModel::LambertConformalConic(LambertConformalConic::new_1sp(
                    ellip, lat_sp, lon0, k0, fe, fn_,
                ))
            }
        }
        "polar_stereographic" => {
            let lon0 = attr_f64(attrs, "straight_vertical_longitude_from_pole")
                .or_else(|| attr_f64(attrs, "longitude_of_projection_origin"))
                .unwrap_or(0.0);
            if let Some(lat_ts) = attr_f64(attrs, "standard_parallel") {
                CrsModel::PolarStereographic(PolarStereographic::new(ellip, lat_ts, lon0, fe, fn_))
            } else {
                let lat0 = attr_f64(attrs, "latitude_of_projection_origin").unwrap_or(90.0);
                let k0 = attr_f64(attrs, "scale_factor_at_projection_origin").unwrap_or(1.0);
                CrsModel::PolarStereographic(PolarStereographic::new_pole_scale(
                    ellip,
                    lat0 >= 0.0,
                    lon0,
                    k0,
                    fe,
                    fn_,
                ))
            }
        }
        "lambert_azimuthal_equal_area" => {
            let lat0 = require_attr_f64(attrs, "latitude_of_projection_origin", name)?;
            let lon0 = require_attr_f64(attrs, "longitude_of_projection_origin", name)?;
            CrsModel::Laea(LambertAzimuthalEqualArea::new(ellip, lat0, lon0, fe, fn_))
        }
        "sinusoidal" => {
            let lon0 = attr_f64(attrs, "longitude_of_projection_origin")
                .or_else(|| attr_f64(attrs, "longitude_of_central_meridian"))
                .unwrap_or(0.0);
            CrsModel::Sinusoidal(Sinusoidal::new(ellip, lon0, fe, fn_))
        }
        "lambert_cylindrical_equal_area" => {
            let lat_ts = attr_f64(attrs, "standard_parallel").unwrap_or(0.0);
            let lon0 = attr_f64(attrs, "longitude_of_central_meridian").unwrap_or(0.0);
            CrsModel::CylindricalEqualArea(CylindricalEqualArea::new(ellip, lat_ts, lon0, fe, fn_))
        }
        "equal_earth" => {
            let lon0 = attr_f64(attrs, "longitude_of_central_meridian").unwrap_or(0.0);
            CrsModel::EqualEarth(EqualEarth::new(ellip, lon0, fe, fn_))
        }
        "mercator" => {
            if explicit_epsg == Some(3857)
                || (attr_f64(attrs, "semi_major_axis") == Some(6_378_137.0)
                    && attr_f64(attrs, "semi_minor_axis") == Some(6_378_137.0))
            {
                CrsModel::Epsg3857
            } else {
                CrsModel::WorldMercator
            }
        }
        other => {
            return Err(ZarrError::UnsupportedCrs {
                crs: other.to_string(),
                detail: format!("CF grid_mapping_name {other:?} is not modelled by cogzilla-proj"),
            });
        }
    };

    let description = match explicit_epsg {
        Some(code) => format!("EPSG:{code}"),
        None => format!("CF:{name}"),
    };
    Ok(ZarrCrs {
        model,
        epsg: explicit_epsg,
        description,
    })
}

fn try_crs_from_epsg_or_geozarr(
    attrs: &BTreeMap<String, Value>,
) -> Result<Option<ZarrCrs>, ZarrError> {
    if let Some(code) = extract_explicit_epsg(attrs) {
        let model = CrsModel::from_epsg(code).map_err(|e| ZarrError::UnsupportedCrs {
            crs: format!("EPSG:{code}"),
            detail: e.to_string(),
        })?;
        return Ok(Some(ZarrCrs {
            model,
            epsg: Some(code),
            description: format!("EPSG:{code}"),
        }));
    }

    // Check for GeoZarr `proj:code` with a non-EPSG authority (refuse rather than silently fallback)
    if let Some(code_str) = attrs
        .get("proj:code")
        .or_else(|| attrs.get("geo-proj").and_then(|v| v.get("code")))
        .and_then(Value::as_str)
    {
        return Err(ZarrError::UnsupportedCrs {
            crs: code_str.to_string(),
            detail: "unsupported proj:code authority (expected EPSG:<code>)".into(),
        });
    }

    Ok(None)
}

fn extract_explicit_epsg(attrs: &BTreeMap<String, Value>) -> Option<u16> {
    // Direct numeric `epsg`, `epsg_code`, `proj:epsg`, or `geo-proj.epsg`
    for key in ["proj:epsg", "epsg", "epsg_code"] {
        if let Some(n) = attrs.get(key).and_then(Value::as_u64)
            && let Ok(code) = u16::try_from(n)
        {
            return Some(code);
        }
    }
    if let Some(gp) = attrs.get("geo-proj").and_then(Value::as_object) {
        if let Some(n) = gp.get("epsg").and_then(Value::as_u64)
            && let Ok(code) = u16::try_from(n)
        {
            return Some(code);
        }
        if let Some(s) = gp.get("code").and_then(Value::as_str)
            && let Some(code) = parse_epsg_str(s)
        {
            return Some(code);
        }
    }
    if let Some(s) = attrs.get("proj:code").and_then(Value::as_str)
        && let Some(code) = parse_epsg_str(s)
    {
        return Some(code);
    }
    None
}

fn parse_epsg_str(s: &str) -> Option<u16> {
    let trimmed = s.trim();
    let rest = trimmed
        .strip_prefix("EPSG:")
        .or_else(|| trimmed.strip_prefix("epsg:"))?;
    rest.trim().parse::<u16>().ok()
}

/// Deliberate shortcut (#368): scrape the trailing outer `AUTHORITY["EPSG","<code>"]` (WKT1) or
/// `ID["EPSG",<code>]` (WKT2) from a `spatial_ref` / `crs_wkt` string without a full WKT parser.
///
/// The outer CRS authority sits at the end of a WKT string, so scanning from the right finds the
/// top-level CRS code rather than an inner datum/ellipsoid/unit authority.
pub fn extract_epsg_from_wkt(wkt: &str) -> Option<u16> {
    // Only consider the trailing segment after the last nested block closes, or the final
    // AUTHORITY/ID occurrence in the WKT string.
    let upper = wkt.to_ascii_uppercase();
    for needle in ["AUTHORITY[\"EPSG\",\"", "ID[\"EPSG\","] {
        if let Some(pos) = upper.rfind(needle) {
            let after = &wkt[pos + needle.len()..];
            let digits: String = after
                .trim_start_matches(['"', ' '])
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(code) = digits.parse::<u16>() {
                // Reject inner-only unit/ellipsoid/datum codes (9001=metre, 7030=WGS84 ellip,
                // 6326=WGS84 datum, 8901=Greenwich, 9122=degree) if they happen to be last.
                if !matches!(code, 6326 | 7030 | 8901 | 9001 | 9002 | 9003 | 9102 | 9122) {
                    return Some(code);
                }
            }
        }
    }
    None
}

fn parse_cf_ellipsoid(attrs: &BTreeMap<String, Value>) -> Ellipsoid {
    let a = attr_f64(attrs, "semi_major_axis")
        .or_else(|| attr_f64(attrs, "earth_radius"))
        .unwrap_or(6_378_137.0);
    if let Some(inv_f) = attr_f64(attrs, "inverse_flattening") {
        return Ellipsoid::new(a, inv_f);
    }
    if let Some(b) = attr_f64(attrs, "semi_minor_axis") {
        if (a - b).abs() < 1e-9 {
            return Ellipsoid::new(a, 0.0);
        }
        let inv_f = a / (a - b);
        return Ellipsoid::new(a, inv_f);
    }
    Ellipsoid::new(a, 298.257_223_563)
}

fn attr_f64(attrs: &BTreeMap<String, Value>, key: &str) -> Option<f64> {
    attrs.get(key).and_then(Value::as_f64)
}

fn require_attr_f64(
    attrs: &BTreeMap<String, Value>,
    key: &'static str,
    proj: &str,
) -> Result<f64, ZarrError> {
    attr_f64(attrs, key).ok_or_else(|| ZarrError::UnsupportedCrs {
        crs: proj.to_string(),
        detail: format!("CF grid_mapping {proj:?} is missing required parameter {key:?}"),
    })
}

fn attr_f64_list(attrs: &BTreeMap<String, Value>, key: &str) -> Vec<f64> {
    match attrs.get(key) {
        Some(Value::Array(arr)) => arr.iter().filter_map(Value::as_f64).collect(),
        Some(Value::Number(n)) => n.as_f64().into_iter().collect(),
        _ => Vec::new(),
    }
}

fn require_two_parallels(
    attrs: &BTreeMap<String, Value>,
    proj: &str,
) -> Result<(f64, f64), ZarrError> {
    let list = attr_f64_list(attrs, "standard_parallel");
    match list.as_slice() {
        [p1, p2, ..] => Ok((*p1, *p2)),
        [p1] => Ok((*p1, *p1)),
        [] => Err(ZarrError::UnsupportedCrs {
            crs: proj.to_string(),
            detail: format!("CF grid_mapping {proj:?} is missing standard_parallel"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemoryStore, ZarrHierarchy};

    #[test]
    fn extracts_epsg_from_wkt1_and_wkt2() {
        let wkt1 = r#"PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AUTHORITY["EPSG","3857"]]"#;
        assert_eq!(extract_epsg_from_wkt(wkt1), Some(3857));

        let wkt2 = r#"PROJCRS["WGS 84 / UTM zone 10N",ID["EPSG",32610]]"#;
        assert_eq!(extract_epsg_from_wkt(wkt2), Some(32610));

        // Inner-only unit authority must NOT be misread as CRS EPSG
        let no_outer = r#"PROJCS["Custom",UNIT["metre",1,AUTHORITY["EPSG","9001"]]]"#;
        assert_eq!(extract_epsg_from_wkt(no_outer), None);
    }

    #[tokio::test]
    async fn geozarr_proj_and_unmodellable_crs_are_handled_explicitly() {
        // 1. GeoZarr `proj:code`
        let mut s1 = MemoryStore::new();
        let root1 = serde_json::json!({
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {},
            "consolidated_metadata": {
                "kind": "inline",
                "metadata": {
                    "band": {
                        "zarr_format": 3, "node_type": "array", "shape": [8, 8],
                        "data_type": "float32",
                        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [8, 8]}},
                        "codecs": ["bytes"], "fill_value": "NaN",
                        "attributes": {"proj:code": "EPSG:32610"}
                    }
                }
            }
        });
        s1.insert("zarr.json", serde_json::to_vec(&root1).unwrap());
        let h1 = ZarrHierarchy::open(&s1).await.unwrap();
        let g1 = h1.group(None).unwrap();
        let crs1 = resolve_crs(&g1, g1.array("band").unwrap()).unwrap();
        assert_eq!(crs1.epsg, Some(32610));
        assert!(matches!(crs1.model, CrsModel::TransverseMercator(_)));

        // 2. Unmodellable CF grid_mapping must return UnsupportedCrs (never silent 4326!)
        let mut s2 = MemoryStore::new();
        let root2 = serde_json::json!({
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {},
            "consolidated_metadata": {
                "kind": "inline",
                "metadata": {
                    "gm": {
                        "zarr_format": 3, "node_type": "array", "shape": [],
                        "data_type": "int32",
                        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": []}},
                        "codecs": ["bytes"], "fill_value": 0,
                        "attributes": {"grid_mapping_name": "geostationary"}
                    },
                    "band": {
                        "zarr_format": 3, "node_type": "array", "shape": [8, 8],
                        "data_type": "float32",
                        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [8, 8]}},
                        "codecs": ["bytes"], "fill_value": "NaN",
                        "attributes": {"grid_mapping": "gm"}
                    }
                }
            }
        });
        s2.insert("zarr.json", serde_json::to_vec(&root2).unwrap());
        let h2 = ZarrHierarchy::open(&s2).await.unwrap();
        let g2 = h2.group(None).unwrap();
        let err = resolve_crs(&g2, g2.array("band").unwrap()).unwrap_err();
        assert!(
            matches!(&err, ZarrError::UnsupportedCrs { crs, .. } if crs == "geostationary"),
            "{err}"
        );
    }
}
