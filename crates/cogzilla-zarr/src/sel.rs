//! `?sel=dim=value` DSL parser, coordinate index resolution, and CF time decoding (#371).

use std::collections::BTreeMap;

use serde_json::Value;

use crate::{ArrayMeta, ZarrError};

/// Selection method matching `xarray.Dataset.sel(method=...)` and `titiler.xarray`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelMethod {
    Exact,
    Nearest,
    /// `pad` / `ffill`: last valid coordinate <= target.
    Pad,
    /// `backfill` / `bfill`: first valid coordinate >= target.
    Backfill,
}

/// Parsed selector for one dimension (may hold multiple values if `?sel=dim=...` is repeated).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelDimSpec {
    pub dim: String,
    pub method: SelMethod,
    pub values: Vec<String>,
}

/// Parse repeatable `?sel=` query parameter values (e.g. `["time=nearest::2021-01-01"]`).
///
/// Rules matching `titiler.xarray`:
/// - `dim=value` selects by exact coordinate value.
/// - `dim=method::value` sets the method (`nearest`, `pad`, `ffill`, `backfill`, `bfill`).
/// - Repeating the same dimension selects multiple values.
/// - Specifying two different methods for the same dimension is a 400 error (`InvalidSel`).
pub fn parse_sel_params(params: &[String]) -> Result<Vec<SelDimSpec>, ZarrError> {
    let mut by_dim: BTreeMap<String, SelDimSpec> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();

    for raw in params {
        let Some((dim_raw, rhs)) = raw.split_once('=') else {
            return Err(ZarrError::InvalidSel {
                detail: format!("sel {raw:?} must have form 'dim=value' or 'dim=method::value'"),
            });
        };
        let dim = dim_raw.trim();
        if dim.is_empty() {
            return Err(ZarrError::InvalidSel {
                detail: format!("sel {raw:?} has an empty dimension name"),
            });
        }

        let (method, val_str) = if let Some((m_str, v_str)) = rhs.split_once("::") {
            let m = match m_str.trim().to_ascii_lowercase().as_str() {
                "nearest" => SelMethod::Nearest,
                "pad" | "ffill" => SelMethod::Pad,
                "backfill" | "bfill" => SelMethod::Backfill,
                other => {
                    return Err(ZarrError::InvalidSel {
                        detail: format!(
                            "unsupported sel method {other:?} on dimension {dim:?} (expected nearest, pad, ffill, backfill, bfill)"
                        ),
                    });
                }
            };
            (m, v_str.trim())
        } else {
            (SelMethod::Exact, rhs.trim())
        };

        if val_str.is_empty() {
            return Err(ZarrError::InvalidSel {
                detail: format!("sel {raw:?} has an empty value"),
            });
        }

        if let Some(existing) = by_dim.get_mut(dim) {
            if existing.method != method {
                return Err(ZarrError::InvalidSel {
                    detail: format!(
                        "conflicting sel methods {:?} and {:?} for dimension {dim:?}",
                        existing.method, method
                    ),
                });
            }
            existing.values.push(val_str.to_string());
        } else {
            order.push(dim.to_string());
            by_dim.insert(
                dim.to_string(),
                SelDimSpec {
                    dim: dim.to_string(),
                    method,
                    values: vec![val_str.to_string()],
                },
            );
        }
    }

    Ok(order
        .into_iter()
        .filter_map(|k| by_dim.remove(&k))
        .collect())
}

/// Supported CF calendars.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfCalendar {
    Standard,
    ProlepticGregorian,
    NoLeap,
    AllLeap,
    Days360,
}

impl CfCalendar {
    pub fn parse(name: Option<&str>) -> Result<Self, ZarrError> {
        match name.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            None | Some("standard") | Some("gregorian") => Ok(Self::Standard),
            Some("proleptic_gregorian") => Ok(Self::ProlepticGregorian),
            Some("noleap") | Some("365_day") => Ok(Self::NoLeap),
            Some("all_leap") | Some("366_day") => Ok(Self::AllLeap),
            Some("360_day") => Ok(Self::Days360),
            Some(other) => Err(ZarrError::UnsupportedCalendar {
                calendar: other.to_string(),
                detail: "supported CF calendars: standard, gregorian, proleptic_gregorian, noleap/365_day, all_leap/366_day, 360_day".into(),
            }),
        }
    }
}

/// Parsed CF time `units` (`"<step> since <ref_datetime>"`) + `calendar`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CfTimeUnits {
    pub step_seconds: f64,
    pub ref_days: i64,
    pub ref_sec_of_day: i64,
    pub calendar: CfCalendar,
}

impl CfTimeUnits {
    /// Inspect an array's CF `units` and `calendar` attributes; returns `Ok(None)` if not a CF
    /// time coordinate, or `Err` if `units` looks like CF time (`"... since ..."`) with an
    /// unsupported unit or calendar.
    pub fn from_meta(meta: &ArrayMeta) -> Result<Option<Self>, ZarrError> {
        let Some(units) = meta.attributes.get("units").and_then(Value::as_str) else {
            return Ok(None);
        };
        let Some((step_raw, since_raw)) = split_since(units) else {
            return Ok(None);
        };
        let step_seconds = match step_raw.trim().to_ascii_lowercase().as_str() {
            "seconds" | "second" | "secs" | "sec" | "s" => 1.0,
            "minutes" | "minute" | "mins" | "min" => 60.0,
            "hours" | "hour" | "hrs" | "hr" | "h" => 3600.0,
            "days" | "day" | "d" => 86400.0,
            other => {
                return Err(ZarrError::InvalidSel {
                    detail: format!("unsupported CF time unit step {other:?} in {units:?}"),
                });
            }
        };
        let cal_str = meta.attributes.get("calendar").and_then(Value::as_str);
        let calendar = CfCalendar::parse(cal_str)?;
        let dt = parse_iso_datetime(since_raw.trim())?;
        let ref_days = date_to_days(dt.year, dt.month, dt.day, calendar)?;
        let ref_sec_of_day = (dt.hour as i64) * 3600 + (dt.minute as i64) * 60 + (dt.second as i64);

        Ok(Some(Self {
            step_seconds,
            ref_days,
            ref_sec_of_day,
            calendar,
        }))
    }

    /// Decode a numeric CF time value to an ISO-8601 string (`YYYY-MM-DDTHH:MM:SS`).
    pub fn decode_to_iso(&self, value: f64) -> String {
        let total_secs = (value * self.step_seconds).round() as i64 + self.ref_sec_of_day;
        let delta_days = total_secs.div_euclid(86400);
        let sec_of_day = total_secs.rem_euclid(86400);
        let (y, m, d) = days_to_date(self.ref_days + delta_days, self.calendar);
        let hh = sec_of_day / 3600;
        let mm = (sec_of_day % 3600) / 60;
        let ss = sec_of_day % 60;
        format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}")
    }

    /// Encode an ISO-8601 date/datetime string back into numeric CF time units.
    pub fn encode_iso(&self, iso: &str) -> Result<f64, ZarrError> {
        let dt = parse_iso_datetime(iso)?;
        let days = date_to_days(dt.year, dt.month, dt.day, self.calendar)?;
        let sec_of_day = (dt.hour as i64) * 3600 + (dt.minute as i64) * 60 + (dt.second as i64);
        let delta_secs = (days - self.ref_days) * 86400 + (sec_of_day - self.ref_sec_of_day);
        Ok((delta_secs as f64) / self.step_seconds)
    }
}

/// Resolve a [`SelDimSpec`] against a 1-D coordinate array's values (with optional CF time units).
///
/// Out-of-range targets always return `ZarrError::SelOutOfRange` with the available `[min, max]`
/// range — never a silent clamp.
pub fn resolve_sel_indices(
    spec: &SelDimSpec,
    coords: &[f64],
    cf_time: Option<&CfTimeUnits>,
) -> Result<Vec<usize>, ZarrError> {
    if coords.is_empty() {
        return Err(ZarrError::InvalidSel {
            detail: format!("coordinate array for dimension {:?} is empty", spec.dim),
        });
    }

    let c_min = coords.iter().copied().fold(f64::INFINITY, f64::min);
    let c_max = coords.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let fmt_bound = |v: f64| match cf_time {
        Some(cf) => cf.decode_to_iso(v),
        None => format!("{v}"),
    };

    let mut indices = Vec::with_capacity(spec.values.len());
    for raw_val in &spec.values {
        let target = match cf_time {
            Some(cf) => cf.encode_iso(raw_val)?,
            None => raw_val.parse::<f64>().map_err(|_| ZarrError::InvalidSel {
                detail: format!(
                    "cannot parse sel value {raw_val:?} as number for dimension {:?}",
                    spec.dim
                ),
            })?,
        };

        let tol = (c_max - c_min).abs() * 1e-9 + 1e-6;
        if target < c_min - tol || target > c_max + tol {
            return Err(ZarrError::SelOutOfRange {
                dim: spec.dim.clone(),
                requested: raw_val.clone(),
                min: fmt_bound(c_min),
                max: fmt_bound(c_max),
            });
        }

        let idx = match spec.method {
            SelMethod::Exact => coords
                .iter()
                .position(|&v| (v - target).abs() <= tol)
                .ok_or_else(|| ZarrError::SelOutOfRange {
                    dim: spec.dim.clone(),
                    requested: raw_val.clone(),
                    min: fmt_bound(c_min),
                    max: fmt_bound(c_max),
                })?,
            SelMethod::Nearest => coords
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (*a - target)
                        .abs()
                        .total_cmp(&(*b - target).abs())
                })
                .map(|(i, _)| i)
                .unwrap(),
            SelMethod::Pad => coords
                .iter()
                .enumerate()
                .filter(|(_, v)| **v <= target + tol)
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .ok_or_else(|| ZarrError::SelOutOfRange {
                    dim: spec.dim.clone(),
                    requested: raw_val.clone(),
                    min: fmt_bound(c_min),
                    max: fmt_bound(c_max),
                })?,
            SelMethod::Backfill => coords
                .iter()
                .enumerate()
                .filter(|(_, v)| **v >= target - tol)
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .ok_or_else(|| ZarrError::SelOutOfRange {
                    dim: spec.dim.clone(),
                    requested: raw_val.clone(),
                    min: fmt_bound(c_min),
                    max: fmt_bound(c_max),
                })?,
        };
        indices.push(idx);
    }

    Ok(indices)
}

#[derive(Debug, Clone, Copy)]
struct SimpleDateTime {
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
}

fn split_since(units: &str) -> Option<(&str, &str)> {
    let lower = units.to_ascii_lowercase();
    let idx = lower.find(" since ")?;
    Some((&units[..idx], &units[idx + 7..]))
}

fn parse_iso_datetime(s: &str) -> Result<SimpleDateTime, ZarrError> {
    let trimmed = s.trim().trim_end_matches('Z').trim_end_matches('z');
    let (date_part, time_part) = match trimmed.split_once(['T', 't', ' ']) {
        Some((d, t)) => (d, Some(t.trim())),
        None => (trimmed, None),
    };
    let d_parts: Vec<&str> = date_part.split('-').collect();
    if d_parts.len() != 3 {
        return Err(ZarrError::InvalidSel {
            detail: format!("invalid ISO date {s:?} (expected YYYY-MM-DD[THH:MM:SS])"),
        });
    }
    let year: i32 = d_parts[0].parse().map_err(|_| ZarrError::InvalidSel {
        detail: format!("invalid year in {s:?}"),
    })?;
    let month: u32 = d_parts[1].parse().map_err(|_| ZarrError::InvalidSel {
        detail: format!("invalid month in {s:?}"),
    })?;
    let day: u32 = d_parts[2].parse().map_err(|_| ZarrError::InvalidSel {
        detail: format!("invalid day in {s:?}"),
    })?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return Err(ZarrError::InvalidSel {
            detail: format!("date out of range in {s:?}"),
        });
    }

    let (hour, minute, second) = if let Some(tp) = time_part {
        // Strip timezone offset if present (+00:00)
        let tp_no_tz = tp.split(['+', '-']).next().unwrap_or(tp);
        let t_parts: Vec<&str> = tp_no_tz.split(':').collect();
        let h: u32 = t_parts
            .first()
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| ZarrError::InvalidSel {
                detail: format!("invalid hour in {s:?}"),
            })?;
        let m: u32 = t_parts
            .get(1)
            .map(|v| v.parse().unwrap_or(0))
            .unwrap_or(0);
        let sec: u32 = t_parts
            .get(2)
            .map(|v| v.split('.').next().unwrap_or("0").parse().unwrap_or(0))
            .unwrap_or(0);
        (h, m, sec)
    } else {
        (0, 0, 0)
    };

    Ok(SimpleDateTime {
        year,
        month,
        day,
        hour,
        minute,
        second,
    })
}

const NOLEAP_MONTH_STARTS: [i64; 13] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365];
const ALLLEAP_MONTH_STARTS: [i64; 13] = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366];

fn date_to_days(y: i32, m: u32, d: u32, cal: CfCalendar) -> Result<i64, ZarrError> {
    match cal {
        CfCalendar::Standard | CfCalendar::ProlepticGregorian => {
            // Howard Hinnant's exact civil-to-days algorithm (days since 1970-01-01).
            let y_adj = (y as i64) - if m <= 2 { 1 } else { 0 };
            let era = y_adj.div_euclid(400);
            let yoe = y_adj.rem_euclid(400);
            let mp = (m as i64 + if m > 2 { -3 } else { 9 }) as u64;
            let doy = (153 * mp + 2) / 5 + (d as u64) - 1;
            let doe = (yoe as u64) * 365 + (yoe as u64) / 4 - (yoe as u64) / 100 + doy;
            Ok(era * 146_097 + (doe as i64) - 719_468)
        }
        CfCalendar::NoLeap => {
            Ok((y as i64) * 365 + NOLEAP_MONTH_STARTS[(m - 1) as usize] + (d as i64 - 1))
        }
        CfCalendar::AllLeap => {
            Ok((y as i64) * 366 + ALLLEAP_MONTH_STARTS[(m - 1) as usize] + (d as i64 - 1))
        }
        CfCalendar::Days360 => {
            if d > 30 {
                return Err(ZarrError::InvalidSel {
                    detail: format!("day {d} invalid in 360_day calendar"),
                });
            }
            Ok((y as i64) * 360 + ((m - 1) as i64) * 30 + (d as i64 - 1))
        }
    }
}

fn days_to_date(days: i64, cal: CfCalendar) -> (i32, u32, u32) {
    match cal {
        CfCalendar::Standard | CfCalendar::ProlepticGregorian => {
            // Howard Hinnant's exact days-to-civil algorithm.
            let z = days + 719_468;
            let era = z.div_euclid(146_097);
            let doe = z.rem_euclid(146_097) as u64;
            let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
            let y = (yoe as i64) + era * 400;
            let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
            let mp = (5 * doy + 2) / 153;
            let d = doy - (153 * mp + 2) / 5 + 1;
            let m = if mp < 10 { mp + 3 } else { mp - 9 };
            let year = y + if m <= 2 { 1 } else { 0 };
            (year as i32, m as u32, d as u32)
        }
        CfCalendar::NoLeap => {
            let y = days.div_euclid(365);
            let doy = days.rem_euclid(365);
            let m = NOLEAP_MONTH_STARTS.partition_point(|&s| s <= doy);
            let d = doy - NOLEAP_MONTH_STARTS[m - 1] + 1;
            (y as i32, m as u32, d as u32)
        }
        CfCalendar::AllLeap => {
            let y = days.div_euclid(366);
            let doy = days.rem_euclid(366);
            let m = ALLLEAP_MONTH_STARTS.partition_point(|&s| s <= doy);
            let d = doy - ALLLEAP_MONTH_STARTS[m - 1] + 1;
            (y as i32, m as u32, d as u32)
        }
        CfCalendar::Days360 => {
            let y = days.div_euclid(360);
            let doy = days.rem_euclid(360);
            let m = (doy / 30) + 1;
            let d = (doy % 30) + 1;
            (y as i32, m as u32, d as u32)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChunkKeyEncoding, DataType, Endianness, ZarrVersion};

    fn time_meta(units: &str, calendar: Option<&str>) -> ArrayMeta {
        let mut attributes = BTreeMap::new();
        attributes.insert("units".into(), Value::String(units.into()));
        if let Some(cal) = calendar {
            attributes.insert("calendar".into(), Value::String(cal.into()));
        }
        ArrayMeta {
            version: ZarrVersion::V3,
            shape: vec![6],
            chunk_shape: vec![6],
            dtype: DataType::Int64,
            endianness: Endianness::Little,
            fill_value: None,
            codecs: vec![],
            chunk_key: ChunkKeyEncoding::Default { separator: '/' },
            dimension_names: Some(vec![Some("time".into())]),
            attributes,
            order_f: false,
        }
    }

    #[test]
    fn sel_dsl_parses_methods_repeats_and_rejects_conflicts() {
        let parsed = parse_sel_params(&[
            "time=nearest::2021-01-15".into(),
            "time=nearest::2021-02-15".into(),
            "level=500".into(),
        ])
        .unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].dim, "time");
        assert_eq!(parsed[0].method, SelMethod::Nearest);
        assert_eq!(parsed[0].values, vec!["2021-01-15", "2021-02-15"]);
        assert_eq!(parsed[1].dim, "level");
        assert_eq!(parsed[1].method, SelMethod::Exact);

        let conflict = parse_sel_params(&[
            "time=nearest::2021-01-15".into(),
            "time=ffill::2021-02-15".into(),
        ])
        .unwrap_err();
        assert!(matches!(conflict, ZarrError::InvalidSel { .. }), "{conflict}");
    }

    #[test]
    fn cf_time_gregorian_and_noleap_round_trip_and_refuse_out_of_range() {
        // 2000 is a leap year in Gregorian (Feb has 29 days), NOT in noleap (Feb has 28 days).
        let cf_std = CfTimeUnits::from_meta(&time_meta("days since 2000-01-01", Some("standard")))
            .unwrap()
            .unwrap();
        assert_eq!(cf_std.decode_to_iso(59.0), "2000-02-29T00:00:00");
        assert_eq!(cf_std.decode_to_iso(60.0), "2000-03-01T00:00:00");

        let cf_noleap = CfTimeUnits::from_meta(&time_meta("days since 2000-01-01", Some("noleap")))
            .unwrap()
            .unwrap();
        // In noleap, day 59 is March 1 (Feb only has 28 days!)
        assert_eq!(cf_noleap.decode_to_iso(59.0), "2000-03-01T00:00:00");
        assert_eq!(cf_noleap.encode_iso("2000-03-01T00:00:00").unwrap(), 59.0);

        // Monthly noleap cube: [0, 31, 59, 90, 120, 151]
        let coords = [0.0, 31.0, 59.0, 90.0, 120.0, 151.0];
        // Exact match
        let exact = parse_sel_params(&["time=2000-03-01".into()]).unwrap();
        assert_eq!(
            resolve_sel_indices(&exact[0], &coords, Some(&cf_noleap)).unwrap(),
            vec![2]
        );
        // Nearest in between
        let near = parse_sel_params(&["time=nearest::2000-03-20".into()]).unwrap();
        assert_eq!(
            resolve_sel_indices(&near[0], &coords, Some(&cf_noleap)).unwrap(),
            vec![3] // closer to April 1 (day 90) than March 1 (day 59)
        );
        // Out of range MUST return SelOutOfRange (never clamp!)
        let oor = parse_sel_params(&["time=nearest::2001-01-01".into()]).unwrap();
        let err = resolve_sel_indices(&oor[0], &coords, Some(&cf_noleap)).unwrap_err();
        assert!(matches!(err, ZarrError::SelOutOfRange { .. }), "{err}");
    }
}
