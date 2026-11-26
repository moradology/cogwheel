//! The one reader of `assets`: `assets=red`, `assets=red,nir`, `assets=visual|bidx=3,2,1`,
//! `assets=red|expression=b1*2`.
//!
//! Two kinds of route take the parameter and they do not read it alike. [`read_assets`] is
//! the one pass over a value, and each [`AssetEntry`] it hands out holds both readings, so
//! that neither kind of route goes over the text by itself:
//!
//! - [`AssetEntry::into_spec`] is TiTiler's reading, an [`AssetSpec`]. [`AssetOptions`]
//!   gathers them: one for each entry, in the order written, a name given twice being two
//!   specs. It is exact about what an option is, and what it cannot take is in
//!   [`AssetOptions::errors`] ([`AssetOptionError::titiler`]), for a route that answers
//!   such a request with a 422.
//! - [`AssetEntry::into_loose`] is the reading the `/stac` routes have always had, which
//!   `parse_query` gathers by name into its `assets`, `asset_bidx` and `asset_exprs`. It
//!   forgives more and complains only about a band index it cannot read
//!   ([`AssetOptionError::loose`]).
//!
//! Where the two differ:
//!
//! | written                         | TiTiler's reading                 | the looser reading             |
//! |---------------------------------|-----------------------------------|--------------------------------|
//! | `a\|indexes=1`                  | not an option: passed over        | the same as `bidx=1`           |
//! | `a\|bidx=1\|bidx=2`             | the last one: `[2]`               | all of them: `[1, 2]`          |
//! | `a\|bidx= 1`                    | refused                           | `[1]`                          |
//! | `a\|bidx=1,x`                   | refused                           | `[1]`, and a complaint         |
//! | `a\|indexes=x`                  | passed over                       | a complaint                    |
//! | `a\|bidx` (no `=`), `a\|`       | refused                           | passed over                    |
//! | ` a \|bidx=1`                   | the name is ` a `                 | the name is `a`                |
//! | `\|bidx=1`                      | refused: no name                  | the entry is left out          |
//! | `a\|expression= b1 `            | ` b1 `                            | `b1`                           |
//! | `a\|expression=`                | an empty expression               | no expression                  |
//! | `a\|expression=b1\|expression=` | the last one: empty               | the last with anything in it   |
//! | `a\|bidx=1` and `a\|bidx=2`     | two specs                         | one asset's bands, `[1, 2]`    |
//!
//! Both read text that is already percent-decoded, and it is the caller that decides what a
//! `+` in the query was: `parse_query` keeps it as a plus, so that `assets=a|expression=b1+b2`
//! is a sum, where a form's decoding makes it a space.
//!
//! `parse_query` runs for every request, so the pass allocates nothing of its own: an entry
//! borrows from the value, and the one list it builds, the band indexes, is given away to
//! whichever reading is taken.

use std::ops::Range;

/// One entry of `assets`: a name and, after a `|`, TiTiler's options for that asset.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct AssetSpec {
    /// The name as written, with any space around it.
    pub(super) name: String,
    /// `bidx=1,2`: the bands to take, counted from one. Of several `bidx=` the last; `None`
    /// also when that one could not be read, which [`AssetOptions::errors`] then says.
    pub(super) indexes: Option<Vec<u16>>,
    /// `expression=b1*2`: band math over this asset's own bands, as written. It wins over
    /// `bidx`.
    pub(super) expression: Option<String>,
}

/// Why an entry of `assets` cannot be taken as written.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum AssetOptionErrorKind {
    /// Nothing before the first `|`.
    NoName,
    /// An option that is not `key=value`.
    NoValue,
    /// A band index that is not a whole number from one up.
    BadIndex {
        /// Everything after the option's `=`.
        value: String,
        /// The piece of it that is not a band index.
        token: String,
        /// Whether it is one once the space around it is taken off.
        padded: bool,
    },
}

/// What is wrong with one option of one entry of `assets`.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct AssetOptionError {
    /// Which entry: its place in [`AssetOptions::specs`].
    pub(super) spec: usize,
    /// That entry's name, as written.
    pub(super) asset: String,
    /// The option's key (`bidx`, `indexes`), the whole option when it has no `=`, and
    /// nothing when it is the name that is missing.
    pub(super) option: String,
    pub(super) kind: AssetOptionErrorKind,
}

impl AssetOptionError {
    /// Whether TiTiler's reading refuses this: a missing name, an option without a value,
    /// and anything in `bidx=` that is not exactly a band index. What is wrong inside
    /// `indexes=` is nothing to it, that not being one of its options.
    pub(super) fn titiler(&self) -> bool {
        match self.kind {
            AssetOptionErrorKind::NoName | AssetOptionErrorKind::NoValue => true,
            AssetOptionErrorKind::BadIndex { .. } => self.option == "bidx",
        }
    }

    /// Whether the looser reading complains of this: only of a band index that is not one
    /// even without the space around it, in `bidx=` or `indexes=`.
    pub(super) fn loose(&self) -> bool {
        matches!(self.kind, AssetOptionErrorKind::BadIndex { padded: false, .. })
    }
}

/// An entry as the `/stac` routes read it.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct LooseSpec<'a> {
    /// The name, without the space around it when options follow. Empty when there is none.
    pub(super) name: &'a str,
    /// Every band index of every `bidx=` and `indexes=`, in order, space around each allowed
    /// and those that cannot be read left out.
    pub(super) indexes: Vec<u16>,
    /// The last `expression=` with anything in it, without the space around it.
    pub(super) expression: Option<&'a str>,
}

/// One entry of an `assets` value as [`read_assets`] found it, to be taken as one reading
/// or the other.
#[derive(Debug)]
pub(super) struct AssetEntry<'a> {
    name: &'a str,
    /// Whether it was written with a `|`.
    piped: bool,
    /// Every band index that could be read, of `bidx=` and `indexes=` alike.
    bands: Vec<u16>,
    /// Where among `bands` the last `bidx=` lies, when every piece of it was exact.
    last_bidx: Option<Range<usize>>,
    /// The last `expression=`, as written.
    expression: Option<&'a str>,
    /// The last `expression=` with anything in it, trimmed.
    loose_expression: Option<&'a str>,
}

impl<'a> AssetEntry<'a> {
    /// An entry with no options yet.
    fn named(name: &'a str, piped: bool) -> Self {
        AssetEntry { name, piped, bands: Vec::new(), last_bidx: None, expression: None, loose_expression: None }
    }

    /// TiTiler's reading of the entry.
    pub(super) fn into_spec(self) -> AssetSpec {
        let indexes = match self.last_bidx {
            Some(range) if range.len() == self.bands.len() => Some(self.bands),
            Some(range) => Some(self.bands[range].to_vec()),
            None => None,
        };
        AssetSpec { name: self.name.to_string(), indexes, expression: self.expression.map(str::to_string) }
    }

    /// The `/stac` routes' reading of the entry.
    pub(super) fn into_loose(self) -> LooseSpec<'a> {
        LooseSpec {
            name: if self.piped { self.name.trim() } else { self.name },
            indexes: self.bands,
            expression: self.loose_expression,
        }
    }
}

/// A band index: a whole number from one up.
pub(super) fn band_index(token: &str) -> Option<u16> {
    token.parse().ok().filter(|band| *band >= 1)
}

/// Read one `assets` value, already percent-decoded, handing each entry to `each` in the
/// order written. With a `|` in it the whole value is one entry and its commas belong to
/// the options; without, it may list several names with commas, and the empty ones among
/// them are nothing.
///
/// What cannot be taken as written is added to `errors`, before the entry it is about is
/// handed over; `first` is the place the value's first entry has in the caller's list, for
/// [`AssetOptionError::spec`].
pub(super) fn read_assets<'a>(
    value: &'a str,
    first: usize,
    errors: &mut Vec<AssetOptionError>,
    mut each: impl FnMut(AssetEntry<'a>),
) {
    if !value.contains('|') {
        for name in value.split(',').filter(|name| !name.is_empty()) {
            each(AssetEntry::named(name, false));
        }
        return;
    }
    let mut parts = value.split('|');
    let name = parts.next().unwrap_or("");
    let mut entry = AssetEntry::named(name, true);
    let complaint = |option: &str, kind| AssetOptionError {
        spec: first,
        asset: name.to_string(),
        option: option.to_string(),
        kind,
    };
    if name.is_empty() {
        errors.push(complaint("", AssetOptionErrorKind::NoName));
    }
    for option in parts {
        let Some((key, text)) = option.split_once('=') else {
            errors.push(complaint(option, AssetOptionErrorKind::NoValue));
            continue;
        };
        match key {
            // `indexes` is this server's other word for `bidx`, which TiTiler does not have.
            "bidx" | "indexes" => {
                let start = entry.bands.len();
                let mut exact = true;
                for token in text.split(',') {
                    let trimmed = token.trim();
                    let band = band_index(trimmed);
                    if let Some(band) = band {
                        entry.bands.push(band);
                    }
                    if band.is_none() || trimmed.len() != token.len() {
                        exact = false;
                        errors.push(complaint(
                            key,
                            AssetOptionErrorKind::BadIndex {
                                value: text.to_string(),
                                token: token.to_string(),
                                padded: band.is_some(),
                            },
                        ));
                    }
                }
                if key == "bidx" {
                    entry.last_bidx = exact.then_some(start..entry.bands.len());
                }
            }
            "expression" => {
                entry.expression = Some(text);
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    entry.loose_expression = Some(trimmed);
                }
            }
            _ => {}
        }
    }
    each(entry);
}

/// Every entry of the `assets` values read so far, in TiTiler's reading.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct AssetOptions {
    /// One spec for each entry, in the order written.
    pub(super) specs: Vec<AssetSpec>,
    /// What could not be taken as written, in the order met. Not every route minds every
    /// one: see [`AssetOptionError::titiler`] and [`AssetOptionError::loose`].
    pub(super) errors: Vec<AssetOptionError>,
}

impl AssetOptions {
    /// Read one more `assets` value, already percent-decoded.
    pub(super) fn read(&mut self, value: &str) {
        let specs = &mut self.specs;
        read_assets(value, specs.len(), &mut self.errors, |entry| specs.push(entry.into_spec()));
    }

    /// The first thing TiTiler's reading refuses, if there is one.
    pub(super) fn titiler_error(&self) -> Option<&AssetOptionError> {
        self.errors.iter().find(|error| error.titiler())
    }
}

#[cfg(test)]
mod tests {
    use super::super::{parse_query, percent_decode_opts};
    use super::*;

    fn spec(name: &str, indexes: Option<&[u16]>, expression: Option<&str>) -> AssetSpec {
        AssetSpec {
            name: name.to_string(),
            indexes: indexes.map(<[u16]>::to_vec),
            expression: expression.map(str::to_string),
        }
    }

    /// The options of a query, its `assets` values decoded as `parse_query` decodes them.
    fn read(query: &str) -> AssetOptions {
        let mut options = AssetOptions::default();
        for value in query.split('&').filter_map(|pair| pair.strip_prefix("assets=")) {
            options.read(&percent_decode_opts(value, false));
        }
        options
    }

    fn bad_index(spec: usize, asset: &str, option: &str, value: &str, token: &str, padded: bool) -> AssetOptionError {
        AssetOptionError {
            spec,
            asset: asset.to_string(),
            option: option.to_string(),
            kind: AssetOptionErrorKind::BadIndex { value: value.to_string(), token: token.to_string(), padded },
        }
    }

    #[test]
    fn each_entry_is_a_spec_in_the_order_written() {
        let options = read("assets=nir&assets=visual%7Cbidx%3D3,2,1&assets=red,blue&assets=B04%7Cexpression%3Db1*2");
        assert_eq!(
            options.specs,
            [
                spec("nir", None, None),
                spec("visual", Some(&[3, 2, 1]), None),
                spec("red", None, None),
                spec("blue", None, None),
                spec("B04", None, Some("b1*2")),
            ]
        );
        assert!(options.errors.is_empty());
        // Other parameters between them change nothing, and `asset_bidx` is not an entry.
        let options = read("assets=b&bidx=1&asset_bidx=z|1&url=x&assets=a&asset_as_band=true");
        assert_eq!(options.specs, [spec("b", None, None), spec("a", None, None)]);
        // No `assets`, no specs.
        assert_eq!(read("url=a.tif&bidx=1"), AssetOptions::default());
    }

    #[test]
    fn a_name_given_twice_is_two_specs() {
        let options = read("assets=visual|bidx=1&assets=red&assets=visual|bidx=2&assets=visual&assets=red,red");
        assert_eq!(
            options.specs,
            [
                spec("visual", Some(&[1]), None),
                spec("red", None, None),
                spec("visual", Some(&[2]), None),
                spec("visual", None, None),
                spec("red", None, None),
                spec("red", None, None),
            ]
        );
        // Each keeps its own options, where the fields of the query gather them by name.
        let options = read("assets=red|expression=b1&assets=red|expression=b2|bidx=4");
        assert_eq!(options.specs, [spec("red", None, Some("b1")), spec("red", Some(&[4]), Some("b2"))]);
    }

    #[test]
    fn both_options_are_kept_and_unknown_ones_passed_over() {
        // The expression winning over the bands is for whoever uses the spec.
        for query in ["assets=v|bidx=1,2|expression=b1/b2", "assets=v|expression=b1/b2|bidx=1,2"] {
            assert_eq!(read(query).specs, [spec("v", Some(&[1, 2]), Some("b1/b2"))], "{query}");
        }
        let options = read("assets=v|nope=1|bidx=2|bands=x|BIDX=9|expression=b1|rescale=0,1|=3");
        assert_eq!(options.specs, [spec("v", Some(&[2]), Some("b1"))]);
        assert!(options.errors.is_empty());
        // Only the first `=` parts the key from the value.
        assert_eq!(read("assets=v|expression=b1=b2").specs, [spec("v", None, Some("b1=b2"))]);
    }

    #[test]
    fn what_is_wrong_is_said_with_its_asset_and_option() {
        let options = read("assets=red&assets=visual|bidx=1,x,0|nope|bidx=%202&assets=|indexes=y&assets=nir|bidx=3");
        assert_eq!(
            options.errors,
            [
                bad_index(1, "visual", "bidx", "1,x,0", "x", false),
                bad_index(1, "visual", "bidx", "1,x,0", "0", false),
                AssetOptionError {
                    spec: 1,
                    asset: "visual".into(),
                    option: "nope".into(),
                    kind: AssetOptionErrorKind::NoValue,
                },
                bad_index(1, "visual", "bidx", " 2", " 2", true),
                AssetOptionError { spec: 2, asset: String::new(), option: String::new(), kind: AssetOptionErrorKind::NoName },
                bad_index(2, "", "indexes", "y", "y", false),
            ]
        );
        // Every entry still has its spec, so an error's place is a place in the list.
        assert_eq!(
            options.specs,
            [spec("red", None, None), spec("visual", None, None), spec("", None, None), spec("nir", Some(&[3]), None)]
        );
        let titiler: Vec<bool> = options.errors.iter().map(AssetOptionError::titiler).collect();
        let loose: Vec<bool> = options.errors.iter().map(AssetOptionError::loose).collect();
        assert_eq!(titiler, [true, true, true, true, true, false]);
        assert_eq!(loose, [true, true, false, false, false, true]);
        assert_eq!(options.titiler_error(), Some(&options.errors[0]));
        // What only the looser reading complains of is no error to TiTiler's.
        assert_eq!(read("assets=v|indexes=x").titiler_error(), None);
        assert_eq!(read("assets=v|bidx=1").titiler_error(), None);
    }

    #[test]
    fn numbers_that_are_not_band_indexes() {
        for good in ["1", "65535", "+1", "007"] {
            assert!(band_index(good).is_some(), "{good}");
        }
        for bad in ["", "0", "-1", "1.0", "65536", "1e1", "0x1", "1_0", " 1", "1 ", "x"] {
            assert_eq!(band_index(bad), None, "{bad}");
            let options = read(&format!("assets=v|bidx={}", bad.replace(' ', "%20")));
            assert_eq!(options.specs, [spec("v", None, None)], "{bad}");
            assert_eq!(options.errors.len(), 1, "{bad}");
            assert!(options.errors[0].titiler(), "{bad}");
        }
        // One complaint for each piece that cannot be read; the value is the whole list.
        assert_eq!(
            read("assets=v|bidx=1,,2,").errors,
            [bad_index(0, "v", "bidx", "1,,2,", "", false), bad_index(0, "v", "bidx", "1,,2,", "", false)]
        );
    }

    #[test]
    fn a_value_is_read_as_its_caller_decoded_it() {
        // `parse_query` keeps a `+` of the query as a plus.
        assert_eq!(read("assets=red|expression=b1+b2").specs, [spec("red", None, Some("b1+b2"))]);
        assert_eq!(read("assets=red|expression=b1%2Bb2").specs, [spec("red", None, Some("b1+b2"))]);
        // A route that decodes as a form does gives the space it got.
        let mut form = AssetOptions::default();
        form.read("red|expression=b1 b2");
        form.read("my asset");
        assert_eq!(form.specs, [spec("red", None, Some("b1 b2")), spec("my asset", None, None)]);
    }

    /// The fields `parse_query` fills for the `/stac` routes.
    type Fields = (Vec<String>, Vec<(String, Vec<u16>)>, Vec<(String, String)>, usize);

    fn fields(names: &[&str], bidx: &[(&str, &[u16])], exprs: &[(&str, &str)], errors: usize) -> Fields {
        (
            names.iter().map(|n| n.to_string()).collect(),
            bidx.iter().map(|(n, b)| (n.to_string(), b.to_vec())).collect(),
            exprs.iter().map(|(n, e)| (n.to_string(), e.to_string())).collect(),
            errors,
        )
    }

    /// Where the two readings part: what the query's fields get, what the list holds, and
    /// whether TiTiler's reading refuses the request.
    #[test]
    fn where_the_readings_differ_each_keeps_its_own() {
        let v = "visual";
        let cases: [(&str, Fields, Vec<AssetSpec>, bool); 16] = [
            // `indexes=` is an option only to the looser reading.
            ("assets=visual|indexes=1,2", fields(&[v], &[(v, &[1, 2])], &[], 0), vec![spec(v, None, None)], false),
            ("assets=visual|indexes=x", fields(&[v], &[], &[], 1), vec![spec(v, None, None)], false),
            (
                "assets=visual|bidx=1|indexes=2",
                fields(&[v], &[(v, &[1, 2])], &[], 0),
                vec![spec(v, Some(&[1]), None)],
                false,
            ),
            // Several `bidx=` on one entry: all of them, or the last.
            (
                "assets=visual|bidx=1|bidx=2",
                fields(&[v], &[(v, &[1, 2])], &[], 0),
                vec![spec(v, Some(&[2]), None)],
                false,
            ),
            // Space around a number.
            ("assets=visual|bidx=%201,2%20", fields(&[v], &[(v, &[1, 2])], &[], 0), vec![spec(v, None, None)], true),
            // A number that is none: the rest kept and a complaint, or the request refused.
            ("assets=visual|bidx=1,x", fields(&[v], &[(v, &[1])], &[], 1), vec![spec(v, None, None)], true),
            ("assets=visual|bidx=x", fields(&[v], &[], &[], 1), vec![spec(v, None, None)], true),
            // An option without a value.
            ("assets=visual|bidx", fields(&[v], &[], &[], 0), vec![spec(v, None, None)], true),
            ("assets=visual|", fields(&[v], &[], &[], 0), vec![spec(v, None, None)], true),
            // Space around a name with options, and no name at all.
            (
                "assets=%20visual%20|bidx=1",
                fields(&[v], &[(v, &[1])], &[], 0),
                vec![spec(" visual ", Some(&[1]), None)],
                false,
            ),
            ("assets=|bidx=1", fields(&[], &[], &[], 0), vec![spec("", Some(&[1]), None)], true),
            // Space around an expression, and an empty one.
            (
                "assets=visual|expression=%20b1%20",
                fields(&[v], &[], &[(v, "b1")], 0),
                vec![spec(v, None, Some(" b1 "))],
                false,
            ),
            ("assets=visual|expression=", fields(&[v], &[], &[], 0), vec![spec(v, None, Some(""))], false),
            (
                "assets=visual|expression=b1|expression=",
                fields(&[v], &[], &[(v, "b1")], 0),
                vec![spec(v, None, Some(""))],
                false,
            ),
            // A name given twice: its options gathered under the name, or two specs.
            (
                "assets=visual|bidx=1&assets=visual|bidx=2",
                fields(&[v, v], &[(v, &[1, 2])], &[], 0),
                vec![spec(v, Some(&[1]), None), spec(v, Some(&[2]), None)],
                false,
            ),
            (
                "assets=visual|expression=b1&assets=visual|expression=b2",
                fields(&[v, v], &[], &[(v, "b2")], 0),
                vec![spec(v, None, Some("b1")), spec(v, None, Some("b2"))],
                false,
            ),
        ];
        for (query, expected, specs, refused) in cases {
            let pq = parse_query(Some(query));
            let got = (pq.assets.clone(), pq.asset_bidx.clone(), pq.asset_exprs.clone(), pq.errors.len());
            assert_eq!(got, expected, "{query}");
            let options = read(query);
            assert_eq!(options.specs, specs, "{query}");
            assert_eq!(options.titiler_error().is_some(), refused, "{query}");
            // The query's complaints are the looser reading's, one for one.
            let loose = options.errors.iter().filter(|e| e.loose()).count();
            assert_eq!(loose, pq.errors.len(), "{query}");
        }
    }

    #[test]
    fn the_looser_reading_of_an_entry() {
        let mut errors = Vec::new();
        let mut loose = Vec::new();
        for value in [" a ", " b |indexes=2|bidx= 1,x|expression=|expression= b1 ", "c,,d", "|bidx=7"] {
            read_assets(value, loose.len(), &mut errors, |entry| loose.push(entry.into_loose()));
        }
        let entry = |name, indexes: &[u16], expression| LooseSpec { name, indexes: indexes.to_vec(), expression };
        assert_eq!(
            loose,
            [
                entry(" a ", &[], None),
                entry("b", &[2, 1], Some("b1")),
                entry("c", &[], None),
                entry("d", &[], None),
                entry("", &[7], None),
            ]
        );
        // The complaints say which entry they are about by its place in that list.
        let places: Vec<usize> = errors.iter().map(|e| e.spec).collect();
        assert_eq!(places, [1, 1, 4]);
    }

    /// One pass gives both readings: whichever is taken, the entries and the complaints
    /// are the same ones.
    #[test]
    fn both_readings_come_of_the_same_entries() {
        for value in ["red,nir", "v|bidx=1,2|expression=b1", " v |indexes=3|bidx=x", "|bidx=1", "v|", ""] {
            let (mut as_specs, mut as_loose) = (Vec::new(), Vec::new());
            let (mut spec_errors, mut loose_errors) = (Vec::new(), Vec::new());
            read_assets(value, 0, &mut spec_errors, |entry| as_specs.push(entry.into_spec()));
            read_assets(value, 0, &mut loose_errors, |entry| as_loose.push(entry.into_loose()));
            assert_eq!(spec_errors, loose_errors, "{value}");
            assert_eq!(as_specs.len(), as_loose.len(), "{value}");
            for (spec, loose) in as_specs.iter().zip(&as_loose) {
                assert_eq!(spec.name.trim(), loose.name.trim(), "{value}");
            }
        }
    }
}
