//! What `parse_query` makes of `assets`, `asset_bidx` and `asset_as_band`, input by input.
//! The table pins the answers the `/stac` routes have always been given, so the code that
//! reads these parameters can be rearranged without any of them moving.

use super::parse_query;

/// The fields of the parsed query that the asset parameters fill, written on one line.
fn seen(query: &str) -> String {
    let pq = parse_query(Some(query));
    format!(
        "assets={:?} bidx={:?} exprs={:?} as_band={} top_bidx={:?} errors={:?}",
        pq.assets, pq.asset_bidx, pq.asset_exprs, pq.asset_as_band, pq.bidx, pq.errors
    )
}

/// A query and what is seen of it.
const PINS: &[(&str, &str)] = &[
    // Plain names.
    (
        "assets=red",
        "assets=[\"red\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red&assets=nir",
        "assets=[\"red\", \"nir\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red,nir",
        "assets=[\"red\", \"nir\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red,nir&assets=blue",
        "assets=[\"red\", \"nir\", \"blue\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red,,nir,",
        "assets=[\"red\", \"nir\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=",
        "assets=[] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=,",
        "assets=[] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets",
        "assets=[] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red&assets=red",
        "assets=[\"red\", \"red\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red,red",
        "assets=[\"red\", \"red\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=%20red%20",
        "assets=[\" red \"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red,%20nir",
        "assets=[\"red\", \" nir\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=my+asset",
        "assets=[\"my+asset\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=my%20asset",
        "assets=[\"my asset\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=my%2Basset",
        "assets=[\"my+asset\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=B04&assets=b04",
        "assets=[\"B04\", \"b04\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=r%C3%A9d",
        "assets=[\"réd\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red%2Cnir",
        "assets=[\"red\", \"nir\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red=1",
        "assets=[\"red=1\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    // Band indexes after a pipe.
    (
        "assets=visual|bidx=1",
        "assets=[\"visual\"] bidx=[(\"visual\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=3,2,1",
        "assets=[\"visual\"] bidx=[(\"visual\", [3, 2, 1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual%7Cbidx%3D3,2,1",
        "assets=[\"visual\"] bidx=[(\"visual\", [3, 2, 1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual%7Cbidx%3D3%2C2%2C1",
        "assets=[\"visual\"] bidx=[(\"visual\", [3, 2, 1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual%7cbidx=1",
        "assets=[\"visual\"] bidx=[(\"visual\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|indexes=1,2",
        "assets=[\"visual\"] bidx=[(\"visual\", [1, 2])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=1|indexes=2",
        "assets=[\"visual\"] bidx=[(\"visual\", [1, 2])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|indexes=2|bidx=1",
        "assets=[\"visual\"] bidx=[(\"visual\", [2, 1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=1|bidx=2",
        "assets=[\"visual\"] bidx=[(\"visual\", [1, 2])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=1,1",
        "assets=[\"visual\"] bidx=[(\"visual\", [1, 1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=1&assets=visual|bidx=2",
        "assets=[\"visual\", \"visual\"] bidx=[(\"visual\", [1, 2])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=1&assets=red|bidx=1",
        "assets=[\"visual\", \"red\"] bidx=[(\"visual\", [1]), (\"red\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=3,1&assets=red",
        "assets=[\"visual\", \"red\"] bidx=[(\"visual\", [3, 1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red&assets=visual|bidx=3,1",
        "assets=[\"red\", \"visual\"] bidx=[(\"visual\", [3, 1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual&assets=visual|bidx=2",
        "assets=[\"visual\", \"visual\"] bidx=[(\"visual\", [2])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=2&assets=visual",
        "assets=[\"visual\", \"visual\"] bidx=[(\"visual\", [2])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual,red|bidx=1",
        "assets=[\"visual,red\"] bidx=[(\"visual,red\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=1,red",
        "assets=[\"visual\"] bidx=[(\"visual\", [1])] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=1,red: expected positive integer band indices in |bidx=\"]",
    ),
    // Band indexes nobody can read.
    (
        "assets=visual|bidx=",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=x",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=x: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=1,x",
        "assets=[\"visual\"] bidx=[(\"visual\", [1])] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=1,x: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=x,1",
        "assets=[\"visual\"] bidx=[(\"visual\", [1])] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=x,1: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=1,",
        "assets=[\"visual\"] bidx=[(\"visual\", [1])] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=1,: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=,1",
        "assets=[\"visual\"] bidx=[(\"visual\", [1])] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=,1: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=1,,2",
        "assets=[\"visual\"] bidx=[(\"visual\", [1, 2])] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=1,,2: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=0",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=0: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=0,1",
        "assets=[\"visual\"] bidx=[(\"visual\", [1])] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=0,1: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=-1",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=-1: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=1.0",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=1.0: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=1.5",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=1.5: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=65535",
        "assets=[\"visual\"] bidx=[(\"visual\", [65535])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=65536",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=65536: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=99999999999999999999",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=99999999999999999999: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=%2B1",
        "assets=[\"visual\"] bidx=[(\"visual\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=+1",
        "assets=[\"visual\"] bidx=[(\"visual\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=1e1",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=1e1: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=0x1",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=0x1: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=1_0",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=1_0: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=1=2",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=1=2: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=x|bidx=y",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=x|bidx=y: expected positive integer band indices in |bidx=\", \"invalid assets=visual|bidx=x|bidx=y: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=x,y,z",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=x,y,z: expected positive integer band indices in |bidx=\", \"invalid assets=visual|bidx=x,y,z: expected positive integer band indices in |bidx=\", \"invalid assets=visual|bidx=x,y,z: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|indexes=x",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|indexes=x: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|indexes=0",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|indexes=0: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=x&assets=red|bidx=y",
        "assets=[\"visual\", \"red\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=x: expected positive integer band indices in |bidx=\", \"invalid assets=red|bidx=y: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=x%20y",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=x y: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=%25",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=%: expected positive integer band indices in |bidx=\"]",
    ),
    // Space around things.
    (
        "assets=visual|bidx=%201",
        "assets=[\"visual\"] bidx=[(\"visual\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=1%20",
        "assets=[\"visual\"] bidx=[(\"visual\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=1,%202",
        "assets=[\"visual\"] bidx=[(\"visual\", [1, 2])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=1%20,2",
        "assets=[\"visual\"] bidx=[(\"visual\", [1, 2])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=1+,2",
        "assets=[\"visual\"] bidx=[(\"visual\", [2])] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=1 ,2: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=%09",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=\\t: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=%20visual%20|bidx=1",
        "assets=[\"visual\"] bidx=[(\"visual\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual%20|bidx=1&assets=visual|bidx=2",
        "assets=[\"visual\", \"visual\"] bidx=[(\"visual\", [1, 2])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|%20bidx=1",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx%20=1",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=1%20|expression=b1",
        "assets=[\"visual\"] bidx=[(\"visual\", [1])] exprs=[(\"visual\", \"b1\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=vis%20ual|bidx=1",
        "assets=[\"vis ual\"] bidx=[(\"vis ual\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    // Band math after a pipe.
    (
        "assets=red|expression=b1*2",
        "assets=[\"red\"] bidx=[] exprs=[(\"red\", \"b1*2\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red%7Cexpression%3Db1*2",
        "assets=[\"red\"] bidx=[] exprs=[(\"red\", \"b1*2\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|expression=b1+b2",
        "assets=[\"red\"] bidx=[] exprs=[(\"red\", \"b1+b2\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|expression=b1%2Bb2",
        "assets=[\"red\"] bidx=[] exprs=[(\"red\", \"b1+b2\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|expression=b1%20%2B%20b2",
        "assets=[\"red\"] bidx=[] exprs=[(\"red\", \"b1 + b2\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|expression=b1+%2B+b2",
        "assets=[\"red\"] bidx=[] exprs=[(\"red\", \"b1+++b2\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|expression=(b1-b2)/(b1+b2)",
        "assets=[\"red\"] bidx=[] exprs=[(\"red\", \"(b1-b2)/(b1+b2)\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|expression=",
        "assets=[\"red\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|expression",
        "assets=[\"red\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|expression=%20",
        "assets=[\"red\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|expression=%20b1*2%20",
        "assets=[\"red\"] bidx=[] exprs=[(\"red\", \"b1*2\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|expression=b1;b2",
        "assets=[\"red\"] bidx=[] exprs=[(\"red\", \"b1;b2\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|expression=b1,b2",
        "assets=[\"red\"] bidx=[] exprs=[(\"red\", \"b1,b2\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|expression=b1=b2",
        "assets=[\"red\"] bidx=[] exprs=[(\"red\", \"b1=b2\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|expression=b1%7Cb2",
        "assets=[\"red\"] bidx=[] exprs=[(\"red\", \"b1\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|expression=b1|expression=b2",
        "assets=[\"red\"] bidx=[] exprs=[(\"red\", \"b2\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|expression=b1|expression=",
        "assets=[\"red\"] bidx=[] exprs=[(\"red\", \"b1\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|expression=|expression=b2",
        "assets=[\"red\"] bidx=[] exprs=[(\"red\", \"b2\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|expression=b1&assets=red|expression=b2",
        "assets=[\"red\", \"red\"] bidx=[] exprs=[(\"red\", \"b2\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|expression=b1&assets=nir|expression=b1*2",
        "assets=[\"red\", \"nir\"] bidx=[] exprs=[(\"red\", \"b1\"), (\"nir\", \"b1*2\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|expression=b1&assets=red",
        "assets=[\"red\", \"red\"] bidx=[] exprs=[(\"red\", \"b1\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|Expression=b1",
        "assets=[\"red\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|expr=b1",
        "assets=[\"red\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red|expression=b1&expression=b1*3",
        "assets=[\"red\"] bidx=[] exprs=[(\"red\", \"b1\")] as_band=false top_bidx=[] errors=[]",
    ),
    // Both, in either order, and with what is not an option.
    (
        "assets=visual|bidx=1,2|expression=b1/b2",
        "assets=[\"visual\"] bidx=[(\"visual\", [1, 2])] exprs=[(\"visual\", \"b1/b2\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|expression=b1/b2|bidx=1,2",
        "assets=[\"visual\"] bidx=[(\"visual\", [1, 2])] exprs=[(\"visual\", \"b1/b2\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=x|expression=b1",
        "assets=[\"visual\"] bidx=[] exprs=[(\"visual\", \"b1\")] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=x|expression=b1: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|expression=b1|bidx=x",
        "assets=[\"visual\"] bidx=[] exprs=[(\"visual\", \"b1\")] as_band=false top_bidx=[] errors=[\"invalid assets=visual|expression=b1|bidx=x: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=visual|bidx=1&assets=visual|expression=b1",
        "assets=[\"visual\", \"visual\"] bidx=[(\"visual\", [1])] exprs=[(\"visual\", \"b1\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|nope=1",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|nope",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual||",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual||bidx=1",
        "assets=[\"visual\"] bidx=[(\"visual\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=1|",
        "assets=[\"visual\"] bidx=[(\"visual\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|nope=1|bidx=2|bands=x|expression=b1",
        "assets=[\"visual\"] bidx=[(\"visual\", [2])] exprs=[(\"visual\", \"b1\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|BIDX=1",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidxs=1",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|=1",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|rescale=0,1",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    // No name.
    (
        "assets=|bidx=1",
        "assets=[] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=%7Cbidx%3D1",
        "assets=[] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=|bidx=x",
        "assets=[] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=|bidx=x: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=|expression=b1",
        "assets=[] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=%20|bidx=1",
        "assets=[] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=|",
        "assets=[] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=||",
        "assets=[] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=|bidx=1&assets=red",
        "assets=[\"red\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    // `asset_bidx`.
    (
        "asset_bidx=red|1",
        "assets=[\"red\"] bidx=[(\"red\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "asset_bidx=red%7C1",
        "assets=[\"red\"] bidx=[(\"red\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "asset_bidx=visual|3,2,1",
        "assets=[\"visual\"] bidx=[(\"visual\", [3, 2, 1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "asset_bidx=visual|1&asset_bidx=visual|2",
        "assets=[\"visual\"] bidx=[(\"visual\", [1, 2])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "asset_bidx=visual|1&asset_bidx=red|1",
        "assets=[\"visual\", \"red\"] bidx=[(\"visual\", [1]), (\"red\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "asset_bidx=visual",
        "assets=[] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid asset_bidx=visual: expected asset_bidx=name|1,2,3\"]",
    ),
    (
        "asset_bidx=",
        "assets=[] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid asset_bidx=: expected asset_bidx=name|1,2,3\"]",
    ),
    (
        "asset_bidx",
        "assets=[] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "asset_bidx=visual|",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid asset_bidx=visual|: expected positive integer band indices in asset_bidx\"]",
    ),
    (
        "asset_bidx=|1",
        "assets=[] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "asset_bidx=|x",
        "assets=[] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid asset_bidx=|x: expected positive integer band indices in asset_bidx\"]",
    ),
    (
        "asset_bidx=%20|1",
        "assets=[] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "asset_bidx=visual|x",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid asset_bidx=visual|x: expected positive integer band indices in asset_bidx\"]",
    ),
    (
        "asset_bidx=visual|1,x,2",
        "assets=[\"visual\"] bidx=[(\"visual\", [1, 2])] exprs=[] as_band=false top_bidx=[] errors=[\"invalid asset_bidx=visual|1,x,2: expected positive integer band indices in asset_bidx\"]",
    ),
    (
        "asset_bidx=visual|0",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid asset_bidx=visual|0: expected positive integer band indices in asset_bidx\"]",
    ),
    (
        "asset_bidx=visual|-1",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid asset_bidx=visual|-1: expected positive integer band indices in asset_bidx\"]",
    ),
    (
        "asset_bidx=visual|1,",
        "assets=[\"visual\"] bidx=[(\"visual\", [1])] exprs=[] as_band=false top_bidx=[] errors=[\"invalid asset_bidx=visual|1,: expected positive integer band indices in asset_bidx\"]",
    ),
    (
        "asset_bidx=visual|%201%20,%202",
        "assets=[\"visual\"] bidx=[(\"visual\", [1, 2])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "asset_bidx=visual|+1",
        "assets=[\"visual\"] bidx=[(\"visual\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "asset_bidx=visual|%2B1",
        "assets=[\"visual\"] bidx=[(\"visual\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "asset_bidx=%20visual%20|1",
        "assets=[\"visual\"] bidx=[(\"visual\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "asset_bidx=vis+ual|1",
        "assets=[\"vis ual\"] bidx=[(\"vis ual\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "asset_bidx=visual|1|2",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid asset_bidx=visual|1|2: expected positive integer band indices in asset_bidx\"]",
    ),
    (
        "asset_bidx=visual|bidx=1",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid asset_bidx=visual|bidx=1: expected positive integer band indices in asset_bidx\"]",
    ),
    (
        "asset_bidx=visual,red|1",
        "assets=[\"visual,red\"] bidx=[(\"visual,red\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    // The two together, in every order.
    (
        "assets=red&asset_bidx=red|1",
        "assets=[\"red\"] bidx=[(\"red\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "asset_bidx=red|1&assets=red",
        "assets=[\"red\", \"red\"] bidx=[(\"red\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red&asset_bidx=nir|1",
        "assets=[\"red\", \"nir\"] bidx=[(\"nir\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "asset_bidx=nir|1&assets=red",
        "assets=[\"nir\", \"red\"] bidx=[(\"nir\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red,nir&asset_bidx=nir|1&asset_bidx=red|2",
        "assets=[\"red\", \"nir\"] bidx=[(\"nir\", [1]), (\"red\", [2])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=1&asset_bidx=visual|2",
        "assets=[\"visual\"] bidx=[(\"visual\", [1, 2])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "asset_bidx=visual|2&assets=visual|bidx=1",
        "assets=[\"visual\", \"visual\"] bidx=[(\"visual\", [2, 1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "asset_bidx=visual|2&assets=visual|bidx=1&asset_bidx=visual|3",
        "assets=[\"visual\", \"visual\"] bidx=[(\"visual\", [2, 1, 3])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|expression=b1&asset_bidx=visual|2",
        "assets=[\"visual\"] bidx=[(\"visual\", [2])] exprs=[(\"visual\", \"b1\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=visual|bidx=x&asset_bidx=visual|y",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid assets=visual|bidx=x: expected positive integer band indices in |bidx=\", \"invalid asset_bidx=visual|y: expected positive integer band indices in asset_bidx\"]",
    ),
    (
        "asset_bidx=visual|y&assets=visual|bidx=x",
        "assets=[\"visual\", \"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid asset_bidx=visual|y: expected positive integer band indices in asset_bidx\", \"invalid assets=visual|bidx=x: expected positive integer band indices in |bidx=\"]",
    ),
    (
        "assets=%20red&asset_bidx=red|1",
        "assets=[\" red\", \"red\"] bidx=[(\"red\", [1])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red%20|bidx=1&asset_bidx=red|2",
        "assets=[\"red\"] bidx=[(\"red\", [1, 2])] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    // Beside the rest of a request.
    (
        "url=item.json&assets=visual|bidx=3,2,1&bidx=1",
        "assets=[\"visual\"] bidx=[(\"visual\", [3, 2, 1])] exprs=[] as_band=false top_bidx=[1] errors=[]",
    ),
    (
        "bidx=2&assets=visual|bidx=x&nodata=oops",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[2] errors=[\"invalid assets=visual|bidx=x: expected positive integer band indices in |bidx=\", \"invalid nodata=oops: expected a number, nan, inf, or -inf\"]",
    ),
    (
        "nodata=oops&assets=visual|bidx=x&asset_bidx=red",
        "assets=[\"visual\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid nodata=oops: expected a number, nan, inf, or -inf\", \"invalid assets=visual|bidx=x: expected positive integer band indices in |bidx=\", \"invalid asset_bidx=red: expected asset_bidx=name|1,2,3\"]",
    ),
    (
        "assets=red&asset_as_band=true",
        "assets=[\"red\"] bidx=[] exprs=[] as_band=true top_bidx=[] errors=[]",
    ),
    (
        "assets=red&asset_as_band=TRUE",
        "assets=[\"red\"] bidx=[] exprs=[] as_band=true top_bidx=[] errors=[]",
    ),
    (
        "assets=red&asset_as_band=1",
        "assets=[\"red\"] bidx=[] exprs=[] as_band=true top_bidx=[] errors=[]",
    ),
    (
        "assets=red&asset_as_band=yes",
        "assets=[\"red\"] bidx=[] exprs=[] as_band=true top_bidx=[] errors=[]",
    ),
    (
        "assets=red&asset_as_band=false",
        "assets=[\"red\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red&asset_as_band=true&asset_as_band=no",
        "assets=[\"red\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets=red&asset_as_band=perhaps",
        "assets=[\"red\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid asset_as_band=perhaps: expected boolean\"]",
    ),
    (
        "assets=red&asset_as_band=",
        "assets=[\"red\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[\"invalid asset_as_band=: expected boolean\"]",
    ),
    (
        "assets=red|bidx=1&assets=nir|bidx=1&asset_as_band=true&rescale=0,3000",
        "assets=[\"red\", \"nir\"] bidx=[(\"red\", [1]), (\"nir\", [1])] exprs=[] as_band=true top_bidx=[] errors=[]",
    ),
    (
        "assets=B04|expression=b1*0.0001&assets=B08|bidx=1&asset_bidx=B03|1&assets=B02,B01&assets=B04|indexes=2",
        "assets=[\"B04\", \"B08\", \"B03\", \"B02\", \"B01\", \"B04\"] bidx=[(\"B08\", [1]), (\"B03\", [1]), (\"B04\", [2])] exprs=[(\"B04\", \"b1*0.0001\")] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "Assets=red",
        "assets=[] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "assets[]=red",
        "assets=[] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
    (
        "&&assets=red&&",
        "assets=[\"red\"] bidx=[] exprs=[] as_band=false top_bidx=[] errors=[]",
    ),
];

#[test]
fn asset_parameters_are_read_as_they_always_were() {
    assert_eq!(PINS.len(), 180);
    for (query, expected) in PINS {
        assert_eq!(seen(query), *expected, "{query}");
    }
}

#[test]
fn a_query_without_asset_parameters_leaves_their_fields_empty() {
    for query in ["", "url=a.tif", "url=a.tif&bidx=1,2&rescale=0,1&expression=b1%2Bb2", "bidx=x"] {
        let pq = parse_query(Some(query));
        assert!(pq.assets.is_empty() && pq.asset_bidx.is_empty() && pq.asset_exprs.is_empty(), "{query}");
        assert!(!pq.asset_as_band, "{query}");
    }
    let pq = parse_query(None);
    assert!(pq.assets.is_empty() && pq.asset_bidx.is_empty() && pq.asset_exprs.is_empty());
}
