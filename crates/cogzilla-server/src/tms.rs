//! All 13 OGC TileMatrixSet 2.0 definitions matching TiTiler 2.4 / `morecantile`.
//! Auto-generated from `morecantile` so every matrix's `scaleDenominator`, `cellSize`,
//! `pointOfOrigin`, `matrixWidth`, and `matrixHeight` matches TiTiler bit-for-bit.

use cogzilla_proj::CrsModel;
use cogzilla_tiff::Affine2D;

use crate::engine::EngineError;

#[derive(Debug, Clone, Copy)]
pub struct TileMatrixDef {
    pub id: &'static str,
    pub scale_denominator: f64,
    pub cell_size: f64,
    pub origin_x: f64,
    pub origin_y: f64,
    pub tile_width: u32,
    pub tile_height: u32,
    pub matrix_width: u64,
    pub matrix_height: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct TmsDef {
    pub id: &'static str,
    pub title: &'static str,
    pub crs_uri: &'static str,
    pub epsg: u16,
    pub ordered_axes: [&'static str; 2],
    pub invert_axis: bool,
    pub matrices: &'static [TileMatrixDef],
}

const MATRICES_CDB1GLOBALGRID: &[TileMatrixDef] = &[
    TileMatrixDef { id: "-10", scale_denominator: 397569609.9759771_f64, cell_size: 1.0_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1, tile_height: 1, matrix_width: 360, matrix_height: 180 },
    TileMatrixDef { id: "-9", scale_denominator: 198784804.98798856_f64, cell_size: 0.5_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 2, tile_height: 2, matrix_width: 360, matrix_height: 180 },
    TileMatrixDef { id: "-8", scale_denominator: 99392402.49399428_f64, cell_size: 0.25_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 4, tile_height: 4, matrix_width: 360, matrix_height: 180 },
    TileMatrixDef { id: "-7", scale_denominator: 49696201.24699714_f64, cell_size: 0.125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 8, tile_height: 8, matrix_width: 360, matrix_height: 180 },
    TileMatrixDef { id: "-6", scale_denominator: 24848100.62349857_f64, cell_size: 0.0625_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 16, tile_height: 16, matrix_width: 360, matrix_height: 180 },
    TileMatrixDef { id: "-5", scale_denominator: 12424050.311749285_f64, cell_size: 0.03125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 32, tile_height: 32, matrix_width: 360, matrix_height: 180 },
    TileMatrixDef { id: "-4", scale_denominator: 6212025.155874643_f64, cell_size: 0.015625_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 64, tile_height: 64, matrix_width: 360, matrix_height: 180 },
    TileMatrixDef { id: "-3", scale_denominator: 3106012.5779373213_f64, cell_size: 0.0078125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 128, tile_height: 128, matrix_width: 360, matrix_height: 180 },
    TileMatrixDef { id: "-2", scale_denominator: 1553006.2889686606_f64, cell_size: 0.00390625_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 360, matrix_height: 180 },
    TileMatrixDef { id: "-1", scale_denominator: 776503.1444843303_f64, cell_size: 0.001953125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 512, tile_height: 512, matrix_width: 360, matrix_height: 180 },
    TileMatrixDef { id: "0", scale_denominator: 388251.57224216516_f64, cell_size: 0.0009765625_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 360, matrix_height: 180 },
    TileMatrixDef { id: "1", scale_denominator: 194125.78612108258_f64, cell_size: 0.00048828125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 720, matrix_height: 360 },
    TileMatrixDef { id: "2", scale_denominator: 97062.89306054129_f64, cell_size: 0.000244140625_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 1440, matrix_height: 720 },
    TileMatrixDef { id: "3", scale_denominator: 48531.446530270645_f64, cell_size: 0.0001220703125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 2880, matrix_height: 1440 },
    TileMatrixDef { id: "4", scale_denominator: 24265.723265135322_f64, cell_size: 6.10351562e-05_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 5760, matrix_height: 2880 },
    TileMatrixDef { id: "5", scale_denominator: 12132.861632567661_f64, cell_size: 3.05175781e-05_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 11520, matrix_height: 5760 },
    TileMatrixDef { id: "6", scale_denominator: 6066.430816283831_f64, cell_size: 1.52587891e-05_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 23040, matrix_height: 11520 },
    TileMatrixDef { id: "7", scale_denominator: 3033.2154081419153_f64, cell_size: 7.6293945e-06_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 46080, matrix_height: 23040 },
    TileMatrixDef { id: "8", scale_denominator: 1516.6077040709577_f64, cell_size: 3.8146973e-06_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 92160, matrix_height: 46080 },
    TileMatrixDef { id: "9", scale_denominator: 758.3038520354788_f64, cell_size: 1.9073486e-06_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 184320, matrix_height: 92160 },
    TileMatrixDef { id: "10", scale_denominator: 379.1519260177394_f64, cell_size: 9.536743e-07_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 368640, matrix_height: 184320 },
    TileMatrixDef { id: "11", scale_denominator: 189.5759630088697_f64, cell_size: 4.768372e-07_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 737280, matrix_height: 368640 },
    TileMatrixDef { id: "12", scale_denominator: 94.7879815044349_f64, cell_size: 2.384186e-07_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 1474560, matrix_height: 737280 },
    TileMatrixDef { id: "13", scale_denominator: 47.3939907522174_f64, cell_size: 1.192093e-07_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 2949120, matrix_height: 1474560 },
    TileMatrixDef { id: "14", scale_denominator: 23.6969953761087_f64, cell_size: 5.96046e-08_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 5898240, matrix_height: 2949120 },
    TileMatrixDef { id: "15", scale_denominator: 11.8484976880544_f64, cell_size: 2.98023e-08_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 11796480, matrix_height: 5898240 },
    TileMatrixDef { id: "16", scale_denominator: 5.9242488440272_f64, cell_size: 1.49012e-08_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 23592960, matrix_height: 11796480 },
    TileMatrixDef { id: "17", scale_denominator: 2.9621244220136_f64, cell_size: 7.4506e-09_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 47185920, matrix_height: 23592960 },
    TileMatrixDef { id: "18", scale_denominator: 1.4810622110068_f64, cell_size: 3.7253e-09_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 94371840, matrix_height: 47185920 },
    TileMatrixDef { id: "19", scale_denominator: 0.7405311055034_f64, cell_size: 1.8626e-09_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 188743680, matrix_height: 94371840 },
    TileMatrixDef { id: "20", scale_denominator: 0.3702655527517_f64, cell_size: 9.313e-10_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 377487360, matrix_height: 188743680 },
    TileMatrixDef { id: "21", scale_denominator: 0.1851327763758_f64, cell_size: 4.657e-10_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 1024, tile_height: 1024, matrix_width: 754974720, matrix_height: 377487360 },
];

const MATRICES_CANADIANNAD83_LCC: &[TileMatrixDef] = &[
    TileMatrixDef { id: "0", scale_denominator: 145000000.0_f64, cell_size: 38364.6600626534_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 5, matrix_height: 5 },
    TileMatrixDef { id: "1", scale_denominator: 85000000.0_f64, cell_size: 22489.6283125899_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 8, matrix_height: 8 },
    TileMatrixDef { id: "2", scale_denominator: 50000000.0_f64, cell_size: 13229.1931250529_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 13, matrix_height: 14 },
    TileMatrixDef { id: "3", scale_denominator: 30000000.0_f64, cell_size: 7937.51587503175_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 21, matrix_height: 22 },
    TileMatrixDef { id: "4", scale_denominator: 17500000.0_f64, cell_size: 4630.21759376852_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 36, matrix_height: 38 },
    TileMatrixDef { id: "5", scale_denominator: 10000000.0_f64, cell_size: 2645.83862501058_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 62, matrix_height: 66 },
    TileMatrixDef { id: "6", scale_denominator: 6000000.0_f64, cell_size: 1587.50317500635_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 103, matrix_height: 110 },
    TileMatrixDef { id: "7", scale_denominator: 3500000.0_f64, cell_size: 926.043518753704_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 177, matrix_height: 188 },
    TileMatrixDef { id: "8", scale_denominator: 2000000.0_f64, cell_size: 529.167725002116_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 309, matrix_height: 329 },
    TileMatrixDef { id: "9", scale_denominator: 1200000.0_f64, cell_size: 317.50063500127_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 515, matrix_height: 548 },
    TileMatrixDef { id: "10", scale_denominator: 700000.0_f64, cell_size: 185.20870375074_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 882, matrix_height: 938 },
    TileMatrixDef { id: "11", scale_denominator: 420000.0_f64, cell_size: 111.125222250444_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 1470, matrix_height: 1563 },
    TileMatrixDef { id: "12", scale_denominator: 250000.0_f64, cell_size: 66.1459656252646_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 2469, matrix_height: 2626 },
    TileMatrixDef { id: "13", scale_denominator: 145000.0_f64, cell_size: 38.3646600626534_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 4257, matrix_height: 4528 },
    TileMatrixDef { id: "14", scale_denominator: 85000.0_f64, cell_size: 22.4896283125899_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 7262, matrix_height: 7723 },
    TileMatrixDef { id: "15", scale_denominator: 50000.0_f64, cell_size: 13.2291931250529_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 12344, matrix_height: 13130 },
    TileMatrixDef { id: "16", scale_denominator: 30000.0_f64, cell_size: 7.93751587503175_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 20574, matrix_height: 21882 },
    TileMatrixDef { id: "17", scale_denominator: 17500.0_f64, cell_size: 4.63021759376852_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 35269, matrix_height: 37512 },
    TileMatrixDef { id: "18", scale_denominator: 10000.0_f64, cell_size: 2.64583862501058_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 61720, matrix_height: 65646 },
    TileMatrixDef { id: "19", scale_denominator: 6000.0_f64, cell_size: 1.58750317500635_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 102866, matrix_height: 109409 },
    TileMatrixDef { id: "20", scale_denominator: 3500.0_f64, cell_size: 0.926043518753704_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 176341, matrix_height: 187558 },
    TileMatrixDef { id: "21", scale_denominator: 2000.0_f64, cell_size: 0.529167725002116_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 308596, matrix_height: 328227 },
    TileMatrixDef { id: "22", scale_denominator: 1200.0_f64, cell_size: 0.31750063500127_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 514327, matrix_height: 547044 },
    TileMatrixDef { id: "23", scale_denominator: 700.0_f64, cell_size: 0.18520870375074_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 881703, matrix_height: 937790 },
    TileMatrixDef { id: "24", scale_denominator: 420.0_f64, cell_size: 0.111125222250444_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 1469505, matrix_height: 1562983 },
    TileMatrixDef { id: "25", scale_denominator: 250.0_f64, cell_size: 0.0661459656252645_f64, origin_x: -34655800.0_f64, origin_y: 39310000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 2468768, matrix_height: 2625811 },
];

const MATRICES_EUROPEANETRS89_LAEAQUAD: &[TileMatrixDef] = &[
    TileMatrixDef { id: "0", scale_denominator: 62779017.8571428_f64, cell_size: 17578.125_f64, origin_x: 2000000.0_f64, origin_y: 5500000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 1, matrix_height: 1 },
    TileMatrixDef { id: "1", scale_denominator: 31389508.9285714_f64, cell_size: 8789.0625_f64, origin_x: 2000000.0_f64, origin_y: 5500000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 2, matrix_height: 2 },
    TileMatrixDef { id: "2", scale_denominator: 15694754.4642857_f64, cell_size: 4394.53125_f64, origin_x: 2000000.0_f64, origin_y: 5500000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 4, matrix_height: 4 },
    TileMatrixDef { id: "3", scale_denominator: 7847377.23214285_f64, cell_size: 2197.265625_f64, origin_x: 2000000.0_f64, origin_y: 5500000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 8, matrix_height: 8 },
    TileMatrixDef { id: "4", scale_denominator: 3923688.61607142_f64, cell_size: 1098.6328125_f64, origin_x: 2000000.0_f64, origin_y: 5500000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 16, matrix_height: 16 },
    TileMatrixDef { id: "5", scale_denominator: 1961844.30803571_f64, cell_size: 549.31640625_f64, origin_x: 2000000.0_f64, origin_y: 5500000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 32, matrix_height: 32 },
    TileMatrixDef { id: "6", scale_denominator: 980922.154017857_f64, cell_size: 274.658203125_f64, origin_x: 2000000.0_f64, origin_y: 5500000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 64, matrix_height: 64 },
    TileMatrixDef { id: "7", scale_denominator: 490461.077008928_f64, cell_size: 137.3291015625_f64, origin_x: 2000000.0_f64, origin_y: 5500000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 128, matrix_height: 128 },
    TileMatrixDef { id: "8", scale_denominator: 245230.538504464_f64, cell_size: 68.6645507812_f64, origin_x: 2000000.0_f64, origin_y: 5500000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 256, matrix_height: 256 },
    TileMatrixDef { id: "9", scale_denominator: 122615.269252232_f64, cell_size: 34.3322753906_f64, origin_x: 2000000.0_f64, origin_y: 5500000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 512, matrix_height: 512 },
    TileMatrixDef { id: "10", scale_denominator: 61307.634626116_f64, cell_size: 17.1661376953_f64, origin_x: 2000000.0_f64, origin_y: 5500000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 1024, matrix_height: 1024 },
    TileMatrixDef { id: "11", scale_denominator: 30653.817313058_f64, cell_size: 8.5830688477_f64, origin_x: 2000000.0_f64, origin_y: 5500000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 2048, matrix_height: 2048 },
    TileMatrixDef { id: "12", scale_denominator: 15326.908656529_f64, cell_size: 4.2915344238_f64, origin_x: 2000000.0_f64, origin_y: 5500000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 4096, matrix_height: 4096 },
    TileMatrixDef { id: "13", scale_denominator: 7663.45432826451_f64, cell_size: 2.1457672119_f64, origin_x: 2000000.0_f64, origin_y: 5500000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 8192, matrix_height: 8192 },
    TileMatrixDef { id: "14", scale_denominator: 3831.72716413225_f64, cell_size: 1.072883606_f64, origin_x: 2000000.0_f64, origin_y: 5500000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 16384, matrix_height: 16384 },
    TileMatrixDef { id: "15", scale_denominator: 1915.86358206612_f64, cell_size: 0.536441803_f64, origin_x: 2000000.0_f64, origin_y: 5500000.0_f64, tile_width: 256, tile_height: 256, matrix_width: 32768, matrix_height: 32768 },
];

const MATRICES_GNOSISGLOBALGRID: &[TileMatrixDef] = &[
    TileMatrixDef { id: "0", scale_denominator: 139770566.00717944_f64, cell_size: 0.3515625_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 4, matrix_height: 2 },
    TileMatrixDef { id: "1", scale_denominator: 69885283.00358972_f64, cell_size: 0.17578125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 8, matrix_height: 4 },
    TileMatrixDef { id: "2", scale_denominator: 34942641.50179486_f64, cell_size: 0.087890625_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 16, matrix_height: 8 },
    TileMatrixDef { id: "3", scale_denominator: 17471320.75089743_f64, cell_size: 0.0439453125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 32, matrix_height: 16 },
    TileMatrixDef { id: "4", scale_denominator: 8735660.375448715_f64, cell_size: 0.02197265625_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 64, matrix_height: 32 },
    TileMatrixDef { id: "5", scale_denominator: 4367830.1877243575_f64, cell_size: 0.010986328125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 128, matrix_height: 64 },
    TileMatrixDef { id: "6", scale_denominator: 2183915.0938621787_f64, cell_size: 0.0054931640625_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 256, matrix_height: 128 },
    TileMatrixDef { id: "7", scale_denominator: 1091957.5469310894_f64, cell_size: 0.0027465820312_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 512, matrix_height: 256 },
    TileMatrixDef { id: "8", scale_denominator: 545978.7734655447_f64, cell_size: 0.0013732910156_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 1024, matrix_height: 512 },
    TileMatrixDef { id: "9", scale_denominator: 272989.38673277234_f64, cell_size: 0.0006866455078_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 2048, matrix_height: 1024 },
    TileMatrixDef { id: "10", scale_denominator: 136494.69336638617_f64, cell_size: 0.0003433227539_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 4096, matrix_height: 2048 },
    TileMatrixDef { id: "11", scale_denominator: 68247.34668319309_f64, cell_size: 0.000171661377_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 8192, matrix_height: 4096 },
    TileMatrixDef { id: "12", scale_denominator: 34123.67334159654_f64, cell_size: 8.58306885e-05_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 16384, matrix_height: 8192 },
    TileMatrixDef { id: "13", scale_denominator: 17061.83667079827_f64, cell_size: 4.29153442e-05_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 32768, matrix_height: 16384 },
    TileMatrixDef { id: "14", scale_denominator: 8530.918335399136_f64, cell_size: 2.14576721e-05_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 65536, matrix_height: 32768 },
    TileMatrixDef { id: "15", scale_denominator: 4265.459167699568_f64, cell_size: 1.07288361e-05_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 131072, matrix_height: 65536 },
    TileMatrixDef { id: "16", scale_denominator: 2132.729583849784_f64, cell_size: 5.364418e-06_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 262144, matrix_height: 131072 },
    TileMatrixDef { id: "17", scale_denominator: 1066.364791924892_f64, cell_size: 2.682209e-06_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 524288, matrix_height: 262144 },
    TileMatrixDef { id: "18", scale_denominator: 533.182395962446_f64, cell_size: 1.3411045e-06_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 1048576, matrix_height: 524288 },
    TileMatrixDef { id: "19", scale_denominator: 266.591197981223_f64, cell_size: 6.705523e-07_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 2097152, matrix_height: 1048576 },
    TileMatrixDef { id: "20", scale_denominator: 133.2955989906115_f64, cell_size: 3.352761e-07_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 4194304, matrix_height: 2097152 },
    TileMatrixDef { id: "21", scale_denominator: 66.6477994953057_f64, cell_size: 1.676381e-07_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 8388608, matrix_height: 4194304 },
    TileMatrixDef { id: "22", scale_denominator: 33.3238997476529_f64, cell_size: 8.3819e-08_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 16777216, matrix_height: 8388608 },
    TileMatrixDef { id: "23", scale_denominator: 16.6619498738264_f64, cell_size: 4.19095e-08_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 33554432, matrix_height: 16777216 },
    TileMatrixDef { id: "24", scale_denominator: 8.3309749369132_f64, cell_size: 2.09548e-08_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 67108864, matrix_height: 33554432 },
    TileMatrixDef { id: "25", scale_denominator: 4.1654874684566_f64, cell_size: 1.04774e-08_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 134217728, matrix_height: 67108864 },
    TileMatrixDef { id: "26", scale_denominator: 2.0827437342283_f64, cell_size: 5.2387e-09_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 268435456, matrix_height: 134217728 },
    TileMatrixDef { id: "27", scale_denominator: 1.0413718671142_f64, cell_size: 2.6193e-09_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 536870912, matrix_height: 268435456 },
    TileMatrixDef { id: "28", scale_denominator: 0.5206859335571_f64, cell_size: 1.3097e-09_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 1073741824, matrix_height: 536870912 },
];

const MATRICES_LINZANTARTICAMAPTILEGRID: &[TileMatrixDef] = &[
    TileMatrixDef { id: "0", scale_denominator: 409600000.0_f64, cell_size: 114687.99999999999_f64, origin_x: -918457.73_f64, origin_y: 6918457.73_f64, tile_width: 256, tile_height: 256, matrix_width: 1, matrix_height: 1 },
    TileMatrixDef { id: "1", scale_denominator: 204800000.0_f64, cell_size: 57343.99999999999_f64, origin_x: -918457.73_f64, origin_y: 6918457.73_f64, tile_width: 256, tile_height: 256, matrix_width: 1, matrix_height: 1 },
    TileMatrixDef { id: "2", scale_denominator: 102400000.0_f64, cell_size: 28671.999999999996_f64, origin_x: -918457.73_f64, origin_y: 6918457.73_f64, tile_width: 256, tile_height: 256, matrix_width: 2, matrix_height: 2 },
    TileMatrixDef { id: "3", scale_denominator: 51200000.0_f64, cell_size: 14335.999999999998_f64, origin_x: -918457.73_f64, origin_y: 6918457.73_f64, tile_width: 256, tile_height: 256, matrix_width: 4, matrix_height: 4 },
    TileMatrixDef { id: "4", scale_denominator: 25600000.0_f64, cell_size: 7167.999999999999_f64, origin_x: -918457.73_f64, origin_y: 6918457.73_f64, tile_width: 256, tile_height: 256, matrix_width: 7, matrix_height: 7 },
    TileMatrixDef { id: "5", scale_denominator: 12800000.0_f64, cell_size: 3583.9999999999995_f64, origin_x: -918457.73_f64, origin_y: 6918457.73_f64, tile_width: 256, tile_height: 256, matrix_width: 13, matrix_height: 13 },
    TileMatrixDef { id: "6", scale_denominator: 6400000.0_f64, cell_size: 1791.9999999999998_f64, origin_x: -918457.73_f64, origin_y: 6918457.73_f64, tile_width: 256, tile_height: 256, matrix_width: 26, matrix_height: 26 },
    TileMatrixDef { id: "7", scale_denominator: 3200000.0_f64, cell_size: 895.9999999999999_f64, origin_x: -918457.73_f64, origin_y: 6918457.73_f64, tile_width: 256, tile_height: 256, matrix_width: 52, matrix_height: 52 },
    TileMatrixDef { id: "8", scale_denominator: 1600000.0_f64, cell_size: 447.99999999999994_f64, origin_x: -918457.73_f64, origin_y: 6918457.73_f64, tile_width: 256, tile_height: 256, matrix_width: 104, matrix_height: 104 },
    TileMatrixDef { id: "9", scale_denominator: 800000.0_f64, cell_size: 223.99999999999997_f64, origin_x: -918457.73_f64, origin_y: 6918457.73_f64, tile_width: 256, tile_height: 256, matrix_width: 207, matrix_height: 207 },
    TileMatrixDef { id: "10", scale_denominator: 400000.0_f64, cell_size: 111.99999999999999_f64, origin_x: -918457.73_f64, origin_y: 6918457.73_f64, tile_width: 256, tile_height: 256, matrix_width: 413, matrix_height: 413 },
    TileMatrixDef { id: "11", scale_denominator: 200000.0_f64, cell_size: 55.99999999999999_f64, origin_x: -918457.73_f64, origin_y: 6918457.73_f64, tile_width: 256, tile_height: 256, matrix_width: 826, matrix_height: 826 },
    TileMatrixDef { id: "12", scale_denominator: 100000.0_f64, cell_size: 27.999999999999996_f64, origin_x: -918457.73_f64, origin_y: 6918457.73_f64, tile_width: 256, tile_height: 256, matrix_width: 1652, matrix_height: 1652 },
    TileMatrixDef { id: "13", scale_denominator: 50000.0_f64, cell_size: 13.999999999999998_f64, origin_x: -918457.73_f64, origin_y: 6918457.73_f64, tile_width: 256, tile_height: 256, matrix_width: 3303, matrix_height: 3303 },
];

const MATRICES_NZTM2000QUAD: &[TileMatrixDef] = &[
    TileMatrixDef { id: "0", scale_denominator: 139770566.007179_f64, cell_size: 39135.75848201011_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 1, matrix_height: 1 },
    TileMatrixDef { id: "1", scale_denominator: 69885283.0035895_f64, cell_size: 19567.879241005056_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 2, matrix_height: 2 },
    TileMatrixDef { id: "2", scale_denominator: 34942641.50179475_f64, cell_size: 9783.939620502528_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 4, matrix_height: 4 },
    TileMatrixDef { id: "3", scale_denominator: 17471320.750897374_f64, cell_size: 4891.969810251264_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 8, matrix_height: 8 },
    TileMatrixDef { id: "4", scale_denominator: 8735660.375448687_f64, cell_size: 2445.984905125632_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 16, matrix_height: 16 },
    TileMatrixDef { id: "5", scale_denominator: 4367830.1877243435_f64, cell_size: 1222.992452562816_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 32, matrix_height: 32 },
    TileMatrixDef { id: "6", scale_denominator: 2183915.0938621718_f64, cell_size: 611.496226281408_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 64, matrix_height: 64 },
    TileMatrixDef { id: "7", scale_denominator: 1091957.5469310859_f64, cell_size: 305.748113140704_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 128, matrix_height: 128 },
    TileMatrixDef { id: "8", scale_denominator: 545978.7734655429_f64, cell_size: 152.874056570352_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 256, matrix_height: 256 },
    TileMatrixDef { id: "9", scale_denominator: 272989.38673277147_f64, cell_size: 76.437028285176_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 512, matrix_height: 512 },
    TileMatrixDef { id: "10", scale_denominator: 136494.69336638573_f64, cell_size: 38.218514142588_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 1024, matrix_height: 1024 },
    TileMatrixDef { id: "11", scale_denominator: 68247.34668319287_f64, cell_size: 19.109257071294_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 2048, matrix_height: 2048 },
    TileMatrixDef { id: "12", scale_denominator: 34123.67334159643_f64, cell_size: 9.554628535647_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 4096, matrix_height: 4096 },
    TileMatrixDef { id: "13", scale_denominator: 17061.836670798217_f64, cell_size: 4.7773142678235_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 8192, matrix_height: 8192 },
    TileMatrixDef { id: "14", scale_denominator: 8530.918335399108_f64, cell_size: 2.38865713391175_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 16384, matrix_height: 16384 },
    TileMatrixDef { id: "15", scale_denominator: 4265.459167699554_f64, cell_size: 1.194328566955875_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 32768, matrix_height: 32768 },
    TileMatrixDef { id: "16", scale_denominator: 2132.729583849777_f64, cell_size: 0.5971642834779375_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 65536, matrix_height: 65536 },
    TileMatrixDef { id: "17", scale_denominator: 1066.3647919248886_f64, cell_size: 0.29858214173896874_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 131072, matrix_height: 131072 },
    TileMatrixDef { id: "18", scale_denominator: 533.1823959624443_f64, cell_size: 0.14929107086948437_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 262144, matrix_height: 262144 },
    TileMatrixDef { id: "19", scale_denominator: 266.59119798122214_f64, cell_size: 0.07464553543474219_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 524288, matrix_height: 524288 },
    TileMatrixDef { id: "20", scale_denominator: 133.29559899061107_f64, cell_size: 0.03732276771737109_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 1048576, matrix_height: 1048576 },
    TileMatrixDef { id: "21", scale_denominator: 66.64779949530553_f64, cell_size: 0.018661383858685546_f64, origin_x: -3260586.7284_f64, origin_y: 10438190.1652_f64, tile_width: 256, tile_height: 256, matrix_width: 2097152, matrix_height: 2097152 },
];

const MATRICES_UPSANTARCTICWGS84QUAD: &[TileMatrixDef] = &[
    TileMatrixDef { id: "0", scale_denominator: 458726544.4_f64, cell_size: 128443.4324_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 1, matrix_height: 1 },
    TileMatrixDef { id: "1", scale_denominator: 229363272.2_f64, cell_size: 64221.71621_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 2, matrix_height: 2 },
    TileMatrixDef { id: "2", scale_denominator: 114681636.1_f64, cell_size: 32110.85811_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 4, matrix_height: 4 },
    TileMatrixDef { id: "3", scale_denominator: 57340818.05_f64, cell_size: 16055.42905_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 8, matrix_height: 8 },
    TileMatrixDef { id: "4", scale_denominator: 28670409.02_f64, cell_size: 8027.714526_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 16, matrix_height: 16 },
    TileMatrixDef { id: "5", scale_denominator: 14335204.51_f64, cell_size: 4013.857263_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 32, matrix_height: 32 },
    TileMatrixDef { id: "6", scale_denominator: 7167602.256_f64, cell_size: 2006.928632_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 64, matrix_height: 64 },
    TileMatrixDef { id: "7", scale_denominator: 3583801.128_f64, cell_size: 1003.464316_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 128, matrix_height: 128 },
    TileMatrixDef { id: "8", scale_denominator: 1791900.564_f64, cell_size: 501.7321579_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 256, matrix_height: 256 },
    TileMatrixDef { id: "9", scale_denominator: 895950.282_f64, cell_size: 250.866079_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 512, matrix_height: 512 },
    TileMatrixDef { id: "10", scale_denominator: 447975.141_f64, cell_size: 125.4330395_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 1024, matrix_height: 1024 },
    TileMatrixDef { id: "11", scale_denominator: 223987.5705_f64, cell_size: 62.71651974_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 2048, matrix_height: 2048 },
    TileMatrixDef { id: "12", scale_denominator: 111993.7852_f64, cell_size: 31.35825987_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 4096, matrix_height: 4096 },
    TileMatrixDef { id: "13", scale_denominator: 55996.89262_f64, cell_size: 15.67912993_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 8192, matrix_height: 8192 },
    TileMatrixDef { id: "14", scale_denominator: 27998.44631_f64, cell_size: 7.839564967_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 16384, matrix_height: 16384 },
    TileMatrixDef { id: "15", scale_denominator: 13999.22316_f64, cell_size: 3.919782484_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 32768, matrix_height: 32768 },
    TileMatrixDef { id: "16", scale_denominator: 6999.611578_f64, cell_size: 1.959891242_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 65536, matrix_height: 65536 },
    TileMatrixDef { id: "17", scale_denominator: 3499.805789_f64, cell_size: 0.979945621_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 131072, matrix_height: 131072 },
    TileMatrixDef { id: "18", scale_denominator: 1749.902894_f64, cell_size: 0.48997281_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 262144, matrix_height: 262144 },
    TileMatrixDef { id: "19", scale_denominator: 874.9514472_f64, cell_size: 0.244986405_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 524288, matrix_height: 524288 },
    TileMatrixDef { id: "20", scale_denominator: 437.4757236_f64, cell_size: 0.122493203_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 1048576, matrix_height: 1048576 },
    TileMatrixDef { id: "21", scale_denominator: 218.7378618_f64, cell_size: 0.061246601_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 2097152, matrix_height: 2097152 },
    TileMatrixDef { id: "22", scale_denominator: 109.3689309_f64, cell_size: 0.030623301_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 4194304, matrix_height: 4194304 },
    TileMatrixDef { id: "23", scale_denominator: 54.68446545_f64, cell_size: 0.01531165_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 8388608, matrix_height: 8388608 },
    TileMatrixDef { id: "24", scale_denominator: 27.34223273_f64, cell_size: 0.007655825_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 16777216, matrix_height: 16777216 },
];

const MATRICES_UPSARCTICWGS84QUAD: &[TileMatrixDef] = &[
    TileMatrixDef { id: "0", scale_denominator: 458726544.4_f64, cell_size: 128443.4324_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 1, matrix_height: 1 },
    TileMatrixDef { id: "1", scale_denominator: 229363272.2_f64, cell_size: 64221.71621_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 2, matrix_height: 2 },
    TileMatrixDef { id: "2", scale_denominator: 114681636.1_f64, cell_size: 32110.85811_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 4, matrix_height: 4 },
    TileMatrixDef { id: "3", scale_denominator: 57340818.05_f64, cell_size: 16055.42905_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 8, matrix_height: 8 },
    TileMatrixDef { id: "4", scale_denominator: 28670409.02_f64, cell_size: 8027.714526_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 16, matrix_height: 16 },
    TileMatrixDef { id: "5", scale_denominator: 14335204.51_f64, cell_size: 4013.857263_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 32, matrix_height: 32 },
    TileMatrixDef { id: "6", scale_denominator: 7167602.256_f64, cell_size: 2006.928632_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 64, matrix_height: 64 },
    TileMatrixDef { id: "7", scale_denominator: 3583801.128_f64, cell_size: 1003.464316_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 128, matrix_height: 128 },
    TileMatrixDef { id: "8", scale_denominator: 1791900.564_f64, cell_size: 501.7321579_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 256, matrix_height: 256 },
    TileMatrixDef { id: "9", scale_denominator: 895950.282_f64, cell_size: 250.866079_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 512, matrix_height: 512 },
    TileMatrixDef { id: "10", scale_denominator: 447975.141_f64, cell_size: 125.4330395_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 1024, matrix_height: 1024 },
    TileMatrixDef { id: "11", scale_denominator: 223987.5705_f64, cell_size: 62.71651974_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 2048, matrix_height: 2048 },
    TileMatrixDef { id: "12", scale_denominator: 111993.7852_f64, cell_size: 31.35825987_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 4096, matrix_height: 4096 },
    TileMatrixDef { id: "13", scale_denominator: 55996.89262_f64, cell_size: 15.67912993_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 8192, matrix_height: 8192 },
    TileMatrixDef { id: "14", scale_denominator: 27998.44631_f64, cell_size: 7.839564967_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 16384, matrix_height: 16384 },
    TileMatrixDef { id: "15", scale_denominator: 13999.22316_f64, cell_size: 3.919782484_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 32768, matrix_height: 32768 },
    TileMatrixDef { id: "16", scale_denominator: 6999.611578_f64, cell_size: 1.959891242_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 65536, matrix_height: 65536 },
    TileMatrixDef { id: "17", scale_denominator: 3499.805789_f64, cell_size: 0.979945621_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 131072, matrix_height: 131072 },
    TileMatrixDef { id: "18", scale_denominator: 1749.902894_f64, cell_size: 0.48997281_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 262144, matrix_height: 262144 },
    TileMatrixDef { id: "19", scale_denominator: 874.9514472_f64, cell_size: 0.244986405_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 524288, matrix_height: 524288 },
    TileMatrixDef { id: "20", scale_denominator: 437.4757236_f64, cell_size: 0.122493203_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 1048576, matrix_height: 1048576 },
    TileMatrixDef { id: "21", scale_denominator: 218.7378618_f64, cell_size: 0.061246601_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 2097152, matrix_height: 2097152 },
    TileMatrixDef { id: "22", scale_denominator: 109.3689309_f64, cell_size: 0.030623301_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 4194304, matrix_height: 4194304 },
    TileMatrixDef { id: "23", scale_denominator: 54.68446545_f64, cell_size: 0.01531165_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 8388608, matrix_height: 8388608 },
    TileMatrixDef { id: "24", scale_denominator: 27.34223273_f64, cell_size: 0.007655825_f64, origin_x: -14440759.350252_f64, origin_y: 18440759.350252_f64, tile_width: 256, tile_height: 256, matrix_width: 16777216, matrix_height: 16777216 },
];

const MATRICES_UTM31WGS84QUAD: &[TileMatrixDef] = &[
    TileMatrixDef { id: "1", scale_denominator: 279072704.500914_f64, cell_size: 78140.3572602559_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 1, matrix_height: 2 },
    TileMatrixDef { id: "2", scale_denominator: 139536352.250457_f64, cell_size: 39070.178630128_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 2, matrix_height: 4 },
    TileMatrixDef { id: "3", scale_denominator: 69768176.1252285_f64, cell_size: 19535.089315064_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 4, matrix_height: 8 },
    TileMatrixDef { id: "4", scale_denominator: 34884088.0626143_f64, cell_size: 9767.5446575319_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 8, matrix_height: 16 },
    TileMatrixDef { id: "5", scale_denominator: 17442044.0313071_f64, cell_size: 4883.772328766_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 16, matrix_height: 32 },
    TileMatrixDef { id: "6", scale_denominator: 8721022.01565356_f64, cell_size: 2441.886164383_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 32, matrix_height: 64 },
    TileMatrixDef { id: "7", scale_denominator: 4360511.00782678_f64, cell_size: 1220.9430821915_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 64, matrix_height: 128 },
    TileMatrixDef { id: "8", scale_denominator: 2180255.50391339_f64, cell_size: 610.471541095749_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 128, matrix_height: 256 },
    TileMatrixDef { id: "9", scale_denominator: 1090127.7519567_f64, cell_size: 305.235770547875_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 256, matrix_height: 512 },
    TileMatrixDef { id: "10", scale_denominator: 545063.875978348_f64, cell_size: 152.617885273937_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 512, matrix_height: 1024 },
    TileMatrixDef { id: "11", scale_denominator: 272531.937989174_f64, cell_size: 76.3089426369687_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 1024, matrix_height: 2048 },
    TileMatrixDef { id: "12", scale_denominator: 136265.968994587_f64, cell_size: 38.1544713184843_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 2048, matrix_height: 4096 },
    TileMatrixDef { id: "13", scale_denominator: 68132.9844972935_f64, cell_size: 19.0772356592422_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 4096, matrix_height: 8192 },
    TileMatrixDef { id: "14", scale_denominator: 34066.4922486467_f64, cell_size: 9.53861782962109_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 8192, matrix_height: 16384 },
    TileMatrixDef { id: "15", scale_denominator: 17033.2461243234_f64, cell_size: 4.76930891481054_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 16384, matrix_height: 32768 },
    TileMatrixDef { id: "16", scale_denominator: 8516.62306216168_f64, cell_size: 2.38465445740527_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 32768, matrix_height: 65536 },
    TileMatrixDef { id: "17", scale_denominator: 4258.31153108084_f64, cell_size: 1.19232722870264_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 65536, matrix_height: 131072 },
    TileMatrixDef { id: "18", scale_denominator: 2129.15576554042_f64, cell_size: 0.596163614351318_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 131072, matrix_height: 262144 },
    TileMatrixDef { id: "19", scale_denominator: 1064.57788277021_f64, cell_size: 0.298081807175659_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 262144, matrix_height: 524288 },
    TileMatrixDef { id: "20", scale_denominator: 532.288941385105_f64, cell_size: 0.149040903587829_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 524288, matrix_height: 1048576 },
    TileMatrixDef { id: "21", scale_denominator: 266.144470692553_f64, cell_size: 0.0745204517939147_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 1048576, matrix_height: 2097152 },
    TileMatrixDef { id: "22", scale_denominator: 133.072235346276_f64, cell_size: 0.0372602258969574_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 2097152, matrix_height: 4194304 },
    TileMatrixDef { id: "23", scale_denominator: 66.5361176731382_f64, cell_size: 0.0186301129484787_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 4194304, matrix_height: 8388608 },
    TileMatrixDef { id: "24", scale_denominator: 33.2680588365691_f64, cell_size: 0.00931505647423934_f64, origin_x: -9501965.72931276_f64, origin_y: 20003931.4586255_f64, tile_width: 256, tile_height: 256, matrix_width: 8388608, matrix_height: 16777216 },
];

const MATRICES_WGS1984QUAD: &[TileMatrixDef] = &[
    TileMatrixDef { id: "0", scale_denominator: 279541132.014358_f64, cell_size: 0.703125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 2, matrix_height: 1 },
    TileMatrixDef { id: "1", scale_denominator: 139770566.007179_f64, cell_size: 0.3515625_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 4, matrix_height: 2 },
    TileMatrixDef { id: "2", scale_denominator: 69885283.0035897_f64, cell_size: 0.17578125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 8, matrix_height: 4 },
    TileMatrixDef { id: "3", scale_denominator: 34942641.5017948_f64, cell_size: 0.087890625_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 16, matrix_height: 8 },
    TileMatrixDef { id: "4", scale_denominator: 17471320.7508974_f64, cell_size: 0.0439453125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 32, matrix_height: 16 },
    TileMatrixDef { id: "5", scale_denominator: 8735660.37544871_f64, cell_size: 0.02197265625_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 64, matrix_height: 32 },
    TileMatrixDef { id: "6", scale_denominator: 4367830.18772435_f64, cell_size: 0.010986328125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 128, matrix_height: 64 },
    TileMatrixDef { id: "7", scale_denominator: 2183915.09386217_f64, cell_size: 0.0054931640625_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 256, matrix_height: 128 },
    TileMatrixDef { id: "8", scale_denominator: 1091957.54693108_f64, cell_size: 0.00274658203125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 512, matrix_height: 256 },
    TileMatrixDef { id: "9", scale_denominator: 545978.773465544_f64, cell_size: 0.001373291015625_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 1024, matrix_height: 512 },
    TileMatrixDef { id: "10", scale_denominator: 272989.386732772_f64, cell_size: 0.0006866455078125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 2048, matrix_height: 1024 },
    TileMatrixDef { id: "11", scale_denominator: 136494.693366386_f64, cell_size: 0.00034332275390625_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 4096, matrix_height: 2048 },
    TileMatrixDef { id: "12", scale_denominator: 68247.346683193_f64, cell_size: 0.000171661376953125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 8192, matrix_height: 4096 },
    TileMatrixDef { id: "13", scale_denominator: 34123.6733415964_f64, cell_size: 8.58306884765625e-05_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 16384, matrix_height: 8192 },
    TileMatrixDef { id: "14", scale_denominator: 17061.8366707982_f64, cell_size: 4.29153442382812e-05_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 32768, matrix_height: 16384 },
    TileMatrixDef { id: "15", scale_denominator: 8530.91833539913_f64, cell_size: 2.14576721191406e-05_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 65536, matrix_height: 32768 },
    TileMatrixDef { id: "16", scale_denominator: 4265.45916769956_f64, cell_size: 1.07288360595703e-05_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 131072, matrix_height: 65536 },
    TileMatrixDef { id: "17", scale_denominator: 2132.72958384978_f64, cell_size: 5.36441802978515e-06_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 262144, matrix_height: 131072 },
    TileMatrixDef { id: "18", scale_denominator: 1066.36479192489_f64, cell_size: 2.68220901489258e-06_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 524288, matrix_height: 262144 },
    TileMatrixDef { id: "19", scale_denominator: 533.182395962445_f64, cell_size: 1.34110450744629e-06_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 1048576, matrix_height: 524288 },
    TileMatrixDef { id: "20", scale_denominator: 266.591197981222_f64, cell_size: 6.7055225372314e-07_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 2097152, matrix_height: 1048576 },
    TileMatrixDef { id: "21", scale_denominator: 133.295598990611_f64, cell_size: 3.3527612686157e-07_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 4194304, matrix_height: 2097152 },
    TileMatrixDef { id: "22", scale_denominator: 66.6477994953056_f64, cell_size: 1.6763806343079e-07_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 8388608, matrix_height: 4194304 },
    TileMatrixDef { id: "23", scale_denominator: 33.3238997476528_f64, cell_size: 8.381903171539e-08_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 16777216, matrix_height: 8388608 },
];

const MATRICES_WEBMERCATORQUAD: &[TileMatrixDef] = &[
    TileMatrixDef { id: "0", scale_denominator: 559082264.0287178_f64, cell_size: 156543.03392804097_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 1, matrix_height: 1 },
    TileMatrixDef { id: "1", scale_denominator: 279541132.0143589_f64, cell_size: 78271.51696402048_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 2, matrix_height: 2 },
    TileMatrixDef { id: "2", scale_denominator: 139770566.00717944_f64, cell_size: 39135.75848201024_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 4, matrix_height: 4 },
    TileMatrixDef { id: "3", scale_denominator: 69885283.00358972_f64, cell_size: 19567.87924100512_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 8, matrix_height: 8 },
    TileMatrixDef { id: "4", scale_denominator: 34942641.50179486_f64, cell_size: 9783.93962050256_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 16, matrix_height: 16 },
    TileMatrixDef { id: "5", scale_denominator: 17471320.75089743_f64, cell_size: 4891.96981025128_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 32, matrix_height: 32 },
    TileMatrixDef { id: "6", scale_denominator: 8735660.375448715_f64, cell_size: 2445.98490512564_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 64, matrix_height: 64 },
    TileMatrixDef { id: "7", scale_denominator: 4367830.1877243575_f64, cell_size: 1222.99245256282_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 128, matrix_height: 128 },
    TileMatrixDef { id: "8", scale_denominator: 2183915.0938621787_f64, cell_size: 611.49622628141_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 256, matrix_height: 256 },
    TileMatrixDef { id: "9", scale_denominator: 1091957.5469310894_f64, cell_size: 305.748113140705_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 512, matrix_height: 512 },
    TileMatrixDef { id: "10", scale_denominator: 545978.7734655447_f64, cell_size: 152.8740565703525_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 1024, matrix_height: 1024 },
    TileMatrixDef { id: "11", scale_denominator: 272989.38673277234_f64, cell_size: 76.43702828517625_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 2048, matrix_height: 2048 },
    TileMatrixDef { id: "12", scale_denominator: 136494.69336638617_f64, cell_size: 38.21851414258813_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 4096, matrix_height: 4096 },
    TileMatrixDef { id: "13", scale_denominator: 68247.34668319309_f64, cell_size: 19.109257071294063_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 8192, matrix_height: 8192 },
    TileMatrixDef { id: "14", scale_denominator: 34123.67334159654_f64, cell_size: 9.554628535647032_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 16384, matrix_height: 16384 },
    TileMatrixDef { id: "15", scale_denominator: 17061.83667079827_f64, cell_size: 4.777314267823516_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 32768, matrix_height: 32768 },
    TileMatrixDef { id: "16", scale_denominator: 8530.918335399136_f64, cell_size: 2.388657133911758_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 65536, matrix_height: 65536 },
    TileMatrixDef { id: "17", scale_denominator: 4265.459167699568_f64, cell_size: 1.194328566955879_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 131072, matrix_height: 131072 },
    TileMatrixDef { id: "18", scale_denominator: 2132.729583849784_f64, cell_size: 0.5971642834779395_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 262144, matrix_height: 262144 },
    TileMatrixDef { id: "19", scale_denominator: 1066.364791924892_f64, cell_size: 0.29858214173896974_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 524288, matrix_height: 524288 },
    TileMatrixDef { id: "20", scale_denominator: 533.182395962446_f64, cell_size: 0.14929107086948487_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 1048576, matrix_height: 1048576 },
    TileMatrixDef { id: "21", scale_denominator: 266.591197981223_f64, cell_size: 0.07464553543474244_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 2097152, matrix_height: 2097152 },
    TileMatrixDef { id: "22", scale_denominator: 133.2955989906115_f64, cell_size: 0.03732276771737122_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 4194304, matrix_height: 4194304 },
    TileMatrixDef { id: "23", scale_denominator: 66.64779949530575_f64, cell_size: 0.01866138385868561_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 8388608, matrix_height: 8388608 },
    TileMatrixDef { id: "24", scale_denominator: 33.323899747652874_f64, cell_size: 0.009330691929342804_f64, origin_x: -20037508.342789244_f64, origin_y: 20037508.342789244_f64, tile_width: 256, tile_height: 256, matrix_width: 16777216, matrix_height: 16777216 },
];

const MATRICES_WORLDCRS84QUAD: &[TileMatrixDef] = &[
    TileMatrixDef { id: "0", scale_denominator: 279541132.014358_f64, cell_size: 0.703125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 2, matrix_height: 1 },
    TileMatrixDef { id: "1", scale_denominator: 139770566.007179_f64, cell_size: 0.3515625_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 4, matrix_height: 2 },
    TileMatrixDef { id: "2", scale_denominator: 69885283.0035897_f64, cell_size: 0.17578125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 8, matrix_height: 4 },
    TileMatrixDef { id: "3", scale_denominator: 34942641.5017948_f64, cell_size: 0.087890625_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 16, matrix_height: 8 },
    TileMatrixDef { id: "4", scale_denominator: 17471320.7508974_f64, cell_size: 0.0439453125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 32, matrix_height: 16 },
    TileMatrixDef { id: "5", scale_denominator: 8735660.37544871_f64, cell_size: 0.02197265625_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 64, matrix_height: 32 },
    TileMatrixDef { id: "6", scale_denominator: 4367830.18772435_f64, cell_size: 0.010986328125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 128, matrix_height: 64 },
    TileMatrixDef { id: "7", scale_denominator: 2183915.09386217_f64, cell_size: 0.0054931640625_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 256, matrix_height: 128 },
    TileMatrixDef { id: "8", scale_denominator: 1091957.54693108_f64, cell_size: 0.00274658203125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 512, matrix_height: 256 },
    TileMatrixDef { id: "9", scale_denominator: 545978.773465544_f64, cell_size: 0.001373291015625_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 1024, matrix_height: 512 },
    TileMatrixDef { id: "10", scale_denominator: 272989.386732772_f64, cell_size: 0.0006866455078125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 2048, matrix_height: 1024 },
    TileMatrixDef { id: "11", scale_denominator: 136494.693366386_f64, cell_size: 0.00034332275390625_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 4096, matrix_height: 2048 },
    TileMatrixDef { id: "12", scale_denominator: 68247.346683193_f64, cell_size: 0.000171661376953125_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 8192, matrix_height: 4096 },
    TileMatrixDef { id: "13", scale_denominator: 34123.6733415964_f64, cell_size: 8.58306884765625e-05_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 16384, matrix_height: 8192 },
    TileMatrixDef { id: "14", scale_denominator: 17061.8366707982_f64, cell_size: 4.29153442382812e-05_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 32768, matrix_height: 16384 },
    TileMatrixDef { id: "15", scale_denominator: 8530.91833539913_f64, cell_size: 2.14576721191406e-05_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 65536, matrix_height: 32768 },
    TileMatrixDef { id: "16", scale_denominator: 4265.45916769956_f64, cell_size: 1.07288360595703e-05_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 131072, matrix_height: 65536 },
    TileMatrixDef { id: "17", scale_denominator: 2132.72958384978_f64, cell_size: 5.36441802978515e-06_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 262144, matrix_height: 131072 },
    TileMatrixDef { id: "18", scale_denominator: 1066.36479192489_f64, cell_size: 2.68220901489258e-06_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 524288, matrix_height: 262144 },
    TileMatrixDef { id: "19", scale_denominator: 533.182395962445_f64, cell_size: 1.34110450744629e-06_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 1048576, matrix_height: 524288 },
    TileMatrixDef { id: "20", scale_denominator: 266.591197981222_f64, cell_size: 6.7055225372314e-07_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 2097152, matrix_height: 1048576 },
    TileMatrixDef { id: "21", scale_denominator: 133.295598990611_f64, cell_size: 3.3527612686157e-07_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 4194304, matrix_height: 2097152 },
    TileMatrixDef { id: "22", scale_denominator: 66.6477994953056_f64, cell_size: 1.6763806343079e-07_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 8388608, matrix_height: 4194304 },
    TileMatrixDef { id: "23", scale_denominator: 33.3238997476528_f64, cell_size: 8.381903171539e-08_f64, origin_x: -180.0_f64, origin_y: 90.0_f64, tile_width: 256, tile_height: 256, matrix_width: 16777216, matrix_height: 8388608 },
];

const MATRICES_WORLDMERCATORWGS84QUAD: &[TileMatrixDef] = &[
    TileMatrixDef { id: "0", scale_denominator: 559082264.028717_f64, cell_size: 156543.033928041_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 1, matrix_height: 1 },
    TileMatrixDef { id: "1", scale_denominator: 279541132.014358_f64, cell_size: 78271.5169640204_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 2, matrix_height: 2 },
    TileMatrixDef { id: "2", scale_denominator: 139770566.007179_f64, cell_size: 39135.7584820102_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 4, matrix_height: 4 },
    TileMatrixDef { id: "3", scale_denominator: 69885283.0035897_f64, cell_size: 19567.8792410051_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 8, matrix_height: 8 },
    TileMatrixDef { id: "4", scale_denominator: 34942641.5017948_f64, cell_size: 9783.93962050256_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 16, matrix_height: 16 },
    TileMatrixDef { id: "5", scale_denominator: 17471320.7508974_f64, cell_size: 4891.96981025128_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 32, matrix_height: 32 },
    TileMatrixDef { id: "6", scale_denominator: 8735660.37544871_f64, cell_size: 2445.98490512564_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 64, matrix_height: 64 },
    TileMatrixDef { id: "7", scale_denominator: 4367830.18772435_f64, cell_size: 1222.99245256282_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 128, matrix_height: 128 },
    TileMatrixDef { id: "8", scale_denominator: 2183915.09386217_f64, cell_size: 611.49622628141_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 256, matrix_height: 256 },
    TileMatrixDef { id: "9", scale_denominator: 1091957.54693108_f64, cell_size: 305.748113140704_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 512, matrix_height: 512 },
    TileMatrixDef { id: "10", scale_denominator: 545978.773465544_f64, cell_size: 152.874056570352_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 1024, matrix_height: 1024 },
    TileMatrixDef { id: "11", scale_denominator: 272989.386732772_f64, cell_size: 76.4370282851762_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 2048, matrix_height: 2048 },
    TileMatrixDef { id: "12", scale_denominator: 136494.693366386_f64, cell_size: 38.2185141425881_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 4096, matrix_height: 4096 },
    TileMatrixDef { id: "13", scale_denominator: 68247.346683193_f64, cell_size: 19.109257071294_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 8192, matrix_height: 8192 },
    TileMatrixDef { id: "14", scale_denominator: 34123.6733415964_f64, cell_size: 9.55462853564703_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 16384, matrix_height: 16384 },
    TileMatrixDef { id: "15", scale_denominator: 17061.8366707982_f64, cell_size: 4.77731426782351_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 32768, matrix_height: 32768 },
    TileMatrixDef { id: "16", scale_denominator: 8530.91833539913_f64, cell_size: 2.38865713391175_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 65536, matrix_height: 65536 },
    TileMatrixDef { id: "17", scale_denominator: 4265.45916769956_f64, cell_size: 1.19432856695587_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 131072, matrix_height: 131072 },
    TileMatrixDef { id: "18", scale_denominator: 2132.72958384978_f64, cell_size: 0.597164283477939_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 262144, matrix_height: 262144 },
    TileMatrixDef { id: "19", scale_denominator: 1066.36479192489_f64, cell_size: 0.29858214173897_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 524288, matrix_height: 524288 },
    TileMatrixDef { id: "20", scale_denominator: 533.182395962445_f64, cell_size: 0.149291070869485_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 1048576, matrix_height: 1048576 },
    TileMatrixDef { id: "21", scale_denominator: 266.591197981222_f64, cell_size: 0.0746455354347424_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 2097152, matrix_height: 2097152 },
    TileMatrixDef { id: "22", scale_denominator: 133.295598990611_f64, cell_size: 0.0373227677173712_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 4194304, matrix_height: 4194304 },
    TileMatrixDef { id: "23", scale_denominator: 66.6477994953056_f64, cell_size: 0.0186613838586856_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 8388608, matrix_height: 8388608 },
    TileMatrixDef { id: "24", scale_denominator: 33.3238997476528_f64, cell_size: 0.0093306919293428_f64, origin_x: -20037508.3427892_f64, origin_y: 20037508.3427892_f64, tile_width: 256, tile_height: 256, matrix_width: 16777216, matrix_height: 16777216 },
];

pub const ALL_TMS: &[TmsDef] = &[
    TmsDef { id: "CDB1GlobalGrid", title: "CDB 1 Global Grid", crs_uri: "http://www.opengis.net/def/crs/EPSG/0/4326", epsg: 4326, ordered_axes: ["Lat", "Lon"], invert_axis: true, matrices: MATRICES_CDB1GLOBALGRID },
    TmsDef { id: "CanadianNAD83_LCC", title: "Lambert conformal conic NAD83 for Canada", crs_uri: "http://www.opengis.net/def/crs/EPSG/0/3978", epsg: 3978, ordered_axes: ["E", "N"], invert_axis: false, matrices: MATRICES_CANADIANNAD83_LCC },
    TmsDef { id: "EuropeanETRS89_LAEAQuad", title: "Lambert Azimuthal Equal Area ETRS89 for Europe", crs_uri: "http://www.opengis.net/def/crs/EPSG/0/3035", epsg: 3035, ordered_axes: ["Y", "X"], invert_axis: true, matrices: MATRICES_EUROPEANETRS89_LAEAQUAD },
    TmsDef { id: "GNOSISGlobalGrid", title: "GNOSIS Global Grid", crs_uri: "http://www.opengis.net/def/crs/EPSG/0/4326", epsg: 4326, ordered_axes: ["Lat", "Lon"], invert_axis: true, matrices: MATRICES_GNOSISGLOBALGRID },
    TmsDef { id: "LINZAntarticaMapTilegrid", title: "LINZ Antarctic Map Tile Grid (Ross Sea Region)", crs_uri: "urn:ogc:def:crs:EPSG::5482", epsg: 5482, ordered_axes: ["Y", "X"], invert_axis: true, matrices: MATRICES_LINZANTARTICAMAPTILEGRID },
    TmsDef { id: "NZTM2000Quad", title: "LINZ NZTM2000Quad Map Tile Grid", crs_uri: "urn:ogc:def:crs:EPSG::2193", epsg: 2193, ordered_axes: ["Y", "X"], invert_axis: true, matrices: MATRICES_NZTM2000QUAD },
    TmsDef { id: "UPSAntarcticWGS84Quad", title: "Universal Polar Stereographic WGS 84 Quad for Antarctic", crs_uri: "http://www.opengis.net/def/crs/EPSG/0/5042", epsg: 5042, ordered_axes: ["E", "N"], invert_axis: false, matrices: MATRICES_UPSANTARCTICWGS84QUAD },
    TmsDef { id: "UPSArcticWGS84Quad", title: "Universal Polar Stereographic WGS 84 Quad for Arctic", crs_uri: "http://www.opengis.net/def/crs/EPSG/0/5041", epsg: 5041, ordered_axes: ["E", "N"], invert_axis: false, matrices: MATRICES_UPSARCTICWGS84QUAD },
    TmsDef { id: "UTM31WGS84Quad", title: "Universal Transverse Mercator Zone 31 WGS84 Quad", crs_uri: "http://www.opengis.net/def/crs/EPSG/0/32631", epsg: 32631, ordered_axes: ["E", "N"], invert_axis: false, matrices: MATRICES_UTM31WGS84QUAD },
    TmsDef { id: "WGS1984Quad", title: "EPSG:4326 for the World", crs_uri: "http://www.opengis.net/def/crs/EPSG/0/4326", epsg: 4326, ordered_axes: ["Lat", "Lon"], invert_axis: true, matrices: MATRICES_WGS1984QUAD },
    TmsDef { id: "WebMercatorQuad", title: "Google Maps Compatible for the World", crs_uri: "http://www.opengis.net/def/crs/EPSG/0/3857", epsg: 3857, ordered_axes: ["X", "Y"], invert_axis: false, matrices: MATRICES_WEBMERCATORQUAD },
    TmsDef { id: "WorldCRS84Quad", title: "CRS84 for the World", crs_uri: "http://www.opengis.net/def/crs/OGC/1.3/CRS84", epsg: 4326, ordered_axes: ["Lon", "Lat"], invert_axis: false, matrices: MATRICES_WORLDCRS84QUAD },
    TmsDef { id: "WorldMercatorWGS84Quad", title: "World Mercator WGS84 (ellipsoid)", crs_uri: "http://www.opengis.net/def/crs/EPSG/0/3395", epsg: 3395, ordered_axes: ["E", "N"], invert_axis: false, matrices: MATRICES_WORLDMERCATORWGS84QUAD },
];

pub fn lookup_tms(id: &str) -> Option<&'static TmsDef> {
    ALL_TMS.iter().find(|t| t.id == id)
}

impl TmsDef {
    pub fn crs_model(&self) -> Result<CrsModel, EngineError> {
        CrsModel::from_epsg(self.epsg)
            .map_err(|e| EngineError::BadRequest(format!("unsupported TMS CRS EPSG:{}: {e}", self.epsg)))
    }

    pub fn matrix(&self, z: u8) -> Option<&'static TileMatrixDef> {
        let zs = z.to_string();
        self.matrices.iter().find(|m| m.id == zs)
    }

    /// Return `[min_x, min_y, max_x, max_y]` in the TMS's native `(easting/lon, northing/lat)` CRS,
    /// expanded by `buffer` pixels on all 4 sides.
    pub fn tile_bounds(&self, z: u8, x: u32, y: u32, buffer: u32, tile_size: u32) -> Option<[f64; 4]> {
        let m = self.matrix(z)?;
        let span_x = m.cell_size * (m.tile_width as f64);
        let span_y = m.cell_size * (m.tile_height as f64);
        let pad_x = span_x * (buffer as f64 / tile_size.max(1) as f64);
        let pad_y = span_y * (buffer as f64 / tile_size.max(1) as f64);
        let min_x = m.origin_x + (x as f64) * span_x - pad_x;
        let max_x = min_x + span_x + 2.0 * pad_x;
        let max_y = m.origin_y - (y as f64) * span_y + pad_y;
        let min_y = max_y - span_y - 2.0 * pad_y;
        Some([min_x, min_y, max_x, max_y])
    }

    /// Compute `(minzoom, maxzoom)` for a COG in this TileMatrixSet.
    pub fn zoom_range_for_cog(
        &self,
        src_crs: &CrsModel,
        pixel_to_crs: &Affine2D,
        width: u32,
        height: u32,
        ov_factor: f64,
    ) -> (u8, u8) {
        if self.id == "WebMercatorQuad" {
            return src_crs.web_mercator_zoom_range(pixel_to_crs, width, height, ov_factor);
        }
        if self.id == "WorldCRS84Quad" || self.id == "WGS1984Quad" {
            return src_crs.world_crs84_zoom_range(pixel_to_crs, width, height, ov_factor);
        }
        let Ok(dst_crs) = self.crs_model() else {
            return (0, 22);
        };
        let (x0, y0) = pixel_to_crs.forward(0.0, 0.0);
        let (x1, y1) = pixel_to_crs.forward(width as f64, height as f64);
        let (lo0, la0) = src_crs.to_lonlat_deg(x0, y0);
        let (lo1, la1) = src_crs.to_lonlat_deg(x1, y1);
        let (dx0, dy0) = dst_crs.from_lonlat_deg(lo0, la0);
        let (dx1, dy1) = dst_crs.from_lonlat_deg(lo1, la1);
        let diag_src = ((width as f64).hypot(height as f64)).max(1.0);
        let diag_dst = (dx1 - dx0).hypot(dy1 - dy0);
        let res = if diag_dst.is_finite() && diag_dst > 0.0 {
            diag_dst / diag_src
        } else {
            return (0, 12);
        };
        let mut max_z = 0u8;
        for m in self.matrices {
            if let Ok(z) = m.id.parse::<u8>() {
                max_z = z;
                if m.cell_size <= res * 1.1 {
                    break;
                }
            }
        }
        let min_res = res * ov_factor.max(1.0);
        let mut min_z = 0u8;
        for m in self.matrices {
            if let Ok(z) = m.id.parse::<u8>() {
                if m.cell_size >= min_res {
                    min_z = z;
                }
            }
        }
        (min_z.min(max_z), max_z)
    }

    pub fn to_ogc_json(&self) -> serde_json::Value {
        let matrices: Vec<serde_json::Value> = self
            .matrices
            .iter()
            .map(|m| {
                let origin = if self.invert_axis {
                    [m.origin_y, m.origin_x]
                } else {
                    [m.origin_x, m.origin_y]
                };
                serde_json::json!({
                    "id": m.id,
                    "scaleDenominator": m.scale_denominator,
                    "cellSize": m.cell_size,
                    "cornerOfOrigin": "topLeft",
                    "pointOfOrigin": origin,
                    "tileWidth": m.tile_width,
                    "tileHeight": m.tile_height,
                    "matrixWidth": m.matrix_width,
                    "matrixHeight": m.matrix_height,
                })
            })
            .collect();
        serde_json::json!({
            "id": self.id,
            "title": self.title,
            "crs": self.crs_uri,
            "orderedAxes": self.ordered_axes,
            "tileMatrices": matrices,
        })
    }
}

pub fn list_all_tms_json(host: &str) -> serde_json::Value {
    let origin = if host.starts_with("http://") || host.starts_with("https://") {
        host.trim_end_matches('/').to_string()
    } else {
        format!("http://{}", host.trim_end_matches('/'))
    };
    let sets: Vec<serde_json::Value> = ALL_TMS
        .iter()
        .map(|t| {
            serde_json::json!({
                "id": t.id,
                "title": t.title,
                "links": [{
                    "href": format!("{origin}/tileMatrixSets/{}", t.id),
                    "rel": "http://www.opengis.net/def/rel/ogc/1.0/tiling-scheme",
                    "type": "application/json",
                }]
            })
        })
        .collect();
    serde_json::json!({ "tileMatrixSets": sets })
}

pub fn get_tms_def_json(id: &str) -> Result<serde_json::Value, EngineError> {
    lookup_tms(id)
        .map(TmsDef::to_ogc_json)
        .ok_or_else(|| EngineError::NotFound(format!("unknown TileMatrixSet '{id}'")))
}

