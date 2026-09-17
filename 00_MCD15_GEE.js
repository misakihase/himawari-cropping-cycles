/**
 * Export MCD15A2H 8-day FPAR over Southeast Asia from Google Earth Engine.
 *
 * One GeoTIFF is exported per 8-day compositing period, named
 *   MCD15_FPAR_<YYYYMMDD>.tif
 * where <YYYYMMDD> is the start date of the period. This is the naming
 * expected by 11_MCD15_FPAR_preprocess.py.
 *
 * Quality screening follows the Methods: only retrievals produced by the main
 * radiative-transfer algorithm are kept (SCF_QC 0 or 1), and pixels flagged for
 * snow/ice, internal cloud or cloud shadow in FparExtra_QC are discarded.
 *
 * Set PRODUCT to 'MCD15A2H' for the combined Terra+Aqua product, or to
 * 'MOD15A2H' for Terra only. This must match what the manuscript states.
 *
 * Run once per year by editing YEAR, then start the tasks from the Tasks tab.
 */
 
// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------
var YEAR = 2016;
var PRODUCT = 'MCD15A2H';          // 'MCD15A2H' (Terra+Aqua) or 'MOD15A2H' (Terra)
var COLLECTION = 'MODIS/061/' + PRODUCT;
 
var REGION = ee.Geometry.Rectangle([92.0, -9.0, 127.0, 23.0]);
var SCALE = 500;                   // native FPAR resolution, m
var DRIVE_FOLDER = 'GEE_MCD15_FPAR_8day';
var FPAR_SCALE_FACTOR = 0.01;      // stored integer -> fraction
 
// ---------------------------------------------------------------------------
// Quality masking
// ---------------------------------------------------------------------------
// FparLai_QC bits 5-7 hold SCF_QC. Values 0 and 1 indicate the main
// radiative-transfer algorithm succeeded, with or without saturation.
function mainAlgorithmMask(image) {
  var scf = image.select('FparLai_QC').rightShift(5).bitwiseAnd(7);
  return scf.lte(1);
}
 
// FparExtra_QC: bit 2 snow/ice, bit 3 aerosol, bit 4 cirrus,
// bit 5 internal cloud, bit 6 cloud shadow.
function clearSkyMask(image) {
  var extra = image.select('FparExtra_QC');
  var snow = extra.rightShift(2).bitwiseAnd(1);
  var cirrus = extra.rightShift(4).bitwiseAnd(1);
  var cloud = extra.rightShift(5).bitwiseAnd(1);
  var shadow = extra.rightShift(6).bitwiseAnd(1);
  return snow.add(cirrus).add(cloud).add(shadow).eq(0);
}
 
function screenFpar(image) {
  var keep = mainAlgorithmMask(image).and(clearSkyMask(image));
  return image.select('Fpar_500m')
              .updateMask(keep)
              .multiply(FPAR_SCALE_FACTOR)
              .rename('FPAR')
              .copyProperties(image, ['system:time_start']);
}
 
// ---------------------------------------------------------------------------
// Export one image per 8-day composite
// ---------------------------------------------------------------------------
var collection = ee.ImageCollection(COLLECTION)
  .filterDate(ee.Date.fromYMD(YEAR, 1, 1), ee.Date.fromYMD(YEAR + 1, 1, 1))
  .filterBounds(REGION)
  .map(screenFpar);
 
var imageList = collection.toList(collection.size());
var n = collection.size().getInfo();
print('Compositing periods found for ' + YEAR + ': ' + n);
 
for (var i = 0; i < n; i++) {
  var image = ee.Image(imageList.get(i));
  var date = ee.Date(image.get('system:time_start')).format('YYYYMMdd').getInfo();
 
  Export.image.toDrive({
    image: image.clip(REGION).unmask(-9999),
    description: 'MCD15_FPAR_' + date,
    fileNamePrefix: 'MCD15_FPAR_' + date,
    folder: DRIVE_FOLDER,
    scale: SCALE,
    region: REGION,
    crs: 'EPSG:4326',
    fileFormat: 'GeoTIFF',
    maxPixels: 1e13
  });
}
 
print('Export tasks created. Start them from the Tasks tab.');
 