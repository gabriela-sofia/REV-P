// SUSC-16A Sentinel-1 flood mapping stub (review-only).
// Requires authenticated Earth Engine environment. This stub does not run in
// Codex by itself and must not persist raster outputs into the public repo.
var target = {
  event_window_id: 'S16AWIN_EXAMPLE',
  aoi_bbox: [-35.05, -8.20, -34.80, -7.90],
  pre_start: 'YYYY-MM-DD',
  pre_end: 'YYYY-MM-DD',
  post_start: 'YYYY-MM-DD',
  post_end: 'YYYY-MM-DD'
};
var aoi = ee.Geometry.Rectangle(target.aoi_bbox);
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));
var pre = s1.filterDate(target.pre_start, target.pre_end).median();
var post = s1.filterDate(target.post_start, target.post_end).median();
var change = post.select('VV').subtract(pre.select('VV'))
  .add(post.select('VH').subtract(pre.select('VH'))).rename('vv_vh_change');
var candidateFlood = change.lt(-2.0).selfMask();
// Export candidateFlood polygonization to a local-only destination after manual review.
